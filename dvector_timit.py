#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn
"""

import glob
import librosa
import numpy as np
import os
import torch

from datetime import timedelta, datetime
from librosa.core import get_duration
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
from utils import get_device, count_parameters
from tqdm import tqdm

USE_CUDA, DEVICE = get_device()


def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        # Diarization window size is 240 ms - we partition into non-overlapping segments
        # of 400ms max length
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
    return avg_embeddings

"""
Raw data format:

/data4/shuyang/TIMIT/data/lisa/data/timit/raw/TIMIT/
    TRAIN
        DR<x>
            <speaker>
                <segment>.WAV
            ...
        ...
    TEST
        ...

Usage:
    python3 -u dvector_timit.py
"""
if __name__=="__main__":
    import glob
    import os
    import argparse
    from collections import defaultdict

    # Set up parser
    parser = argparse.ArgumentParser(
        description='Train speech embeddings.')
    parser.add_argument(
        '--data-dir', type=str,
        default='/data4/shuyang/TIMIT/data/lisa/data/timit/raw/TIMIT',
        help='Audio data root directory')
    parser.add_argument(
        '--model-dir', type=str,
        default='./speech_id_checkpoint',
        help='Model checkpoint directory')
    parser.add_argument(
        '--out-dir', type=str,
        default='/data4/shuyang/TIMIT_dvec',
        help='Output directory',
    )
    args = parser.parse_args()

    # Parse arguments
    data_dir = args.data_dir
    model_dir = args.model_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load latest model
    latest_model_path = max(glob.glob(os.path.join(model_dir, '*')), key=os.path.getctime)
    print('Latest model from: {}'.format(latest_model_path))
    embedder_net = SpeechEmbedder().to(DEVICE)
    embedder_net.load_state_dict(torch.load(latest_model_path))
    embedder_net.eval()
    print('MODEL with {:,} parameters'.format(
        count_parameters(embedder_net)
    ))
    print(embedder_net)

    # Get audio
    glob_path = os.path.join(
        data_dir,
        '*',  # Split
        '*',  # Disk (DR<X>)
        '*',  # Speaker folder
        '*.WAV',  # Audio files
    )
    audio_files = glob.glob(glob_path)
    n_files = len(audio_files)

    all_speakers = []

    # Structure: split : sequence(s)
    sequences = defaultdict(list)
    cluster_ids = defaultdict(list)

    with torch.no_grad():
        debug = True
        count = 0
        for fpath in tqdm(audio_files, total=n_files):
            start = datetime.now()

            # Get speaker ID
            split, _, speaker, _ = fpath.split(os.sep)[:-4]
            if speaker in all_speakers:
                speaker_id = all_speakers.index(speaker)
            else:
                speaker_id = len(all_speakers)
                all_speakers.append(speaker)

            # Get duration
            if debug:
                print('\n==== DEBUG ====')
                print(fpath)
            try:
                duration = str(timedelta(seconds=get_duration(filename=fpath)))
                if debug:
                    print('File duration: {}'.format(duration))
            except:
                print('UNABLE TO GET DURATION')
                raise

            # Chunk into segments with speech audio
            times, segs = VAD_chunk(2, fpath)
            if segs == []:
                print('No voice activity detected')
                continue
            if debug:
                print('{} - {:,} segments'.format(
                    datetime.now() - start, len(segs)
                ))

            # Short-term Fourier Transform
            concat_seg = concat_segs(times, segs)
            if debug:
                print('{} - Concatenated segments'.format(
                    datetime.now() - start
                ))
            STFT_frames = get_STFTs(concat_seg)
            if debug:
                print('{} - Got STFT frames'.format(
                    datetime.now() - start
                ))
            STFT_frames = np.stack(STFT_frames, axis=2)

            # Batch-first
            STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).to(DEVICE)

            # Create d-vectors
            embeddings = embedder_net(STFT_frames)
            if debug:
                print('{} - Embedded STFT frames'.format(
                    datetime.now() - start
                ))

            # Align them, collapsing frames into partitions
            aligned_embeddings = align_embeddings(embeddings.cpu().detach().numpy())
            if debug:
                print('{:,} partitions'.format(
                    len(aligned_embeddings)
                ))

            # Store into sequences and cluster labels
            sequences[split].append(aligned_embeddings)
            cluster_ids[split].extend([speaker_id] * len(aligned_embeddings))
            
            if debug:
                print()
                debug = False

            count = count + 1

            if count % 1000 == 0:
                print('Processed {:,}/{:,} files'.format(count, n_files))
                debug = True

for split, seq in sequences.items():
    # Save sequence
    seq_loc = os.path.join(out_dir, '{}_sequence.npy'.format(split))
    seq_save = np.concatenate(seq, axis=0)
    np.save(seq_loc, seq_save)
    print('{}: {} shape sequence ({:,.3f} MB) at {}'.format(
        split, seq_save.shape, os.path.getsize(seq_loc) / 1024 / 1024, seq_loc
    ))

    # Save cluster IDs
    clusters_loc = os.path.join(out_dir, '{}_cluster_id.npy'.format(split))
    clusters_save = np.asarray(cluster_ids[split])
    np.save(clusters_loc, clusters_save)
    print('{}: {} shape cluster IDs ({:,.3f} MB) at {}'.format(
        split, clusters_save.shape, os.path.getsize(clusters_loc) / 1024 / 1024, clusters_loc
    ))

print('\n\n== DONE ==')
