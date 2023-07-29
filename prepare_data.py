import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm
from glob import glob
import logging
from random import shuffle
from pathlib import Path
import kvc_utils
import torch
import torchaudio
from torchaudio.functional import resample

def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", f"hubert_soft").cuda().eval()

    print("Generating out_dir if not exist")
    if(not os.path.isdir(args.out_dir)):
        os.mkdir(args.out_dir)
    else:
        print(f"""out dir path = "{args.out_dir}" already exists, processing can generate errors!""")
    
    sampling_rate = args.config.data['sampling_rate']
    hop_length = args.config.data['hop_length']

    print(f"Processing {len(os.listdir(args.in_dir))} files at {args.in_dir}")
    for in_path in tqdm(os.listdir(args.in_dir)):
        if(args.extension in in_path):
            in_path = os.path.join(args.in_dir, in_path)
            out_path = os.path.join(args.out_dir, in_path)

            w_path = out_path 
            wav, sr = librosa.load(in_path, sr=None)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = 0.98 * wav / peak
            wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
            wav2 /= max(wav2.max(), -wav2.min())
            wavfile.write(
                w_path,
                sampling_rate,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
            )

            c_path = out_path + '.soft.pt'
            if not os.path.exists(c_path):
                wav16k = resample(wav, sr, 16000)
                wav16k = wav.unsqueeze(0).cuda()

                with torch.inference_mode():
                    units = hubert.units(wav16k)

                torch.save(units.permute(0,2,1).cpu(), c_path)

            f0_path = out_path + ".f0.npy"
            if not os.path.exists(f0_path):
                f0 = kvc_utils.compute_f0_dio(wav2, sampling_rate=sampling_rate, hop_length=hop_length)
                np.save(f0_path, f0)

            energy_path = out_path + ".energy.npy"
            if not os.path.exists(energy_path):
                energy = kvc_utils.compute_energy(wav2, sampling_rate=sampling_rate, hop_length=hop_length)
                np.save(energy_path, energy)

    print("Generating training files...")
    train = []
    val = []
    test = []
    wavs = []
    for wav_file in tqdm(os.listdir(args.out_dir)):
        wavs.append(os.path.join(args.out_dir, wav_file))

    shuffle(wavs)
    train += wavs[2:-2]
    val += wavs[:2]
    test += wavs[-2:]

    shuffle(train)
    shuffle(val)
    shuffle(test)
    
    train_list = f'./dataset/train_list_{args.model}'
    val_list = f'./dataset/val_list_{args.model}'
    test_list = f'./dataset/test_list_{args.model}'

    print("Writing", train_list)
    with open(train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")
        
    print("Writing", val_list)
    with open(val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")
            
    print("Writing", test_list)
    with open(test_list, "w") as f:
        for fname in tqdm(test):
            wavpath = fname
            f.write(wavpath + "\n")


    print("Data preprocessing is complete, you can start training now!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "--in_dir", 
        type=str, 
        default="./dataset_raw", 
        help="path to source dir")
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="./dataset", 
        help="path to target dir")
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".wav",
        type=str,
    )
    parser.add_argument(
        '-m', 
        '--model', 
        type=str, 
        required=True,
        help='Model name'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="./configs/quickvc.json",
        help='JSON file for configuration')
    
    args = parser.parse_args()
    encode_dataset(args)

