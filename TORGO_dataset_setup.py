import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import soundfile as sf
import kaggle
import torch
import torchlibrosa
import torchaudio

def stereotomono(path):
    wf, sr = torchaudio.load(path)
    if (wf.shape[0]>1):
        wf = torch.mean(wf, dim=0, keepdim=True)
    return wf

#adapted from audio and PANN
def logmelspec(path):
    window_size = 512
    hop_size = 160
    fmin = 50
    fmax = 8000
    ref = 1.0
    amin = 1e-10
    mel_bins = 64
    sample_rate = 16000
    spectrogram = torchlibrosa.stft.Spectrogram( n_fft=window_size, win_length=window_size, hop_length=hop_size)
    mel_spec = torchlibrosa.stft.LogmelFilterBank(sample_rate,
            fmin=fmin,
            fmax=fmax,
            n_mels=mel_bins,
            n_fft=window_size,
            ref=ref,
            amin=amin,)
    log_mel_spec = mel_spec(spectrogram(path))
    return log_mel_spec


def expand_center(spectrogram):
    length = spectrogram.shape[0]            
    pad = 301 - length
    if pad <= 0:                             
        return spectrogram
    left = pad // 2
    right = pad - left
    pad_width = [(0, 0)] * spectrogram.ndim
    pad_width[1] = (left, right)             
    return np.pad(spectrogram, pad_width, mode="edge")

def center_crop(spectrogram):
    length = spectrogram.shape[1]            
    start = (length - 301) // 2
    end = start + 301
    return spectrogram[:,start:end, :] 

def clean():
    paths = ["torgo/1", "torgo/0"]
    for path in paths:
        for files in os.listdir(path):
            os.remove(os.path.join(path,files))

def download():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("iamhungundji/dysarthria-detection", path = None, unzip=True)



def folderSetup():
    os.makedirs("torgo/0", exist_ok=True)
    os.makedirs("torgo/1", exist_ok=True)

    kaggle_data = "torgo_data"
    
    for subgroups in os.listdir(kaggle_data):
        dest = "torgo"
        subgroups_path = os.path.join(kaggle_data, subgroups)
        if not os.path.isdir(subgroups_path):
            continue

        if subgroups.startswith("dysarthria"):
            dest = os.path.join(dest, "1")
        else:
            dest = os.path.join(dest, "0")
        
        for files in os.listdir(subgroups_path):
            file_path = os.path.join(subgroups_path, files)
            try: 
                with sf.SoundFile(file_path) as f:
                    if (len(f) / f.samplerate) < 0.3:
                        f.close()
                        continue
            except Exception as e:
                continue
            wf = stereotomono(file_path)
            log_mel_spec = logmelspec(wf).squeeze(1)
            if log_mel_spec.shape[1] > 301:
                log_mel_spec = center_crop(log_mel_spec)
            elif log_mel_spec.shape[1] < 301:
                log_mel_spec = expand_center(log_mel_spec)
            np.save(os.path.join(dest, files[:-4] ), log_mel_spec)

def main():
    #clean()
    download()
    folderSetup()


main()

    
    

