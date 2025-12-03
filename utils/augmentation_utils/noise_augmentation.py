import os
import argparse
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from joblib import Parallel, delayed

import os
import glob
import random
import torch
import numpy
import librosa
from scipy import signal



class NoiseDataAugmentation:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.RIR_PATH = '/data/tools/RIRS_NOISES'
        self.MUSAN_PATH = '/data/tools/musan'
        self.ESC50_PATH = '/data/tools/ESC-50-master/audio'

        self.noisesnr = {
            "noise": [0, 15],
            "speech": [13, 20],
            "music": [5, 15],
        }
        self.numnoise = {
            "noise": [1, 1],
            "speech": [3, 8],
            "music": [1, 1],
        }

        # Load RIR files
        self.rir_files = glob.glob(os.path.join(self.RIR_PATH, "*/*/*/*.wav"))

        # Load MUSAN files grouped by category
        self.noiselist = {"noise": [], "speech": [], "music": []}
        musan_files = glob.glob(os.path.join(self.MUSAN_PATH, "*/*/*.wav"))
        for file in musan_files:
            category = file.split("/")[-3]
            if category in self.noiselist:
                self.noiselist[category].append(file)

        # Add ESC-50 files only to 'noise' category
        esc_files = glob.glob(os.path.join(self.ESC50_PATH, "*.wav"))
        self.noiselist["noise"].extend(esc_files)

    def _augment(self, waveform, emphasis="original"):
        if emphasis == "original":
            pass
        elif emphasis == "reverb":
            waveform, noise_type, noise_file = self.add_reverb(waveform)
        elif emphasis == "noise":
            waveform, noise_type, noise_file = self.add_noise(waveform)
        else:
            raise ValueError(f"Unsupported emphasis type: {emphasis}")
        return waveform, noise_type, noise_file

    def add_reverb(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = librosa.load(rir_file, sr=self.sampling_rate)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        result = signal.convolve(audio, rir, mode="full")[: audio.shape[0]]
        return result, 'reverb', rir_file

    def add_noise(self, audio):
        # Randomly choose noise type
        noise_type = random.choice(["noise", "speech", "music"])
        audio_db = 10 * numpy.log10(numpy.mean(audio**2) + 1e-4)

        noise_file = random.choice(self.noiselist[noise_type])
        noise, sr = librosa.load(noise_file, sr=self.sampling_rate)

        if noise.shape[0] <= audio.shape[0]:
            noise = numpy.pad(noise, (0, audio.shape[0] - noise.shape[0]), "wrap")
        else:
            noise = noise[: audio.shape[0]]

        noise_db = 10 * numpy.log10(numpy.mean(noise**2) + 1e-4)
        snr = random.uniform(*self.noisesnr[noise_type])
        scaled_noise = numpy.sqrt(10 ** ((audio_db - noise_db - snr) / 10)) * noise
        return audio + scaled_noise, noise_type, noise_file

    def augment(self, waveform, sr):
        emphasis = random.choice(["noise", "reverb"])
        augmented, noise_type, noise_file = self._augment(waveform, emphasis=emphasis)

        return augmented

