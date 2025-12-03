import os
import argparse
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from joblib import Parallel, delayed
from audiomentations import Compose, TimeStretch, PitchShift, LowPassFilter, HighPassFilter, BandPassFilter, TimeMask, BitCrush, Aliasing
from tqdm import tqdm
import uuid
import random

def generate_transformations(sr):
    return [
        ("lowpass", LowPassFilter(min_cutoff_freq=200, max_cutoff_freq=4000, p=1.0)),
        ("highpass", HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=4000, p=1.0)),
        ("bandpass", BandPassFilter(min_center_freq=500, max_center_freq=3000, p=1.0)),
        ("pitchshift", PitchShift(min_semitones=-4, max_semitones=4, p=1.0)),
        ("timemask", TimeMask(min_band_part=0.1, max_band_part=0.3, p=1.0)),
        ("bitcrush", BitCrush(min_bit_depth=4, max_bit_depth=8, p=1.0))
    ]

def perturb(samples, sampling_rate):

    transforms = generate_transformations(sampling_rate)
    aug_name, augmenter = random.choice(transforms)

    augmented = augmenter(samples=samples, sample_rate=sampling_rate)

    return augmented
