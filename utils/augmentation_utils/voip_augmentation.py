import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
import io
import random
import ffmpeg
import soundfile as sf
import librosa
############################
# 1. Packet Loss Simulation
############################
def simulate_packet_loss(waveform, sr, loss_rate=0.05, frame_ms=20):
    num_samples = waveform.shape[1]
    frame_len = int(sr * frame_ms / 1000)
    num_frames = int(np.ceil(num_samples / frame_len))  # use ceil to include last partial frame
    
    # Create mask of which frames to keep/drop
    mask = np.ones(num_frames, dtype=bool)
    drop_count = int(loss_rate * num_frames)
    if drop_count > 0:
        drop_indices = np.random.choice(num_frames, size=drop_count, replace=False)
        mask[drop_indices] = False

    frames = []
    for i in range(num_frames):
        start = i * frame_len
        end = min(start + frame_len, num_samples)  # handle last frame correctly
        frame = waveform[:, start:end]
        if not mask[i]:
            frame = torch.zeros_like(frame)
        frames.append(frame)    
    
    return torch.cat(frames, dim=1), {"type": "packet_loss", "loss_rate": loss_rate, "frame_ms": frame_ms}


############################
# 2. Jitter Simulation
############################
def simulate_jitter(waveform, sr, jitter_prob=0.1, max_shift_ms=10):
    frame_len = int(sr * max_shift_ms / 1000)
    shift = int(np.random.uniform(-frame_len, frame_len))
    if np.random.rand() < jitter_prob:
        return torch.roll(waveform, shifts=shift, dims=1), {"type": "jitter", "shift_samples": shift}
    return waveform, {"type": "jitter", "applied": False}

def apply_random_codec_in_memory(
    waveform: np.ndarray, sr: int, target_sr: int = 16000
):
    codecs = [
        ("G.711 µ-law", "pcm_mulaw", ["-ar", "8000"], "wav"),
        ("G.711 A-law", "pcm_alaw", ["-ar", "8000"], "wav"),
        ("G.722", "g722", ["-ar", "16000"], "wav"),
        ("Opus", "libopus", ["-b:a", "16k"], "ogg"),
    ]

    codec_name, codec_flag, extra_args, ext = random.choice(codecs)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="wav")
    buf.seek(0)

    # build pipeline
    process = ffmpeg.input("pipe:", format="wav")

    # encode to codec, then decode back to PCM
    if codec_name == "G.722":
        # encode -> decode explicitly back to PCM 16-bit
        process = process.output(
            "pipe:", format="wav", acodec="pcm_s16le",
            ar="16000"  # G.722 standard sample rate
        )
    else:
        process = process.output(
            "pipe:", format="wav", acodec="pcm_s16le",
            **{extra_args[i].lstrip("-"): extra_args[i+1]
               for i in range(0, len(extra_args), 2)}
        )

    out, _ = process.overwrite_output().global_args("-loglevel", "error").run(
        input=buf.read(), capture_stdout=True, capture_stderr=True
    )

    audio_bytes = io.BytesIO(out)
    aug_waveform, aug_sr = sf.read(audio_bytes, dtype="float32")

    if aug_sr != target_sr:
        aug_waveform = librosa.resample(aug_waveform, orig_sr=aug_sr, target_sr=target_sr)
        aug_sr = target_sr

    return aug_waveform, {
        "codec": codec_name,
        "ffmpeg_flag": codec_flag,
        "params": extra_args,
        "sr": aug_sr,
    }
    


############################
# 4. Bandwidth Limitation
############################
def simulate_bandwidth(waveform, sr, lowpass_hz=3400):
    return torchaudio.functional.lowpass_biquad(waveform, sr, cutoff_freq=lowpass_hz), {"type": "bandwidth", "cutoff_hz": lowpass_hz}


############################
# 5. Background Noise
############################
def add_noise(waveform, snr_db=20):
    rms_signal = torch.sqrt(torch.mean(waveform**2))
    rms_noise = rms_signal / (10**(snr_db/20))
    noise = torch.randn_like(waveform) * rms_noise
    return waveform + noise, {"type": "noise", "snr_db": snr_db}


############################
# Full VoIP Augmentation Pipeline
############################
def voip_augment(waveform, sr):
    l = waveform.shape[1]
    metadata = []

    if random.random() < 0.5:
        snr = np.random.randint(10, 30)
        waveform, md = add_noise(waveform, snr_db=snr)
        if l != waveform.shape[1]:
            print('noise')
        metadata.append(md)

    if random.random() < 0.5:
        cutoff = random.choice([3000, 3400, 4000])
        waveform, md = simulate_bandwidth(waveform, sr, lowpass_hz=cutoff)
        if l != waveform.shape[1]:
            print('bandwidth')
        metadata.append(md)

    if random.random() < 0.5:
        waveform, md = apply_random_codec_in_memory(waveform.squeeze(0), sr)
        waveform = torch.tensor(waveform).unsqueeze(0)
        if l != waveform.shape[1]:
            print('codec')
        metadata.append(md)

    if random.random() < 0.5:
        waveform, md = simulate_packet_loss(waveform, sr, loss_rate=np.random.uniform(0.01, 0.1))
        if l != waveform.shape[1]:
            print('packet loss', waveform.shape, l)
        metadata.append(md)

    if random.random() < 0.5:
        waveform, md = simulate_jitter(waveform, sr, jitter_prob=0.5, max_shift_ms=np.random.randint(5, 20))
        if l != waveform.shape[1]:
            print('jitter', waveform.shape, l)
        metadata.append(md)

    return waveform, metadata