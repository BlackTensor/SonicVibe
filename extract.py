# -*- coding: utf-8 -*-
"""
Module 1: Audio Feature Engineering
====================================
Extracts a 28-dimensional mean-pooled feature vector from a 60-second
audio clip using Librosa. Optimized for CPU-only execution.

Features extracted:
    - MFCCs (1-13):       13 dims  → Timbral texture
    - Spectral Centroid:   1 dim   → Brightness
    - Chroma (12 bins):   12 dims  → Harmonic content / mood
    - Tempo (BPM):         1 dim   → Rhythmic energy
    - RMS Energy:          1 dim   → Volume / intensity
    ─────────────────────────────────────
    Total:                28 dims
"""

import numpy as np
import librosa


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION = 60  # seconds
N_MFCC = 13


def get_feature_names() -> list[str]:
    """Return ordered list of feature names matching the feature vector layout."""
    names = [f"mfcc_{i+1}" for i in range(N_MFCC)]
    names.append("spectral_centroid")
    names += [f"chroma_{i+1}" for i in range(12)]
    names.append("tempo")
    names.append("rms_energy")
    return names


def extract_features(file_path: str) -> np.ndarray:
    """
    Extract a 28-dimensional feature vector from an audio file.

    Parameters
    ----------
    file_path : str
        Path to a WAV or MP3 file.

    Returns
    -------
    np.ndarray
        A 1-D array of shape (28,) containing the mean-pooled features.
    """
    # Load audio -- mono, resampled, truncated to 60s max
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    # Note: No zero-padding needed. Mean-pooling normalizes across time,
    # so shorter clips produce valid feature vectors without silent padding.

    features = []

    # ── MFCCs (13 coefficients, mean-pooled over time) ──
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_means = np.mean(mfccs, axis=1)  # shape: (13,)
    features.extend(mfcc_means)

    # ── Spectral Centroid (mean-pooled) ──
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # -- Chroma Features (12 bins, mean-pooled) --
    # Pass tuning=0 to avoid estimate_tuning -> piptrack -> numba gufunc crash
    # on Python 3.14. Pre-compute the power spectrogram for efficiency.
    S = np.abs(librosa.stft(y)) ** 2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, tuning=0)
    chroma_means = np.mean(chroma, axis=1)  # shape: (12,)
    features.extend(chroma_means)

    # ── Tempo (BPM) ──
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa >= 0.10 returns tempo as an array
    tempo_val = float(np.atleast_1d(tempo)[0])
    features.append(tempo_val)

    # ── RMS Energy (mean-pooled) ──
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    return np.array(features, dtype=np.float64)


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Feature names:", get_feature_names())
    print(f"Total features: {len(get_feature_names())}")

    # Generate a synthetic sine-wave test
    print("\n-- Synthetic test (440 Hz sine, 60s) --")
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

    import soundfile as sf, tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "_aura_test.wav")
    sf.write(tmp_path, sine_wave, SAMPLE_RATE)

    feats = extract_features(tmp_path)
    print(f"Feature vector shape: {feats.shape}")
    for name, val in zip(get_feature_names(), feats):
        print(f"  {name:>20s}: {val:10.4f}")

    os.remove(tmp_path)
    print("\n[OK] extract.py self-test passed.")
