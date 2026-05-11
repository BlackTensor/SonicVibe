"""
Module 2: The "Vibe" Classifier
================================
Trains a Random Forest classifier on synthetic audio-feature data to
predict one of 5 Emotional Vibes:

    ┌─────────────────────────────────────────────────────────┐
    │  Vibe                │ Label        │ Colour Signature   │
    ├─────────────────────┼──────────────┼────────────────────┤
    │  Obsidian              │ melancholic  │ Low energy, dark   │
    │  Solaris       │ high_energy  │ Fast, bright       │
    │  Emerald             │ ambient      │ Calm, textured     │
    │  Crimson             │ aggressive   │ Loud, harsh        │
    │  Amber              │ nostalgic    │ Warm, mid-tempo    │
    └─────────────────────────────────────────────────────────┘

The model is a Random Forest (100 trees) saved via joblib.
A synthetic dataset of 100 samples (20 per class) is generated from
hand-crafted Gaussian profiles that encode domain knowledge about how
each audio feature correlates with emotional perception.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from extract import get_feature_names, N_MFCC


# ──────────────────────────────────────────────
# Vibe definitions
# ──────────────────────────────────────────────
VIBE_LABELS = ["melancholic", "high_energy", "ambient", "aggressive", "nostalgic"]

VIBE_DISPLAY = {
    "melancholic": "Obsidian / Melancholic",
    "high_energy": "Solaris / High Energy",
    "ambient":     "Emerald / Ambient",
    "aggressive":  "Crimson / Aggressive",
    "nostalgic":   "Amber / Nostalgic",
}

VIBE_COLORS = {
    "melancholic": "#8B5CF6",
    "high_energy": "#FBBF24",
    "ambient":     "#34D399",
    "aggressive":  "#F87171",
    "nostalgic":   "#F59E0B",
}

VIBE_EMOJIS = {
    "melancholic": "💜",
    "high_energy": "⚡",
    "ambient":     "🌿",
    "aggressive":  "🔥",
    "nostalgic":   "✨",
}

# ──────────────────────────────────────────────
# Synthetic data profiles
# ──────────────────────────────────────────────
# Each profile defines (mean, std) for all 28 features.
# Layout: [mfcc_1..13, spectral_centroid, chroma_1..12, tempo, rms_energy]
#
# Mapping rationale:
#   - MFCCs capture timbral texture; lower-order MFCCs relate to spectral
#     envelope shape. Melancholic tracks tend to have smoother envelopes
#     (higher MFCC-1), while aggressive tracks show sharper variation.
#   - Spectral Centroid = perceived "brightness". Higher = brighter.
#   - Chroma features encode pitch-class energy. Minor-key bias (chroma
#     bins 1,3,6,8,10 boosted) → melancholic; major-key bias → nostalgic.
#   - Tempo: slow ↔ calm/melancholic, fast ↔ energetic/aggressive.
#   - RMS Energy: quiet ↔ ambient/melancholic, loud ↔ aggressive/energetic.

def _build_profiles() -> dict:
    """
    Build Gaussian (mean, std) profiles for each vibe.

    Returns dict: {label: {"means": np.array(28,), "stds": np.array(28,)}}
    """
    profiles = {}

    # Helper: create chroma profile (12 bins)
    def _chroma(base, boosts=None):
        c = np.full(12, base)
        if boosts:
            for idx, val in boosts.items():
                c[idx] = val
        return c

    # ── Obsidian / Melancholic ──
    # Smooth timbre, low brightness, minor chroma, slow, quiet
    mfcc = np.array([-200, 80, -10, 20, -5, 10, -3, 5, -2, 3, -1, 2, -1], dtype=float)
    centroid = np.array([1500.0])
    chroma = _chroma(0.3, {1: 0.6, 3: 0.55, 6: 0.5, 8: 0.55, 10: 0.5})  # minor bias
    tempo = np.array([70.0])
    rms = np.array([0.02])
    profiles["melancholic"] = {
        "means": np.concatenate([mfcc, centroid, chroma, tempo, rms]),
        "stds":  np.concatenate([np.full(13, 15.0), [200], np.full(12, 0.08), [8], [0.005]])
    }

    # ── Solaris / High Energy ──
    # Bright timbre, high brightness, fast, loud
    mfcc = np.array([-150, 60, 5, 15, 0, 8, 2, 4, 1, 2, 0, 1, 0], dtype=float)
    centroid = np.array([3500.0])
    chroma = _chroma(0.45, {0: 0.6, 4: 0.6, 7: 0.55})  # major bias
    tempo = np.array([140.0])
    rms = np.array([0.12])
    profiles["high_energy"] = {
        "means": np.concatenate([mfcc, centroid, chroma, tempo, rms]),
        "stds":  np.concatenate([np.full(13, 15.0), [300], np.full(12, 0.08), [12], [0.02]])
    }

    # ── Emerald / Ambient ──
    # Textured timbre (high MFCC variance), mid brightness, slow, quiet
    mfcc = np.array([-180, 90, -15, 25, -8, 12, -5, 8, -3, 5, -2, 3, -1], dtype=float)
    centroid = np.array([2200.0])
    chroma = _chroma(0.40)  # spread, no strong bias
    tempo = np.array([60.0])
    rms = np.array([0.03])
    profiles["ambient"] = {
        "means": np.concatenate([mfcc, centroid, chroma, tempo, rms]),
        "stds":  np.concatenate([np.full(13, 20.0), [250], np.full(12, 0.06), [7], [0.008]])
    }

    # ── Crimson / Aggressive ──
    # Sharp timbre, very high brightness, fast, very loud
    mfcc = np.array([-120, 40, 15, 10, 5, 5, 3, 2, 2, 1, 1, 0, 0], dtype=float)
    centroid = np.array([4500.0])
    chroma = _chroma(0.35, {0: 0.5, 5: 0.5, 7: 0.45})  # dissonant intervals
    tempo = np.array([160.0])
    rms = np.array([0.18])
    profiles["aggressive"] = {
        "means": np.concatenate([mfcc, centroid, chroma, tempo, rms]),
        "stds":  np.concatenate([np.full(13, 12.0), [350], np.full(12, 0.07), [15], [0.025]])
    }

    # ── Amber / Nostalgic ──
    # Warm timbre, mid-warm brightness, mid tempo, moderate volume
    mfcc = np.array([-170, 70, -5, 18, -3, 9, -2, 6, -1, 4, -1, 2, 0], dtype=float)
    centroid = np.array([2500.0])
    chroma = _chroma(0.38, {0: 0.6, 4: 0.58, 7: 0.55, 9: 0.5})  # major / warm
    tempo = np.array([100.0])
    rms = np.array([0.07])
    profiles["nostalgic"] = {
        "means": np.concatenate([mfcc, centroid, chroma, tempo, rms]),
        "stds":  np.concatenate([np.full(13, 14.0), [220], np.full(12, 0.07), [10], [0.015]])
    }

    return profiles


# ──────────────────────────────────────────────
# Synthetic dataset generation
# ──────────────────────────────────────────────
def generate_synthetic_dataset(n_per_class: int = 20, seed: int = 42):
    """
    Generate a synthetic training dataset.

    Parameters
    ----------
    n_per_class : int
        Number of samples per vibe class.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_per_class * 5, 28)
    y : np.ndarray of shape (n_per_class * 5,) — string labels
    """
    rng = np.random.RandomState(seed)
    profiles = _build_profiles()
    X_parts, y_parts = [], []

    for label in VIBE_LABELS:
        p = profiles[label]
        samples = rng.normal(loc=p["means"], scale=p["stds"], size=(n_per_class, 28))
        X_parts.append(samples)
        y_parts.extend([label] * n_per_class)

    X = np.vstack(X_parts)
    y = np.array(y_parts)
    return X, y


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "vibe_model.pkl")


def train_model(n_per_class: int = 20, seed: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest on synthetic data and save the model.

    Returns the trained classifier.
    """
    print("=" * 46)
    print("   Vibe Classifier -- Training Pipeline")
    print("=" * 46)
    print()

    # Generate data
    X, y = generate_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    print(f"  Synthetic dataset : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Classes           : {VIBE_LABELS}")

    # Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Cross-validation score
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n  [OK] Model saved to: {MODEL_PATH}")

    return clf


def load_model() -> RandomForestClassifier:
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run `python train.py` or `python main.py --train` first."
        )
    return joblib.load(MODEL_PATH)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = train_model()
    print("\n-- Quick prediction test on first synthetic sample --")
    X, y = generate_synthetic_dataset()
    pred = model.predict(X[:1])
    print(f"  True label : {y[0]}")
    print(f"  Predicted  : {pred[0]}")
    print(f"  Display    : {VIBE_DISPLAY[pred[0]]}")
    print("\n[OK] train.py complete.")
