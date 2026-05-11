"""
Modules 3 & 4: SHAP Explainability + Agentic Publicist
=======================================================
Module 3 — Uses SHAP TreeExplainer to identify top contributing audio
           features and converts them into natural-language descriptions.

Module 4 — Connects to a local Ollama instance (Llama 3.2) to generate
           brand-aligned Instagram captions and YouTube descriptions.
"""

import numpy as np
import shap
import requests
import json

from extract import get_feature_names
from train import load_model, VIBE_DISPLAY


# ══════════════════════════════════════════════
# Module 3: SHAP Explainability
# ══════════════════════════════════════════════

# Human-readable descriptors for each feature group.
# Maps (feature_prefix) → (low_descriptor, high_descriptor)
FEATURE_DESCRIPTORS = {
    "mfcc_1":            ("smooth tonal warmth",        "sharp tonal edge"),
    "mfcc_2":            ("flat timbral texture",       "rich timbral texture"),
    "mfcc_3":            ("steady harmonic tone",       "dynamic harmonic shifts"),
    "mfcc_4":            ("simple overtone structure",  "complex overtone structure"),
    "mfcc_5":            ("muted spectral detail",      "vivid spectral detail"),
    "mfcc_6":            ("restrained articulation",    "expressive articulation"),
    "mfcc_7":            ("subdued resonance",          "pronounced resonance"),
    "mfcc_8":            ("minimal spectral motion",    "active spectral motion"),
    "mfcc_9":            ("calm sonic texture",         "animated sonic texture"),
    "mfcc_10":           ("uniform sound body",         "varied sound body"),
    "mfcc_11":           ("low spectral definition",    "high spectral definition"),
    "mfcc_12":           ("soft micro-texture",         "crisp micro-texture"),
    "mfcc_13":           ("flat upper harmonics",       "bright upper harmonics"),
    "spectral_centroid":  ("dark, warm tone",           "bright, brilliant tone"),
    "chroma_1":          ("minimal root presence",      "strong root presence"),
    "chroma_2":          ("low minor-second energy",    "high minor-second tension"),
    "chroma_3":          ("soft second harmony",        "bright second harmony"),
    "chroma_4":          ("sparse minor-third color",   "deep minor-third color"),
    "chroma_5":          ("thin major-third presence",  "warm major-third glow"),
    "chroma_6":          ("airy fourth resonance",      "grounded fourth resonance"),
    "chroma_7":          ("low tritone dissonance",     "high tritone dissonance"),
    "chroma_8":          ("quiet fifth harmony",        "powerful fifth harmony"),
    "chroma_9":          ("faint minor-sixth tone",     "lush minor-sixth tone"),
    "chroma_10":         ("sparse major-sixth color",   "rich major-sixth color"),
    "chroma_11":         ("low minor-seventh weight",   "heavy minor-seventh weight"),
    "chroma_12":         ("thin major-seventh edge",    "shimmering major-seventh edge"),
    "tempo":             ("low rhythmic complexity",    "high rhythmic drive"),
    "rms_energy":        ("quiet sonic intensity",      "high sonic intensity"),
}


def explain_prediction(
    model,
    feature_vector: np.ndarray,
    predicted_vibe: str,
    top_k: int = 3,
) -> str:
    """
    Use SHAP TreeExplainer to identify the top-K features driving the
    predicted vibe and return a natural-language summary.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained vibe classifier.
    feature_vector : np.ndarray
        Shape (28,) — the extracted feature vector.
    predicted_vibe : str
        The predicted vibe label (e.g., "melancholic").
    top_k : int
        Number of top features to include in the summary.

    Returns
    -------
    str
        Natural-language summary, e.g.:
        "bright, brilliant tone, high rhythmic drive, and high sonic intensity"
    """
    feature_names = get_feature_names()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vector.reshape(1, -1))

    # shap_values can be either:
    #   - 3D array: shape (n_samples, n_features, n_classes)  [SHAP >= 0.43]
    #   - list of arrays: one (n_samples, n_features) per class  [legacy]
    class_labels = list(model.classes_)
    class_idx = class_labels.index(predicted_vibe)

    # SHAP values for the predicted class, shape (28,)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[0, :, class_idx]
    else:
        sv = shap_values[class_idx][0]

    # Get top-K features by absolute SHAP value
    top_indices = np.argsort(np.abs(sv))[::-1][:top_k]

    # Build natural-language descriptors
    descriptors = []
    for idx in top_indices:
        fname = feature_names[idx]
        fvalue = feature_vector[idx]
        shap_val = sv[idx]

        low_desc, high_desc = FEATURE_DESCRIPTORS.get(
            fname, ("low " + fname, "high " + fname)
        )

        # Use the SHAP sign to decide direction:
        # Positive SHAP → this feature pushed *toward* this class
        # We use the actual feature value to decide "high" vs "low"
        # by comparing against a rough midpoint heuristic.
        if shap_val > 0:
            descriptors.append(high_desc)
        else:
            descriptors.append(low_desc)

    # Join with Oxford comma style
    if len(descriptors) == 1:
        return descriptors[0]
    elif len(descriptors) == 2:
        return f"{descriptors[0]} and {descriptors[1]}"
    else:
        return ", ".join(descriptors[:-1]) + f", and {descriptors[-1]}"


def get_shap_details(
    model,
    feature_vector: np.ndarray,
    predicted_vibe: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Return structured SHAP data for the top-K features.

    Returns a list of dicts with keys:
        feature, shap_value, feature_value, descriptor
    """
    feature_names = get_feature_names()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vector.reshape(1, -1))

    class_labels = list(model.classes_)
    class_idx = class_labels.index(predicted_vibe)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[0, :, class_idx]
    else:
        sv = shap_values[class_idx][0]

    top_indices = np.argsort(np.abs(sv))[::-1][:top_k]

    details = []
    for idx in top_indices:
        fname = feature_names[idx]
        low_desc, high_desc = FEATURE_DESCRIPTORS.get(
            fname, ("low " + fname, "high " + fname)
        )
        desc = high_desc if sv[idx] > 0 else low_desc
        details.append({
            "feature": fname,
            "shap_value": float(sv[idx]),
            "feature_value": float(feature_vector[idx]),
            "descriptor": desc,
        })

    return details


# ══════════════════════════════════════════════
# Module 4: The Agentic Song Namer (Ollama)
# ══════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

NAMER_PROMPT_TEMPLATE = """You are a professional Music Curator and Sonic Brand Architect. \
The AI analyzed a new track and classified it as {winner_vibe} (Runner-up: {runner_up_vibe}). \
The technical analysis shows the following core characteristics: {shap_data}.

Based on these specific technical traits, generate 3 unique and evocative song titles. \
For each title, provide a one-sentence technical "Why" that links the title's mood directly to the audio features mentioned.

Format your response exactly as:

### 1. [Title Name]
**Why:** [Technical link to audio features]

### 2. [Title Name]
**Why:** [Technical link to audio features]

### 3. [Title Name]
**Why:** [Technical link to audio features]"""


def contrast_analysis(
    model,
    feature_vector: np.ndarray,
    winner_vibe: str,
    runner_up_vibe: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Compare the SHAP contributions of the winner vs the runner-up vibe.
    Identifies features that pushed the model toward the winner while 
    simultaneously pulling it away from the runner-up.
    """
    feature_names = get_feature_names()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vector.reshape(1, -1))

    class_labels = list(model.classes_)
    win_idx = class_labels.index(winner_vibe)
    run_idx = class_labels.index(runner_up_vibe)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_win = shap_values[0, :, win_idx]
        sv_run = shap_values[0, :, run_idx]
    else:
        sv_win = shap_values[win_idx][0]
        sv_run = shap_values[run_idx][0]

    # Contrast score: how much MORE did this feature favor the winner over the runner-up?
    contrast_scores = sv_win - sv_run
    top_indices = np.argsort(np.abs(contrast_scores))[::-1][:top_k]

    contrast_details = []
    for idx in top_indices:
        fname = feature_names[idx]
        low_desc, high_desc = FEATURE_DESCRIPTORS.get(
            fname, ("low " + fname, "high " + fname)
        )
        
        # Determine the "winner" descriptor
        desc = high_desc if sv_win[idx] > 0 else low_desc
        
        contrast_details.append({
            "feature": fname,
            "win_shap": float(sv_win[idx]),
            "run_shap": float(sv_run[idx]),
            "diff": float(contrast_scores[idx]),
            "descriptor": desc
        })

    return contrast_details


def generate_song_names(
    winner_vibe: str, 
    runner_up_vibe: str, 
    shap_data: str, 
    model_name: str = OLLAMA_MODEL
) -> str:
    """
    Connect to a local Ollama instance and generate song titles based on vibe contrast.
    """
    prompt = NAMER_PROMPT_TEMPLATE.format(
        winner_vibe=winner_vibe,
        runner_up_vibe=runner_up_vibe,
        shap_data=shap_data
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.85,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[No response generated]")
    except Exception as e:
        return f"[WARNING] Song Namer Error: {e}"


def generate_copy(vibe: str, shap_data: str, model_name: str = OLLAMA_MODEL) -> str:
    """
    Connect to a local Ollama instance and generate social media copy.

    Parameters
    ----------
    vibe : str
        The display name of the predicted vibe (e.g., "Obsidian / Melancholic").
    shap_data : str
        Natural-language SHAP summary string.
    model_name : str
        Ollama model to use (default: llama3.2).

    Returns
    -------
    str
        Generated Instagram caption + YouTube description.
    """
    prompt = PROMPT_TEMPLATE.format(vibe=vibe, shap_data=shap_data)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        print("\n  [WAIT] Connecting to Ollama (Llama 3.2)...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[No response generated]")

    except requests.exceptions.ConnectionError:
        return (
            "[WARNING] Could not connect to Ollama. Make sure it is running:\n"
            "     1. Install Ollama: https://ollama.com/download\n"
            "     2. Pull the model: ollama pull llama3.2\n"
            "     3. Start the server: ollama serve\n\n"
            f"  [Fallback] Vibe: {vibe}\n"
            f"  [Fallback] SHAP: {shap_data}\n\n"
            "  Here is a template you can use manually:\n"
            f"  \"This track radiates a {vibe} energy, characterized by {shap_data}.\""
        )
    except requests.exceptions.Timeout:
        return (
            "[WARNING] Ollama request timed out (120s). The model may still be loading.\n"
            "     Try again in a moment, or use a smaller model: ollama pull llama3.2:1b"
        )
    except Exception as e:
        return f"[WARNING] Ollama error: {e}"


# ──────────────────────────────────────────────
# CLI self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from train import generate_synthetic_dataset, train_model
    import os

    # Ensure model exists
    from train import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("No model found -- training now...\n")
        model = train_model()
    else:
        model = load_model()

    # Generate a test sample
    X, y = generate_synthetic_dataset()
    test_sample = X[0]
    true_label = y[0]

    # Predict
    pred = model.predict(test_sample.reshape(1, -1))[0]
    vibe_display = VIBE_DISPLAY[pred]
    print(f"\n  True label : {true_label}")
    print(f"  Predicted  : {vibe_display}")

    # SHAP explanation
    shap_summary = explain_prediction(model, test_sample, pred)
    print(f"  SHAP summary: {shap_summary}")

    details = get_shap_details(model, test_sample, pred)
    print("\n  Top SHAP contributors:")
    for d in details:
        print(f"    {d['feature']:>20s}  SHAP={d['shap_value']:+.4f}  -> {d['descriptor']}")

    # Ollama test
    print("\n-- Ollama Copy Generation --")
    copy = generate_copy(vibe_display, shap_summary)
    print(copy)

    print("\n[OK] agent.py self-test complete.")
