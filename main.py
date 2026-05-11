"""
Main Pipeline Orchestrator
===========================
CLI entry point that chains all modules:

    python main.py --audio path/to/track.wav    # Full pipeline
    python main.py --train                      # Train on synthetic data
    python main.py --audio track.wav --train     # Train then analyze

Pipeline flow:
    Audio File -> Feature Extraction -> Vibe Prediction -> SHAP Explanation -> LLM Copy
"""

import argparse
import os
import sys
import numpy as np

from extract import extract_features, get_feature_names
from train import train_model, load_model, VIBE_DISPLAY, MODEL_PATH
from agent import explain_prediction, get_shap_details, generate_copy


def print_banner():
    """Print the pipeline banner."""
    print()
    print("  " + "=" * 55)
    print("  |                                                     |")
    print("  |   Music Sentiment Vibe Pipeline                     |")
    print("  |   by Questline Forge                                |")
    print("  |                                                     |")
    print("  " + "=" * 55)
    print()


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n  +--- {title} {'-' * (45 - len(title))}+")


def print_features(feature_vector: np.ndarray):
    """Print the extracted features in a formatted table."""
    names = get_feature_names()
    print(f"  | {'Feature':<22s} | {'Value':>12s} |")
    print(f"  |{'-' * 23}+{'-' * 14}|")
    for name, val in zip(names, feature_vector):
        print(f"  | {name:<22s} | {val:>12.4f} |")
    print(f"  +{'-' * 23}+{'-' * 14}+")


def run_pipeline(audio_path: str):
    """Execute the full analysis pipeline on an audio file."""

    # -- Step 1: Feature Extraction --
    print_section("Step 1: Feature Extraction")
    print(f"  | Source: {audio_path}")
    print(f"  | Extracting 28 audio features...")

    feature_vector = extract_features(audio_path)
    print(f"  | [OK] Feature vector: shape {feature_vector.shape}")
    print(f"  |")
    print_features(feature_vector)

    # -- Step 2: Vibe Classification --
    print_section("Step 2: Vibe Classification")

    if not os.path.exists(MODEL_PATH):
        print("  | No trained model found -- training now...")
        model = train_model()
    else:
        model = load_model()
        print("  | [OK] Model loaded from disk")

    predicted_vibe = model.predict(feature_vector.reshape(1, -1))[0]
    vibe_display = VIBE_DISPLAY[predicted_vibe]

    # Get prediction probabilities
    proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
    class_labels = list(model.classes_)

    print(f"  |")
    print(f"  | * Predicted Vibe: {vibe_display}")
    print(f"  |")
    print(f"  | Class Probabilities:")
    for label, prob in sorted(zip(class_labels, proba), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        display = VIBE_DISPLAY[label]
        marker = " <--" if label == predicted_vibe else ""
        print(f"  |   {display:<30s} {prob:6.1%} {bar}{marker}")
    print(f"  +{'-' * 50}+")

    # -- Step 3: SHAP Explainability --
    print_section("Step 3: SHAP Explainability")

    shap_summary = explain_prediction(model, feature_vector, predicted_vibe)
    shap_details = get_shap_details(model, feature_vector, predicted_vibe)

    print(f"  |")
    print(f"  | Top contributing features:")
    for i, d in enumerate(shap_details, 1):
        direction = "[+]" if d["shap_value"] > 0 else "[-]"
        print(f"  |   {i}. {d['feature']:<22s} SHAP={d['shap_value']:+.4f} {direction}")
        print(f"  |      -> {d['descriptor']}")
    print(f"  |")
    print(f"  | Natural language summary:")
    print(f"  |   \"{shap_summary}\"")
    print(f"  +{'-' * 50}+")

    # -- Step 4: LLM Copy Generation --
    print_section("Step 4: Agentic Publicist (Ollama)")
    print(f"  | Vibe  : {vibe_display}")
    print(f"  | SHAP  : {shap_summary}")
    print(f"  | Model : Llama 3.2 via Ollama")
    print(f"  +{'-' * 50}+")

    copy = generate_copy(vibe_display, shap_summary)

    print("\n  +--- Generated Content --------------------------------+")
    for line in copy.strip().split("\n"):
        print(f"  | {line}")
    print(f"  +{'-' * 50}+")

    # -- Summary --
    print("\n  " + "=" * 50)
    print(f"  [OK] Pipeline complete.")
    print(f"    File  : {os.path.basename(audio_path)}")
    print(f"    Vibe  : {vibe_display}")
    print(f"    Why   : {shap_summary}")
    print("  " + "=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Music Sentiment Vibe Pipeline -- Analyze audio, classify mood, generate content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                     Train the model on synthetic data
  python main.py --audio track.wav           Analyze a track
  python main.py --audio track.mp3 --train   Train first, then analyze
        """,
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to a WAV or MP3 audio file to analyze.",
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Train (or retrain) the classifier on synthetic data.",
    )

    args = parser.parse_args()

    if not args.audio and not args.train:
        parser.print_help()
        sys.exit(1)

    print_banner()

    # Training
    if args.train:
        train_model()

    # Analysis
    if args.audio:
        if not os.path.isfile(args.audio):
            print(f"\n  [ERROR] Audio file not found: {args.audio}")
            sys.exit(1)
        run_pipeline(args.audio)


if __name__ == "__main__":
    main()
