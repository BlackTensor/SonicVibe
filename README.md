# 🎵 Music Emotional Aura Pipeline

A CPU-optimized, multimodal AI pipeline that analyzes audio clips to determine their **Emotional Aura**, explains the decision using **SHAP**, and generates brand-aligned social media content via a local **LLM (Ollama)**.

Built for **Questline Forge**.

---

## Architecture

```
Audio File (WAV/MP3)
        │
        ▼
┌──────────────────┐
│   extract.py     │  Librosa Feature Extraction
│   28-dim vector  │  MFCCs, Centroid, Chroma, Tempo, RMS
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    train.py      │  Random Forest Classifier (100 trees)
│    5 Aura Labels │  Trained on synthetic data (100 samples)
└──────┬───────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌──────────────┐     ┌────────────────┐
│  SHAP        │     │  Aura          │
│  Explainer   │     │  Prediction    │
│  (Top 3)     │     │                │
└──────┬───────┘     └───────┬────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
         ┌────────────────┐
         │   agent.py     │  Ollama / Llama 3.2
         │   LLM Copy     │  Instagram + YouTube
         └────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

### 2. Install Ollama (for LLM content generation)

```bash
# Download from https://ollama.com/download
# Then pull the model:
ollama pull llama3.2

# Start the server (if not auto-started):
ollama serve
```

### 3. Train the Model

```bash
python main.py --train
```

This generates 100 synthetic samples and trains a Random Forest classifier.

### 4. Analyze a Track

```bash
python main.py --audio path/to/your/track.wav
```

The pipeline will:
1. Extract 28 audio features
2. Predict the Emotional Aura
3. Explain the decision via SHAP
4. Generate Instagram + YouTube copy via Llama 3.2

---

## The 5 Emotional Auras

| Aura | Color | Signature | Audio Profile |
|------|-------|-----------|---------------|
| **Obsidian** | 🟣 | Melancholic | Low tempo (60-80 BPM), dark tone, minor key, quiet |
| **Solaris** | 🟡 | High Energy | Fast (130-160 BPM), bright tone, major key, loud |
| **Emerald** | 🟢 | Ambient | Slow (50-70 BPM), textured tone, spread harmony, quiet |
| **Crimson** | 🔴 | Aggressive | Very fast (140-180 BPM), harsh tone, dissonant, very loud |
| **Amber** | 🟠 | Nostalgic | Mid-tempo (90-110 BPM), warm tone, major key, moderate |

---

## Audio Features → Color Auras: Mapping Logic

The mapping operates at two levels:

### Level 1: Training Data (Domain Knowledge Encoding)

Each aura has a **hand-crafted Gaussian profile** defining expected mean and standard deviation for all 28 features. This embeds music psychology research into the model:

- **MFCCs (1-13)**: Capture timbral "texture." Melancholic music tends toward smoother spectral envelopes (higher MFCC-1 values), while aggressive music shows sharper, more jagged spectral variation.

- **Spectral Centroid**: The perceived "brightness" of sound. Ambient/melancholic tracks cluster around 1500-2200 Hz, while aggressive tracks push above 4000 Hz.

- **Chroma Features (12 bins)**: Encode pitch-class energy distribution. Minor-key bias (boosted bins at scale degrees ♭2, ♭3, ♭6, ♭7) correlates with melancholic auras. Major-key bias (boosted root, M3, P5) correlates with nostalgic/high-energy auras.

- **Tempo**: Direct mapping — slow (< 80 BPM) → calm/melancholic; mid (90-120) → nostalgic; fast (> 130) → energetic/aggressive.

- **RMS Energy**: Volume correlate — quiet (< 0.04) → ambient/melancholic; moderate (0.05-0.10) → nostalgic; loud (> 0.12) → energetic/aggressive.

### Level 2: SHAP Interpretation (Transparency)

At inference time, SHAP `TreeExplainer` decomposes the prediction into per-feature contributions. The top-3 features are translated into natural language using a descriptor lookup table, making the AI's reasoning transparent to both humans and the downstream LLM.

Example output:
> "This track was classified as **Obsidian / Melancholic** because of its  
> *dark, warm tone, low rhythmic complexity, and quiet sonic intensity.*"

---

## File Structure

```
├── extract.py          # Module 1: Audio feature extraction (Librosa)
├── train.py            # Module 2: Synthetic data generation + RF training
├── agent.py            # Module 3+4: SHAP explainability + Ollama LLM
├── main.py             # CLI orchestrator
├── requirements.txt    # Python dependencies
├── models/             # Saved model artifacts (auto-created)
│   └── aura_model.pkl
└── README.md           # This file
```

---

## Module Self-Tests

Each module can be run independently for testing:

```bash
python extract.py    # Tests feature extraction on a synthetic sine wave
python train.py      # Trains model + runs quick prediction test
python agent.py      # Tests SHAP + Ollama on synthetic data
```

---

## Requirements

- Python 3.10+
- Ollama with Llama 3.2 (for content generation)
- No GPU required — fully CPU-optimized

---

## License

Built for Questline Forge. All rights reserved.
