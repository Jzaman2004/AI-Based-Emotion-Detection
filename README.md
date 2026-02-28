# 🧠 Emotions in the Eye of the Machine: AI-Based Facial Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Research](https://img.shields.io/badge/Research-AI%20Ethics-red.svg)]()

**Student Researchers:** [Jawad Zaman](https://github.com/Jzaman2004), [Gauri Shrestha](#)  
**Faculty Advisors:** Dr. Anna Egbert, Dr. Michael Burke  
**Repository:** [AI-Based Emotion Detection](https://github.com/Jzaman2004/AI-Based-Emotion-Detection)

---

## 📖 Table of Contents

- [📖 Abstract](#-abstract)
- [🎯 Research Questions & Hypotheses](#-research-questions--hypotheses)
- [🤖 Models Evaluated](#-models-evaluated)
- [📊 Dataset & Methodology](#-dataset--methodology)
- [🛠️ Installation & Requirements](#️-installation--requirements)
- [🚀 Usage Guide](#-usage-guide)
- [🎭 Demo Web App](#-demo-web-app)
- [📂 Repository Structure](#-repository-structure)
- [📈 Statistical Analysis & Outputs](#-statistical-analysis--outputs)
- [⚠️ Limitations & Ethical Considerations](#️-limitations--ethical-considerations)
- [📄 License & Citation](#-license--citation)

---

## 📖 Abstract

Artificial intelligence (AI) has rapidly shaped industries since the release of transformer models. AI-based facial emotion recognition is entering high-stakes domains like school surveillance systems. While designed for safety, such invasive solutions risk reinforcing bias and misinterpreting emotions, threatening psychological well-being. The accuracy of transformer-based AI (TB-AI) at interpreting emotions remains uncertain, especially due to biased training data.

This study systematically evaluates TB-AI's accuracy at classifying **seven emotions** (joy, sadness, anger, fear, disgust, surprise, hate) and assesses how **race/ethnicity**, **sex**, and **background context** affect this accuracy using a demographically balanced dataset of pictures presenting spontaneous facial expressions.

**Findings will inform fairness-aware benchmarks and ethical guidelines for emotion recognition systems that are more accurate, inclusive, and ethically aligned with human values.**

---

## 🎯 Research Questions & Hypotheses

### 1. Accuracy
*   **Q:** How accurately can TB-AI correctly identify emotions based on facial expressions?
*   **H₁:** TB-AI underperforms for positive and complex emotions (e.g., disgust) more than for basic emotions (e.g., anger).

### 2. Demographic Bias
*   **Q:** How does accuracy vary based on sex and race?
*   **H₂:** TB-AI is less accurate at recognizing emotions of minority groups. Higher accuracy is expected for White individuals and males.

### 3. Model Comparison
*   **Q:** Which model (Qwen, Idefics, Phi) demonstrates the highest accuracy?
*   **H₃:** Qwen2-VL-2B-Instruct is expected to achieve higher accuracy due to better attention to subtle facial details.

### 4. Context Sensitivity
*   **Q:** How does background manipulation affect accuracy?
*   **H₄:** Black background removal improves TB-AI emotion recognition accuracy compared to original backgrounds.

---

## 🤖 Models Evaluated

This project utilizes three open-weight multimodal vision-language models hosted on Hugging Face.

| Model | Parameters | License | Developer | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL-2B-Instruct** | 2B | Apache 2.0 | Alibaba Cloud | Specialized JSON prompt output |
| **Idefics2-8B** | 8B | Apache 2.0 | Hugging Face M4 | High computational demand |
| **Phi-3.5-vision-instruct** | 4.2B (Lang) | MIT | Microsoft Research | Optimized for efficiency |

---

## 📊 Dataset & Methodology

### Dataset Composition
*   **Total Images:** 70 open-source facial images.
*   **Balance:** Balanced by sex, race/ethnicity, and expression intensity.
*   **Demographics:**
    *   **Sex:** Male, Female (35 each)
    *   **Race:** Caucasian White, Black American, South Asian, East Asian, Latino (14 each)
    *   **Emotions:** Joy, Sadness, Anger, Fear, Disgust, Surprise, Hate (10 each)
*   **Naming Convention:** 4-letter Picture ID (e.g., `MCWJ` = **M**ale, **CW** (Caucasian White), **J**oy).

### Processing Pipeline
1.  **Pictureset Preparation:** Validation of 4-character ID formats.
2.  **Context Generation:** Each image is processed twice:
    *   **Original:** Unmodified image.
    *   **Black Background:** Background removed using `rembg` (U²-Net) and replaced with pure black.
3.  **Inference:** All 3 models process both contexts.
4.  **Normalization:** Raw outputs are parsed (JSON/Regex) and normalized to sum to 100%.
5.  **Statistical Analysis:** 8 distinct statistical blocks analyze bias, accuracy, and calibration.

---

## 🛠️ Installation & Requirements

This project is designed to run on **Google Colab Pro+** due to high GPU memory requirements.

### Hardware Requirements
*   **System RAM:** 32 GB minimum (64 GB suggested)
*   **GPU VRAM:** 80 GB minimum (141 GB suggested) *Required for batch processing 8B models*
*   **Storage:** 50 GB minimum

### Dependencies
Run the following in your Colab environment:

```bash
!pip install -q --upgrade-strategy only-if-needed \
"rembg==2.0.58" "pymatting==1.1.14" "onnxruntime-gpu==1.18.1" "onnx==1.16.1"

!pip install -q "transformers==4.45.2" "tokenizers==0.20.3" \
"accelerate==0.33.0" "safetensors==0.4.4" \
"torch==2.3.1" "torchvision==0.18.1" \
--extra-index-url https://download.pytorch.org/whl/cu121

!pip install -q "qwen-vl-utils==0.0.7" "timm>=0.9.0,<1.0.0" \
"bitsandbytes>=0.43.0,<0.45.0" "ipywidgets>=8.0.0" \
"diffusers>=0.27.0,<0.30.0"

# Statistical Analysis Libraries
!pip install -q scipy statsmodels scikit-learn seaborn matplotlib pandas
```

<details>
<summary><strong>📄 View Full Processing Code (Step 1-6)</strong></summary>

```python
# STEP 1: DEPENDENCY INSTALLATION
# (See Installation Section Above)

# STEP 2: DATASET PREPARATION & VALIDATION
import os, pandas as pd
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/MyDrive/Jawad Emotion AI"
# Discover valid images (4-character IDs)
valid_ids = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(DATASET_PATH)
    if f.lower().endswith(('.jpg', '.png')) and len(os.path.splitext(f)[0]) == 4
])

# STEP 3: HELPER FUNCTIONS & PROMPTS
# (Includes JSON parsing for Qwen, Regex for Phi/Idefics, Normalization logic)
# ... [Refer to code/Emotion_AI_Final_Code.ipynb for full functions] ...

# STEP 4: MODEL LOADING
# Loads Qwen, Idefics, and Phi from HuggingFace with float16 precision

# STEP 5: CONTEXT GENERATION
# Uses rembg to create Black Background variants

# STEP 6: AUTOMATED BATCH PROCESSING
# Loops through all 70 images, 2 contexts, 3 models
# Outputs: emotion_results.csv (68 columns)
```
</details>

---

## 🚀 Usage Guide

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Jzaman2004/AI-Based-Emotion-Detection.git
    cd AI-Based-Emotion-Detection
    ```
2.  **Prepare Data:**
    *   Upload the 70 facial images to your Google Drive folder `/MyDrive/Jawad Emotion AI`.
    *   Ensure images follow the 4-character ID naming convention (e.g., `MCWJ.png`).
3.  **Run the Notebook:**
    *   Open `code/Emotion_AI_Final_Code.ipynb` in Google Colab.
    *   Connect to a GPU runtime (T4/A100 recommended).
    *   Run cells sequentially from **Step 1** to **Step 6**.
4.  **Statistical Analysis:**
    *   Run the **Statistical Analysis** blocks (1-8) to generate CSV reports and PNG visualizations.
5.  **Output:**
    *   Results are saved to `emotion_results.csv` in the root directory.
    *   Statistical CSVs are saved to `results/csv/`.
    *   Visualizations are saved to `results/figures/`.

---

## 🎭 Demo Web App

This repository also includes a lightweight demo web app in the `demo/` folder for quick camera-based emotion analysis previews.

The demo results page includes an ethical feedback section where users can vote on perspective questions, with live per-option counters tracked for the current browser session.

For demo setup and usage details, see: `demo/demoREADME.md`.

---

## 📂 Repository Structure

```text
AI-Based-Emotion-Detection/
│
├── README.md                          # Project documentation
├── emotion_results.csv                # Main results file (68 columns)
│
├── code/                              # Source code
│   ├── Emotion_AI_Final_Code.ipynb   # Main Colab notebook
│   └── emotion_ai_final_code.py      # Python script version
│
├── data/                              # Dataset (70 facial expression images, 4-char IDs)
│   ├── MCWJ.jpg                      # Male Caucasian White Joy
│   ├── FBAA.jpg                      # Female Black American Anger
│   └── ... (68 more images)
│
└── results/                           # Generated analysis outputs
    ├── csv/                          # Statistical analysis results
    │   ├── per_emotion_accuracy.csv
    │   ├── demographic_bias_tests.csv
    │   ├── context_shift_analysis.csv
    │   ├── brier_scores.csv
    │   ├── fairness_metrics.csv
    │   └── mcnemar_tests.csv
    │
    └── figures/                      # Visualizations (PNG)
        ├── per_emotion_accuracy.png
        ├── mcnemar_comparison.png
        ├── demographic_bias_heatmap.png
        ├── context_shift_diverging.png
        ├── brier_calibration.png
        ├── confusion_matrix_qwen2.png
        ├── confusion_matrix_phi.png
        ├── confusion_matrix_idefics2.png
        └── fairness_ladder_sophisticated.png
```

---

## 📈 Statistical Analysis & Outputs

The repository includes generated CSV and PNG files detailing the following metrics. Each block corresponds to a specific research hypothesis.

<details>
<summary><strong>📊 Block 1: Per-Emotion Accuracy (Clopper-Pearson)</strong></summary>

*   **Purpose:** Addresses **Hypothesis 1**. Provides uncertainty bounds for accuracy estimates.
*   **Method:** Binomial proportion confidence interval (does not rely on normal approximation).
*   **Output:** `results/csv/per_emotion_accuracy.csv`, `results/figures/per_emotion_accuracy.png`
*   **Formula:** Lower bound: β(α/2; k, n−k+1), Upper bound: β(1−α/2; k+1, n−k)
</details>

<details>
<summary><strong>📊 Block 2: McNemar's Exact Test</strong></summary>

*   **Purpose:** Addresses **Hypothesis 3**. Compares paired model accuracy.
*   **Method:** Non-parametric test for paired nominal data (2×2 contingency table).
*   **Output:** `results/csv/mcnemar_tests.csv`, `results/figures/mcnemar_comparison.png`
*   **Formula:** p-value = 2 × min{ P(X ≤ min(b,c)), P(X ≥ max(b,c)) }
</details>

<details>
<summary><strong>📊 Block 3: Fisher's Exact Test (Demographic Bias)</strong></summary>

*   **Purpose:** Addresses **Hypothesis 2**. Quantifies accuracy differences between demographic subgroups.
*   **Method:** Statistical significance test for contingency tables (valid for small sample sizes).
*   **Output:** `results/csv/demographic_bias_tests.csv`, `results/figures/demographic_bias_heatmap.png`
</details>

<details>
<summary><strong>📊 Block 4: Context Shift Analysis</strong></summary>

*   **Purpose:** Addresses **Hypothesis 4**. Quantifies prediction instability when background is manipulated.
*   **Method:** Classifies shifts as **Harmful** (Correct → Wrong) or **Helpful** (Wrong → Correct).
*   **Output:** `results/csv/context_shift_analysis.csv`, `results/figures/context_shift_diverging.png`
</details>

<details>
<summary><strong>📊 Block 5: Brier Score (Calibration)</strong></summary>

*   **Purpose:** Measures probabilistic calibration (confidence vs. accuracy).
*   **Method:** Strictly proper scoring rule. Lower scores indicate better calibration.
*   **Output:** `results/csv/brier_scores.csv`, `results/figures/brier_calibration.png`
*   **Formula:** BS = (1/N) Σ (pᵢ − oᵢ)²
</details>

<details>
<summary><strong>📊 Block 6: Confusion Matrices</strong></summary>

*   **Purpose:** Visualizes systematic error patterns per model.
*   **Output:** `results/figures/confusion_matrix_qwen2.png`, `results/figures/confusion_matrix_phi.png`, `results/figures/confusion_matrix_idefics2.png`
</details>

<details>
<summary><strong>📊 Block 7: Equal Opportunity Difference</strong></summary>

*   **Purpose:** Fairness metric measuring True Positive Rate (TPR) disparities across races.
*   **Output:** `results/csv/fairness_metrics.csv`, `results/figures/fairness_ladder_sophisticated.png`
*   **Formula:** Max TPR Difference = max(TPRᵢ) − min(TPRᵢ)
</details>

<details>
<summary><strong>📊 Block 8: Model Majority Ensemble (MME)</strong></summary>

*   **Purpose:** Establishes an upper-bound performance benchmark.
*   **Method:** Emotion considered "correct" if ≥1 model identifies it accurately.
*   **Output:** Included in `emotion_results.csv`
</details>

---

## ⚠️ Limitations & Ethical Considerations

*   **Age:** Age information is not available for faces in the pictureset; results should not be generalized to age-related performance.
*   **Sample Size:** Only one image is used per (sex × race × emotion) combination due to resource constraints.
*   **Diversity:** The pictureset does not include multiracial individuals.
*   **Model Bias:** Each model was trained on different data sources, leading to uneven performance across races, genders, and emotional expressions.
*   **Ethical Warning:** This research highlights risks in deploying AI for emotion recognition. These models should **not** be used for high-stakes decision-making (hiring, policing, surveillance) without extensive fairness auditing.

---

## 📄 License & Citation

### License
*   **Research Code:** MIT License
*   **Models:** 
    *   Qwen2-VL-2B & Idefics2-8B: Apache 2.0
    *   Phi-3.5-vision-instruct: MIT License
*   **Dataset:** Open-source images with eligible licenses (Apache 2.0, Creative Commons, etc.).

### Citation
If you use this code or methodology in your research, please cite:

```bibtex
@misc{zaman2025emotions,
  title={Emotions in the Eye of the Machine: Accuracy, Bias, and Ethical Implications of AI-Based Facial Analysis},
  author={Zaman, Jawad and Shrestha, Gauri},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/Jzaman2004/AI-Based-Emotion-Detection}}
}
```

---

## 🤝 Contributing

This is an academic research project. For questions regarding the methodology, data, or statistical analysis, please contact the student researchers or faculty advisors via the repository Issues tab.

---

<div align="center">

**✨ Findings will inform fairness-aware benchmarks and ethical guidelines for emotion recognition systems that are more accurate, inclusive, and ethically aligned with human values. ✨**

</div>