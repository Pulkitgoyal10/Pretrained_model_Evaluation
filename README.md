# TOPSIS for Text Summarization Model Selection

[![PyPI version](https://img.shields.io/pypi/v/Pulkit-Model-Evaluation.svg)](https://pypi.org/project/Pulkit-Model-Evaluation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## 📋 Project Overview

This project implements a structured, mathematical approach to selecting the best pre-trained **Text Summarization Model** from Hugging Face. It utilizes the **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) method to rank models based on multiple conflicting criteria such as ROUGE scores, Inference Time, and Model Size.

The project is divided into two main components:
1.  **Data Generation:** A Python script to evaluate Hugging Face models and generate performance metrics.
2.  **TOPSIS Analysis:** A custom PyPI package (`Pulkit-Model-Evaluation`) that applies the TOPSIS algorithm to rank the models.

---

## 🚀 Part 1: Generating the Dataset

The first step involves running the evaluation script to test models like BART, T5, and Pegasus.

### 1. Prerequisites
You need to install the following libraries to run the model evaluation script:

---

## Sample Data Generated from pretrained models

<img width="491" height="164" alt="image" src="https://github.com/user-attachments/assets/610afe31-6763-4df3-b9c4-effc9e0d1d97" />

---

# Sample Result Generated

<img width="768" height="194" alt="image" src="https://github.com/user-attachments/assets/f40bbe0c-1eae-47ec-a1dd-7fec3b917b8b" />


---
```bash
pip install torch transformers evaluate rouge_score pandas sentencepiece
