# TOPSIS-YourName-1025

**Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)**

This Python package implements the TOPSIS method for Multi-Criteria Decision Making (MCDM). It takes a dataset of various options (models, products, etc.) with multiple criteria (features), accepts weights and impacts for each criterion, and ranks the options from best to worst.

This is particularly useful for ranking pre-trained models, selecting best-fit hardware, or any scenario where you need to choose the "best" option based on conflicting metrics (e.g., High Accuracy vs. Low Latency).

## Installation

You can install this package via pip:

## Installation
`pip install Pulkit_Model_Evaluation`

## Usage
`topsis <InputDataFile> <Weights> <Impacts> <ResultFileName>`

## Example
`topsis data.csv "0.25,0.25,0.25,0.25" "+,+,-,-" result.csv`