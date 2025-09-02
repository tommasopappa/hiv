# Graph-PE-LLM-HIV: Structure-Aware Fine-Tuning for Molecular Property Prediction

Implementation of graph positional encoding techniques from [SAFT](https://arxiv.org/abs/2407.13381) adapted for molecular property prediction using the HIV dataset.

## Overview
This project reimplements the structure-aware fine-tuning approach from "SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation" for a different domain - predicting HIV activity from molecular structures represented as SMILES strings.

### Key Modifications:
- AMR graphs → Molecular graphs (SMILES)
- Text generation → Binary classification
- Magnetic Laplacian → Standard graph Laplacian
- LitGPT + LoRA on Pythia-160m

## Google Colab Notebook
The main implementation is in `HIV.ipynb` which should be run in Google Colab with GPU. It uses the following supporting files:
- `data_import.py` - Data preprocessing utilities
- `utils.py` - Helper functions

Git Clone this repo as Project_HIV within your main drive folder so that gdrive_path='/content/gdrive/MyDrive/Project_HIV'
