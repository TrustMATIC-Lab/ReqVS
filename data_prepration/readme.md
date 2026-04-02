# ChEMBL36 Data Processing Pipeline

This document describes the complete data processing pipeline for building training datasets from the ChEMBL36 database for DrugCS2 model training.

## Overview

This pipeline processes the ChEMBL36 database to create two types of training datasets:
1. **Classification Dataset**: Binary classification (active/inactive) based on binding affinity thresholds
2. **Regression Dataset**: Continuous affinity prediction using Ki/Kd values

The pipeline consists of 13 sequential steps, all integrated into a single Jupyter Notebook for data cleaning, filtering, clustering, and augmentation.

## Required Files

### ChEMBL36 Database Files

You need to download the following files from ChEMBL:
- File: `chembl_36.db`
   [Download link](https://chembl.gitbook.io/chembl-interface-documentation/downloads)

### Test Set Files

DUD-E and DEKOIS2.0 can be downloaded from [here]

### Directory Structure

```
data_prepration/
├── readme.md
├── data_preparation.ipynb          # Complete data processing pipeline Notebook
├── chembl_36.db                    # ChEMBL36 SQLite database
├── DUD-E.csv                       # DUD-E test set
└── DEKOIS2.0.csv                   # DEKOIS2.0 test set
```

## Dependencies

### Python Packages
- pandas
- numpy
- rdkit
- matplotlib
- tqdm
- jupyter

### External Tools
- **cd-hit**: Required for protein sequence clustering
  - Installation: `conda install -c bioconda cd-hit`

## Quick Start

1. **Install dependencies**:
   ```bash
   conda install -c conda-forge pandas numpy matplotlib tqdm rdkit jupyter
   conda install -c bioconda cd-hit
   ```

2. **Prepare data files**:
   - Place `chembl_36.db` in the `data_prepration/` directory
   - Place `DUD-E.csv` and `DEKOIS2.0.csv` in the `data_prepration/` directory

3. **Run Notebook**:
   ```bash
   jupyter notebook data_preparation.ipynb
   ```
   Then execute all cells in order.

## Pipeline Description

All steps are integrated in the `data_preparation.ipynb` Notebook, organized as different sections in sequence.

Classification Dataset:
Final output: `step9_final_dataset.csv`

Regression Dataset:
Final output: `step13_regression_dataset_with_negatives.csv`
