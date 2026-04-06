# ReqVS
Source codes for paper "From Sequence to Hit: Reliable Virtual Screening via Interaction Entropy Enables HCAR1 Antagonist Discovery"

## Framework Overview

![ReqVS Framework](framework0401.odf)

## Environment Setup and Dependencies

The provided environments have been tested and can run on Linux systems with CUDA 11.8. You can either download the model environments directly via the provided download links, or visit their project pages to set up environments yourself.

### DTI Models
- **Drugban**: [Repository Link](https://github.com/peizhenbai/DrugBAN)
- **MCANet**: [Repository Link](https://github.com/MrZQAQ/MCANet)
- **HitScreen**: [Repository Link](https://github.com/chengeng17/HitScreen)
- **ColdstartCPI**: [Repository Link](https://github.com/zhaoqichang/ColdstartCPI)

### DTA Models
- **GraphDTA**: [Repository Link](https://github.com/thinng/GraphDTA)
- **MGraphDTA**: [Repository Link](https://github.com/guaguabujianle/MGraphDTA)
- **PSICHIC**: [Repository Link](https://github.com/huankoh/PSICHIC)
- **DeepDTAGen**: [Repository Link](https://github.com/CSUBioGroup/DeepDTAGen)

We sincerely acknowledge and thank the authors of these models for their valuable contributions to the community, which have made this work possible.
## Quick Reproduction

You can directly download the **[pre-computed results](https://zenodo.org/records/18464350)** for DUD-E and DEKOIS2.0 of all 8 models, extract them to the `test_results` folder, and run `analysis_pipeline_DEKOIS2.ipynb` and `analysis_pipeline_DUD_E.ipynb` to test the results directly.

## Quick Start

In the quick start guide, you can test using pre-built environments by downloading them, test your VS tasks by simply modifying the test folder path, or test on benchmarks such as DUD-E.csv or DEKOIS2.0.csv **[download](https://zenodo.org/records/18464350)**.

### 1. Download and Extract Environments, Model Weights, and Pre-trained Weights

Download pre-packaged conda environments, model weights, and pre-trained weights from **[link](https://zenodo.org/records/18476309)**:

After downloading, extract all environment archives to the `envs/` directory. Each environment should be extracted to its corresponding folder (e.g., `envs/drugban/`, `envs/mcanet/`, etc.).

Place the corresponding model weights in each model's `weights` folder.

There is one pre-trained model weight `pytorch_model.bin` that should be placed in `ReqVS/models/coldstartcpi/Feature_generation/ProtTrans/prot_bert_bfd`.

### 2. Setup Environments

Run the environment setup script to unpack all conda environments:

```bash
bash unpack_all_envs.sh
```

This script will:
- Automatically detect all packed environments in the `envs/` directory
- Unpack each environment using `conda-unpack`
- List all available environments

### 3. Configure Inference

Edit `inference_config.json` to configure your inference task:

```json
{
  "dataset": "/path/to/your/screening/dataset.csv",
  "execution_mode": "parallel",
  "log_dir": "./logs",
  "models": [
    {
      "name": "drugban",
      "enabled": true,
      "script_path": "benchmark_test.py",
      "env_name": "drugban",
      "device": "cuda:0",
      "extra_args": {}
    }
    // ... other models
  ]
}
```

### 4. Run Model Inference

Execute batch inference to run all configured models:

```bash
python run_all_models_inference.py --config inference_config.json
```

**Note on HitScreen**: Due to environment conflicts, we are unable to provide a direct inference pipeline for HitScreen. For inference on any chemical library, you need to navigate to `hitscreen/data` and manually build the feature database files for the VS library. For DUD-E, this may result in files exceeding 40GB. We recommend removing the HitScreen configuration from the inference pipeline initially. The one-click inference functionality for the HitScreen model will be available in future versions.

### 5. Configure Pipeline

Edit `pipeline_config.yaml` to configure the analysis pipeline:

```yaml
input_folder: "test_results/datasetname"        # Folder containing model prediction results
input_file_prefix: "datasetname"                # Prefix of prediction files
output:
  folder_name: "results"
  suffix: "_DUD-E_analysis"
models:
  classifier_models:
    - name: drugban
      display_name: Drugban
    # ... other classifier models
  regressor_models:
    - name: graphdta
      display_name: GraphDTA
    # ... other regressor models

```

### 6. Run Complete Pipeline

Execute the complete analysis pipeline:

```bash
python complete_pipeline.py
```

This pipeline executes four steps:
1. **Aggregate Model Results**: Merge prediction results from all models
2. **Calculate Consistency and Entropy**: Compute consensus scores and entropy metrics
3. **Generate EF Files**: Calculate enrichment factors for each model and consensus methods
4. **Performance Analysis**: Generate performance plots (run separately)

### 7. Generate Performance Plots

Run performance analysis to generate visualization plots:

```bash
python analyze_performance.py
```

This will generate:
- `performance_summary_{suffix}.csv`: Summary statistics for all methods
- `performance_boxplot_{suffix}.png`: Box plots comparing different methods

## Demo Notebooks

The `analysis_pipeline.ipynb` notebook provides an interactive demonstration of the complete workflow. You can:

1. **Download Pre-computed Results**: Use the provided download links to obtain pre-computed prediction files for testing
2. **Run Step-by-Step**: Execute each cell to view intermediate results
3. **Customize Analysis**: Interactively modify parameters and visualize results

## Adding Your Own Model

To easily integrate your own model into the unified multi-model prediction framework, follow these steps:

### Prerequisites

Install conda-pack:
```bash
conda install conda-pack
```

### Steps

1. **Package Your Environment**: Create a conda-pack archive of your model's conda environment:
   ```bash
   conda pack -n your_env_name -o envs/your_model_name.tar.gz
   ```

2. **Extract Environment**: Extract the archive to `envs/your_model_name/`:
   ```bash
   mkdir -p envs/your_model_name
   tar -xzf envs/your_model_name.tar.gz -C envs/your_model_name
   ```

3. **Add Model Code**: Place your model code in `models/your_model_name/` directory.

4. **Create Inference Script**: Provide an inference script (e.g., `benchmark_test.py`) with the following input/output requirements:
   - **Input**: CSV file with columns: `ID`, `Protein`, `SMILES`, `Y`
   - **Output**: CSV file with columns: `ID`, `Protein`, `SMILES`, `Y`, `prediction_{model_name}`
   - The script should accept command-line arguments:
     - `--input`: Path to input CSV file
     - `--output`: Path to output CSV file
     - `--device`: Device to use (e.g., `cuda:0`)

5. **Update Configuration**: Add your model information to `inference_config.json`:
   ```json
   {
     "name": "your_model_name",
     "enabled": true,
     "script_path": "benchmark_test.py",
     "env_name": "your_model_name",
     "device": "cuda:0",
     "extra_args": {}
   }
   ```

6. **Update Pipeline Config**: Add your model to `pipeline_config.yaml`:
   - If it's a classifier model (outputs probabilities 0-1), add it to `classifier_models`
   - If it's a regressor model (outputs continuous values), add it to `regressor_models`

## Citation

If you use ReqVS in your research, please cite:

```bibtex

```

## Retraining and Training Data Preparation

If you want to build training datasets from scratch and train these models, each model provides a corresponding `main_chembl36.py` training script. For all data processing procedures, please refer to the `data_prepration` folder.

## Acknowledgments

This project integrates the following models:
- Drugban
- MCANet
- HitScreen
- ColdstartCPI
- GraphDTA
- MGraphDTA
- PSICHIC
- DeepDTAGen

We thank the authors of these models for their contributions to the field.
