# Liver Disease Data Pipeline & LLM Embedding Tools

Standalone scripts for clinical data extraction, de-identification, data cleaning/imputation, text embedding (LLM-based, BGE-M3-based, and Doc2Vec-based), and multi-algorithm classification modeling (dimensionality reduction, hyperparameter selection, cross-validation, Internal Hold-out evaluation with Bootstrap 95% CI).

## Repository Structure

```
├── DataEmbedding/
│   ├── LLM_embedding.py
│   ├── bge-m3_embedding.py
│   └── doc2vec_embedding.py
├── DataProcessing/
│   ├── data_preprocessing.py
│   ├── clean_drug_keywords.py
│   └── clean_lab_results.py
├── Pipeline/
│   ├── config.py
│   ├── training.py
│   ├── metrics.py
│   └── pipeline.py
└── README.md
```

## 1. `DataProcessing/data_preprocessing.py`

Runs a four-stage local pipeline over already-extracted per-patient JSON files (one JSON file per patient, organized per disease): (1) drop samples with an empty lab-test list, (2) drop samples whose lab results are entirely placeholder values, (3) filter by missing-ratio and adult-age threshold while writing per-disease statistics CSVs, and (4) merge all surviving samples into a natural-language text corpus (`total.txt` + `labels.txt`) and a wide-format lab-results Excel table.

> Note: the previous MySQL-extraction stage has been removed from this script. Patient JSON files must already exist locally (e.g. produced by a separate extraction step); you now point the script at those directories directly via `--raw-dir`.

```bash
python DataProcessing/data_preprocessing.py \
  --work-root ./data/YA_LLM_data_v2 \
  --class-labels AIH=0 PBC=1 DILI=2 CHB=3 \
  --raw-dir AIH=./raw/AIH_raw PBC=./raw/PBC_raw DILI=./raw/DILI_raw CHB=./raw/CHB_raw \
  --missing-ratio-threshold 0.6
```

Arguments:

| Argument | Required | Description |
|---|---|---|
| `--work-root` | Yes | Root output directory; per-disease intermediate folders and stats CSVs are created underneath it |
| `--class-labels` | Yes | Space-separated `Disease=Label` pairs, e.g. `AIH=0 PBC=1 DILI=2 CHB=3` |
| `--raw-dir` | Yes | Space-separated `Disease=JSON目录路径` pairs pointing to each disease's local per-patient JSON directory |
| `--missing-ratio-threshold` | No (default `0.6`) | Minimum non-missing lab-result ratio required to keep a sample (stage 3) |

Outputs (under `--work-root`):
- `<disease>_filtered/`, `<disease>_cleaned/` — intermediate per-disease sample folders
- `<disease>_stats.csv` — per-sample missing-ratio statistics from stage 3
- `total_nl_labels_<threshold>/total.txt` + `labels.txt` — merged natural-language corpus and labels
- `lab_results_df_<threshold>.xlsx` — wide-format lab-results table (input to `clean_lab_results.py` and, later, the modeling pipeline's `--excel_path`)

## 2. `DataProcessing/clean_drug_keywords.py`

Removes drug names and related clinical-description keywords from a text file, line by line, for de-identification purposes.

```bash
python DataProcessing/clean_drug_keywords.py -i raw.txt -o cleaned.txt
```

## 3. `DataProcessing/clean_lab_results.py`

Cleans a lab-results wide table (normalizes positive/negative text, scientific notation, comparison-symbol values), then KNN-imputes missing values (default flag `-1`) in continuous numeric columns only; categorical columns (binary 0/1, comparison-symbol strings, etc.) are left untouched with `-1` preserved.

```bash
python DataProcessing/clean_lab_results.py \
  --input_path ./data/lab_results_df_0.2_0.6.xlsx \
  --output_path ./data/lab_results_df_0.2_0.6_cleaned.xlsx \
  --exclude_columns 样本来源 文件名 完整路径 疾病标签 \
  --n_neighbors 5
```

## 4. `DataEmbedding/LLM_embedding.py`

Batch-encodes text lines into normalized sentence embeddings using a local HuggingFace causal LM (tested with Qwen3-8B, HuatuoGPT-o1-8B, II-Medical-8B).

```bash
python DataEmbedding/LLM_embedding.py \
  --model_path ./Qwen3-8B \
  --input_path ./data/total.txt \
  --output_path ./data/total_afterembedding_Qwen3.txt \
  --cuda_visible_devices 0,1,2,3
```

## 5. `DataEmbedding/bge_m3_embedding.py`

Batch-encodes text lines into normalized sentence embeddings using a local BGE-M3 model, via mean pooling + L2 normalization.

```bash
python DataEmbedding/bge_m3_embedding.py \
  --model_path ./bge-m3 \
  --input_path ./data/total.txt \
  --output_path ./data/total_afterembedding_bge-m3.txt
```

## 6. `DataEmbedding/doc2vec_embedding.py`

Trains (or loads) a Doc2Vec model and infers a fixed-dimension vector per text line, matching the same input/output format as the Transformer embedding scripts above.

```bash
python DataEmbedding/doc2vec_embedding.py \
  --input_path ./data/total.txt \
  --output_path ./data/total_afterembedding_doc2vec.txt \
  --vector_size 384 --epochs 40 --model_save_path ./doc2vec_model.bin
```

## 7. Modeling Pipeline (`Modeling/`)

Four scripts that take the embeddings/labels/lab-results table produced above and run a full multi-algorithm classification workflow: per-fold dimensionality reduction (no data leakage), standard k-fold hyperparameter selection, 5-fold CV evaluation, full-training-set retraining, and Internal Hold-out evaluation with Bootstrap 95% confidence intervals.

| Script | Contents |
|---|---|
| `config.py` | Global constants (`N_SPLITS`, `RANDOM_SEED`, `AUTO_VARIANCE_THRESHOLD`, `HYPERPARAM_SELECTION_SCORING`, `DEFAULT_TEST_SIZE`, etc.) shared by the other three scripts |
| `training.py` | Data loading (raw vectors / labels / Excel features), train/Hold-out splitting (time-based or stratified random), dimensionality reduction (PCA/ScaledPCA/LDA/UMAP/Auto, with caching and diagnostic plots), classifier configs (RF/MLP/LR/XGBoost/SVM) with GridSearch grids, standard k-fold hyperparameter selection, and single-fold training/evaluation with a fixed hyperparameter set |
| `metrics.py` | Weighted/macro sensitivity, specificity, and AUC computation; Bootstrap 95% CI; all plotting (confusion matrices, ROC curves, CV/Hold-out comparison charts, dimensionality-reduction diagnostic plots); best-hyperparameter report generation (per-fold CSV, JSON, text summary) |
| `pipeline.py` | `run_pipeline(...)` — the main orchestration function — plus an `argparse` CLI entry point (`python pipeline.py --...`) |

`pipeline.py`, `training.py`, and `metrics.py` import from each other via plain module imports (`from config import ...`, `from training import ...`), so all four files must sit in the same directory.

### Command-line usage

```bash
python Pipeline/main.py \
  --labels_path ./data/total_nl_labels_0.6/labels.txt \
  --raw_vectors_path ./data/total_afterembedding_Qwen3.txt \
  --output_dir ./models/Qwen3_run \
  --excel_path ./data/lab_results_df_0.6_cleaned.xlsx \
  --excel_sheet Sheet1 \
  --concat_excel true \
  --reduction_method scaled_pca \
  --n_components 64 \
  --enable_grid_search true \
  --split_method random \
  --test_size 0.2
```

| Argument | Default | Description |
|---|---|---|
| `--labels_path` | required | Label file path (one integer class label per line) |
| `--raw_vectors_path` | required | Raw embedding vectors file (space/tab-separated) |
| `--output_dir` | required | Root output directory for all intermediate results and reports |
| `--excel_path` | `None` | Wide-format lab-results Excel path; required when `--split_method time` (reads the "检验日期" column), and used for feature concatenation when `--concat_excel true` |
| `--excel_sheet` | `Sheet2` | Excel sheet name |
| `--n_components` | `64` | Target dimensionality after reduction |
| `--concat_excel` | `true` | Whether to concatenate Excel lab features with the reduced embedding features |
| `--reduction_method` | `scaled_pca` | One of `pca`, `scaled_pca`, `auto`, `lda`, `umap`, `none` |
| `--enable_grid_search` | `true` | Whether to run GridSearch hyperparameter search per algorithm |
| `--variance_threshold` | `0.95` | Variance threshold used by the `auto` reduction method |
| `--split_method` | `time` | `time` (split by lab-test-date year, 2010–2019 vs. 2020–2025) or `random` (stratified random split) |
| `--test_size` | `0.2` | Hold-out fraction when `--split_method random` |

### Python usage

```python
from pipeline import run_pipeline

run_pipeline(
    labels_path="./data/total_nl_labels_0.6/labels.txt",
    raw_vectors_path="./data/total_afterembedding_Qwen3.txt",
    output_dir="./models/Qwen3_run",
    excel_path="./data/lab_results_df_0.6_cleaned.xlsx",
    excel_sheet="Sheet1",
    concat_excel=True,
    reduction_method="scaled_pca",
    n_components=64,
    enable_grid_search=True,
    split_method="random",
    test_size=0.2,
)
```

### Outputs (under `--output_dir`)

- `modeling_data/`, `fold_splits/`, `reduction_diagnostics/` — per-fold modeling data, split indices, and dimensionality-reduction diagnostics (component/variance CSVs, scatter plots)
- `best_params/` — per-algorithm hyperparameter ranking CSVs, selected-config JSON, and a text summary report
- `confusion_matrices/`, `roc_curves/` — per-algorithm, per-fold and Hold-out plots
- `algorithm_comparison_summary_{weighted,macro}.csv`, `cv_detailed_reports_{weighted,macro}.csv`, `cv_performance_comparison.png` — 5-fold CV summaries
- `train_reports/`, `holdout_val_reports/`, `holdout_val_results_comparison.csv`, `holdout_val_performance_comparison.png`, `auc_comparison_detail.png` — Internal Hold-out Validation results
- `confidence_intervals/best_models_95CI_{weighted,macro}.csv` — Bootstrap 95% CI on the Hold-out set
- `best_models/`, `final_retrained_models/` — pickled model bundles (classifier + reducer + scaler + selected hyperparameters) retrained on the full training set
- `config.json` — full run configuration for reproducibility

## Requirements

```bash
# DataProcessing/data_preprocessing.py
pip install pandas openpyxl

# DataProcessing/clean_drug_keywords.py
# stdlib only (os, re) — no extra install needed

# DataProcessing/clean_lab_results.py
pip install pandas scikit-learn openpyxl

# DataEmbedding/LLM_embedding.py, DataEmbedding/bge_m3_embedding.py
pip install torch transformers

# DataEmbedding/doc2vec_embedding.py
pip install gensim numpy

# Modeling/config.py, training.py, metrics.py, pipeline.py
pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib xgboost umap-learn
```

Python >= 3.9. The Transformer-based embedding scripts also require a GPU environment and locally downloaded model weights. For the modeling pipeline, `scikit-learn >= 1.7` is recommended so that `MLPClassifier` fully supports `sample_weight`; on older versions the pipeline automatically falls back to training without sample weights and prints a warning.
