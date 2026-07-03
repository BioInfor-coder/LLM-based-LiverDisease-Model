# Liver Disease Data Processing & LLM Embedding Tools

Six standalone scripts for clinical data extraction, de-identification, data cleaning/imputation, and text embedding (Qwen3-based,Huatuo-o1-based,IIMedical-based, BGE-M3-based, and Doc2Vec-based).

## Repository Structure

```
├── DataEmbedding/
│   ├── LLM_embedding.py
│   ├── bge_m3_embedding.py
│   └── doc2vec_embedding.py
├── DataProcessing/
│   ├── data_preprocessing.py
│   ├── clean_drug_keywords.py
│   └── clean_lab_results.py
└── README.md
```

## 1. `dataprocessing/data_preprocessing.py`

Extracts patient records from a MySQL database, filters them through three stages (non-empty labs, remove all-placeholder results, missing-ratio/age threshold), and converts the results into natural-language text plus a wide-format lab results table.

```bash
python dataprocessing/data_preprocessing.py \
  --work-root ./data/YA_LLM_data_v2 \
  --lab-table-config ./config/lab_table_config.json \
  --class-labels AIH=0 PBC=1 DILI=2 CHB=3 \
  --disease-csv AIH=./csv/hit_AIH_earliest.csv PBC=./csv/hit_PBC_earliest.csv \
  --db-host <host> --db-user <user> --db-database <db_name> \
  --missing-ratio-threshold 0.6
```

(DB password via `PIPELINE_DB_PASSWORD` env var, or `--db-password`. Use `--skip-db-extraction` to skip stage 1.)

## 2. `dataprocessing/clean_drug_keywords.py`

Removes drug names and related clinical-description keywords from a text file, line by line, for de-identification purposes.

```bash
python dataprocessing/clean_drug_keywords.py -i raw.txt -o cleaned.txt
```

## 3. `dataprocessing/clean_lab_results.py`

Cleans a lab-results wide table (normalizes positive/negative text, scientific notation, comparison-symbol values), then KNN-imputes missing values (default flag `-1`) in continuous numeric columns only; categorical columns (binary 0/1, comparison-symbol strings, etc.) are left untouched with `-1` preserved.

```bash
python dataprocessing/clean_lab_results.py \
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
  --vector_size 4096 --epochs 40 --model_save_path ./doc2vec_model.bin
```

## Requirements

```bash
# dataprocessing/data_preprocessing.py
pip install mysql-connector-python pandas openpyxl

# dataprocessing/clean_drug_keywords.py
# stdlib only (os, re) — no extra install needed

# dataprocessing/clean_lab_results.py
pip install pandas scikit-learn openpyxl

# DataEmbedding/LLM_embedding.py, DataEmbedding/bge_m3_embedding.py
pip install torch transformers

# DataEmbedding/doc2vec_embedding.py
pip install gensim numpy
```

Python >= 3.9. The Transformer-based embedding scripts also require a GPU environment and locally downloaded model weights.
