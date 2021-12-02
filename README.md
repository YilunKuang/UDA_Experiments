# UDA_Text_Generation

## Approach

### 1. Continued Pretraining of bert-base-uncased on the IMDB dataset

```bash
sbatch mlm_script.slurm
```

### 2. Adapter training for bert-base-uncased on the IMDB dataset

Use the uploaded version from [AdapterHub/bert-base-uncased-pf-imdb for bert-base-uncased](https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb)

### 3. Continued Pretraining of bert-base-uncased-with-imdb on the SST dataset

```bash
sbatch mlm_script_target.slurm
```

### 4. Adapter Evaluation for bert-base-uncased-with-imdb-and-sst and adapter on the SST dataset

```bash
sbatch adapter_script_target.slurm
```

## Benchmark

### 1. Fine-Tune the bert-base-uncased(-with-imdb) on IMDB (Source Domain) using the Classification Objective
```bash
sbatch source_fine_tune.slurm
```

### 2. Fine-Tune the bert-base-uncased(-with-imdb)-with-classification-on-imdb on SST (Target Domain) using the MLM Objective

```python
python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/source_finetune_vanilla --dataset_name sst --dataset_config_name default --do_train --do_eval --output_dir /scratch/yk2516/UDA_Text_Generation/source_finetune_vanilla_then_mlm_output --cache_dir /scratch/yk2516/cache
```


```python
python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/source_finetune_after_imdb_mlm --dataset_name sst --dataset_config_name default --do_train --do_eval --output_dir /scratch/yk2516/UDA_Text_Generation/source_finetune_after_imdb_mlm_then_mlm_output --cache_dir /scratch/yk2516/cache
```

### 3. Evaluate the final model on the SST (Target Domain) using the Classification Objective



