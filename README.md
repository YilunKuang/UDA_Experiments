# UDA_Text_Generation

## Approach

### 1. Continued Pretraining of bert-base-uncased on the IMDB dataset

```bash
cd adapters
sbatch mlm_script_source.slurm
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

### 1. Fine-Tune the bert-base-uncased on IMDB (Source Domain) using the Classification Objective

```bash
cd ..
python3 benchmark/run_imdb_finetune.py
```

### 2. Zero-shot evaluate the finetuned BERT on the SST-2 (Target Domain)
```python
python3 benchmark/run_sst2_evaluate.py
``` 

## Result

### 1. Adapter Result (SST-2)

```shell
***** eval metrics *****
  eval_accuracy           =     0.5516
  eval_loss               =     0.6816
  eval_samples_per_second =    318.612
  eval_steps_per_second   =     39.826
```

### 2. Benchmark Result (SST-2)

```shell
***** eval metrics *****
  eval_accuracy           =     0.8784
  eval_loss               =     0.6598
  eval_runtime            = 0:00:11.51
  eval_samples_per_second =     75.722
  eval_steps_per_second   =      9.465
  perplexity              =     1.9345
```

<!-- ### 1. Fine-Tune the bert-base-uncased(-with-imdb) on IMDB (Source Domain) using the Classification Objective
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
```python
python run_sst_evaluate.py
``` -->


