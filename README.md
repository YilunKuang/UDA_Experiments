# UDA_Text_Generation

## 1. Continued Pretraining of bert-base-uncased on the IMDB dataset

```bash
sbatch mlm_script.slurm
```

## 2. Adapter training for bert-base-uncased on the IMDB dataset

Use the uploaded version from [AdapterHub/bert-base-uncased-pf-imdb for bert-base-uncased](https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb)

## 3. Continued Pretraining of bert-base-uncased-with-imdb on the SST dataset

```bash
sbatch mlm_script_target.slurm
```

## 4. Adapter Evaluation for bert-base-uncased-with-imdb-and-sst and adapter on the SST dataset

```bash
sbatch adapter_script_target.slurm
```
## Additional: Fine-Tune bert-base-uncased(-with-imdb) on IMDB (Source Domain)
```bash
sbatch source_fine_tune.slurm
```
