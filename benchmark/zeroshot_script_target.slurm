#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --job-name=zeroshot_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-8.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Zeroshot on Target datasets"
source /ext3/env.sh
conda activate

# IMDB | Yelp

python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-17
python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-42
python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-83

python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-17
python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-42
python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-83

python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-17
python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-42
python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-83
'

# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# IMDB | Yelp
# ---------------------------------------------------------------------------------------------------------------- #
# 17-x
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-17
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-42
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-83

# 42-x
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-17
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-42
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-83

# 83-x
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-17
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-42
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-83
# ---------------------------------------------------------------------------------------------------------------- #



# IMDB | SST-2
# ---------------------------------------------------------------------------------------------------------------- #
# 17-x
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-17
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-42
# python run_target_evaluate.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-83

# 42-x
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-17
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-42
# python run_target_evaluate.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-83

# 83-x
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-17
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-42
# python run_target_evaluate.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-83
# ---------------------------------------------------------------------------------------------------------------- #

