#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --job-name=mlm_bert_benchmark
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-4.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT MLM on Target Datasets"
source /ext3/env.sh
conda activate

# SST2 | Yelp (Every one of the script below takes roughly 18hrs - Just use one seed then)
# ---------------------------------------------------------------------------------------------------------------- #
# 17-x
python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/17-17 --cache_dir /scratch/yk2516/cache
# ---------------------------------------------------------------------------------------------------------------- #

'

# # SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# ***** Target ***** #

# IMDB | SST2 (3 minutes)
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/17-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/17-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/17-83 --cache_dir /scratch/yk2516/cache

# # 42-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/42-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/42-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/42-83 --cache_dir /scratch/yk2516/cache

# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/83-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/83-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name sst2 --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-sst2/83-83 --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #

# IMDB | Yelp (Tokenization: 20 minutes | Training: 9 hrs)
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/17-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/17-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/17-83 --cache_dir /scratch/yk2516/cache

# # 42-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/42-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/42-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/42-83 --cache_dir /scratch/yk2516/cache

# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/83-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/83-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83/bert_imdb_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/imdb-yelp/83-83 --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #


# SST2 | Yelp (Tokenization: 20 minutes | Training: 9 hrs)
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/17-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/17-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/17-83 --cache_dir /scratch/yk2516/cache

# # 42-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/42-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/42-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/42-83 --cache_dir /scratch/yk2516/cache

# # 17-x
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 17 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/83-17 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 42 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/83-42 --cache_dir /scratch/yk2516/cache
# python target_mlm.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --dataset_name yelp_polarity --do_train --do_eval --seed 83 --overwrite_output_dir --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_mlm_output/sst2-yelp/83-83 --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #
