#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --job-name=adapter_bert_source
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-2.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Adapter Training on Source datasets"
source /ext3/env.sh
conda activate

# SNLI Training (Really long! -> 14:48:10)
# ------------------------------------------------------------------------------ #
# 17-x
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-17-earlystop \
                               --random_seed 17 \
                               --dataset_name snli \
                               --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/snli/17-17-earlystop/
# ------------------------------------------------------------------------------ #




'

# # SNLI Training (Really long! -> 14:48:10)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-17-earlystop \
#                                --random_seed 17 \
#                                --dataset_name snli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/snli/17-17-earlystop/
# # ------------------------------------------------------------------------------ #

# # MNLI Training (10 hrs each one)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17-earlystop \
#                                --random_seed 17 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/17-17-earlystop/
# # ------------------------------------------------------------------------------ #

# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# # QQP Training (8:59:28)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-17 \
#                                --random_seed 17 \
#                                --dataset_name qqp \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/qqp/17-17/

# # ------------------------------------------------------------------------------ #

# # MNLI Training (10 hrs each one)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17/checkpoint-7000 \
#                                --random_seed 17 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/17-17/
# # ------------------------------------------------------------------------------ #

# # SNLI Training (Really long! -> 14:48:10)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-17/checkpoint-6000 \
#                                --random_seed 17 \
#                                --dataset_name snli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/snli/17-17/

# # ------------------------------------------------------------------------------ #

# # MNLI Training (10 hrs each one)
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17 \
#                                --random_seed 17 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/17-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17 \
#                                --random_seed 42 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/17-42/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17 \
#                                --random_seed 83 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/17-83/

# # 42-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-42 \
#                                --random_seed 17 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/42-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-42 \
#                                --random_seed 42 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/42-42/
                               
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-42 \
#                                --random_seed 83 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/42-83/

# # 83-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-83 \
#                                --random_seed 17 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/83-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-83 \
#                                --random_seed 42 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/83-42/
                               
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-83 \
#                                --random_seed 83 \
#                                --dataset_name mnli \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/mnli/83-83/

# # ------------------------------------------------------------------------------ #

# # RTE Training
# # ------------------------------------------------------------------------------ #
# # 17-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-17 \
#                                --random_seed 17 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/17-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-17 \
#                                --random_seed 42 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/17-42/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-17 \
#                                --random_seed 83 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/17-83/

# # 42-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-42 \
#                                --random_seed 17 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/42-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-42 \
#                                --random_seed 42 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/42-42/
                               
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-42 \
#                                --random_seed 83 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/42-83/

# # 83-x
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-83 \
#                                --random_seed 17 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/83-17/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-83 \
#                                --random_seed 42 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/83-42/
                               
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-83 \
#                                --random_seed 83 \
#                                --dataset_name rte \
#                                --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/rte/83-83/

# # ------------------------------------------------------------------------------ #

# SST-2 Training
# ------------------------------------------------------------------------------ #
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --random_seed 17 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-17/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --random_seed 42 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-42/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --random_seed 83 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-83/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --random_seed 17 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-17/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --random_seed 42 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-42/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --random_seed 83 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-83/

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --random_seed 17 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-17/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --random_seed 42 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-42/
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --random_seed 83 --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-83/
# ------------------------------------------------------------------------------ #

# IMDB Training
# ------------------------------------------------------------------------------ #
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 17 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/ --cache_dir /scratch/yk2516/cache
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 42 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/ --cache_dir /scratch/yk2516/cache
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 83 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/ --cache_dir /scratch/yk2516/cache

# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 17 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/ --cache_dir /scratch/yk2516/cache
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 42 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/ --cache_dir /scratch/yk2516/cache
# python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 83 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/ --cache_dir /scratch/yk2516/cache
# ------------------------------------------------------------------------------ #
