#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --job-name=mlm_bert
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-2.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT MLM Training on Source datasets"
source /ext3/env.sh
conda activate

# *** MLM on QQP *** # 
# -------------------------------------------------------------------------------------------- #
# ** seed 17 **
python run_mlm_source.py --model_name_or_path bert-base-uncased \
                         --dataset_name qqp \
                         --do_train \
                         --do_eval \
                         --seed 17 \
                         --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-17 \
                         --cache_dir /scratch/yk2516/cache

# ** seed 42 **
python run_mlm_source.py --model_name_or_path bert-base-uncased \
                         --dataset_name qqp \
                         --do_train \
                         --do_eval \
                         --seed 42 \
                         --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-42 \
                         --cache_dir /scratch/yk2516/cache

# ** seed 83 **
python run_mlm_source.py --model_name_or_path bert-base-uncased \
                         --dataset_name qqp \
                         --do_train \
                         --do_eval \
                         --seed 83 \
                         --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-83 \
                         --cache_dir /scratch/yk2516/cache
# -------------------------------------------------------------------------------------------- #
'

# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# # *** MLM on QQP *** # 
# # -------------------------------------------------------------------------------------------- #
# # ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name qqp \
#                          --do_train \
#                          --do_eval \
#                          --seed 17 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-17 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name qqp \
#                          --do_train \
#                          --do_eval \
#                          --seed 42 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-42 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name qqp \
#                          --do_train \
#                          --do_eval \
#                          --seed 83 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/qqp/checkpoint-random-seed-83 \
#                          --cache_dir /scratch/yk2516/cache
# # -------------------------------------------------------------------------------------------- #



# # *** MLM on RTE *** # 
# # -------------------------------------------------------------------------------------------- #
# # ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name rte \
#                          --do_train \
#                          --do_eval \
#                          --seed 17 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-17 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name rte \
#                          --do_train \
#                          --do_eval \
#                          --seed 42 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-42 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name rte \
#                          --do_train \
#                          --do_eval \
#                          --seed 83 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/rte/checkpoint-random-seed-83 \
#                          --cache_dir /scratch/yk2516/cache
# # -------------------------------------------------------------------------------------------- #



# # *** MLM on SNLI *** # 
# # -------------------------------------------------------------------------------------------- #
# # ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name snli \
#                          --do_train \
#                          --do_eval \
#                          --seed 17 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-17 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name snli \
#                          --do_train \
#                          --do_eval \
#                          --seed 42 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-42 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name snli \
#                          --do_train \
#                          --do_eval \
#                          --seed 83 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-83 \
#                          --cache_dir /scratch/yk2516/cache
# # -------------------------------------------------------------------------------------------- #



# # *** MLM on MNLI *** # 
# # -------------------------------------------------------------------------------------------- #
# # ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name mnli \
#                          --do_train \
#                          --do_eval \
#                          --seed 17 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name mnli \
#                          --do_train \
#                          --do_eval \
#                          --seed 42 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-42 \
#                          --cache_dir /scratch/yk2516/cache

# # ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased \
#                          --dataset_name mnli \
#                          --do_train \
#                          --do_eval \
#                          --seed 83 \
#                          --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-83 \
#                          --cache_dir /scratch/yk2516/cache
# # -------------------------------------------------------------------------------------------- #



# *** MLM on IMDB ***
# -------------------------------------------------------------------------------------------- #
# ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name imdb --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --cache_dir /scratch/yk2516/cache

# ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name imdb --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --cache_dir /scratch/yk2516/cache

# ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name imdb --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --cache_dir /scratch/yk2516/cache
# -------------------------------------------------------------------------------------------- #



# *** MLM on SST-2 ***
# -------------------------------------------------------------------------------------------- #
# ** seed 42 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name sst2 --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --cache_dir /scratch/yk2516/cache

# ** seed 17 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name sst2 --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --cache_dir /scratch/yk2516/cache

# ** seed 83 **
# python run_mlm_source.py --model_name_or_path bert-base-uncased --dataset_name sst2 --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --cache_dir /scratch/yk2516/cache
# -------------------------------------------------------------------------------------------- #



# *** MLM on Yelp ***
# -------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------- #
