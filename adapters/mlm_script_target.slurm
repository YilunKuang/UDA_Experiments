#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --job-name=mlm_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - Run BERT MLM Training on Target datasets"
source /ext3/env.sh
conda activate

python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-42 --cache_dir /scratch/yk2516/cache

'

# ?
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --cache_dir /scratch/yk2516/cache

# 17-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --cache_dir /scratch/yk2516/cache


# 42-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --cache_dir /scratch/yk2516/cache


# 83-x 
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name sst2 --dataset_config_name default --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --cache_dir /scratch/yk2516/cache



