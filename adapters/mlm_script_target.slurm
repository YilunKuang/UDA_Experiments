#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --job-name=mlm_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-4.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT MLM Training on Target datasets"
source /ext3/env.sh
conda activate

# MNLI | SNLI
# ----------------------------------------------------------------------------------------------- #
# 17-17 (42 minutes)

python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17-earlystop \
                        --dataset_name snli \
                        --do_train \
                        --do_eval \
                        --seed 17 \
                        --overwrite_output_dir \
                        --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/mnli-snli/checkpoint-17-random-seed-17-earlystop \
                        --cache_dir /scratch/yk2516/cache \
                        --evaluation_strategy steps \
                        --eval_steps 500 \
                        --num_train_epochs 3 \
                        --greater_is_better False \
                        --metric_for_best_model eval_loss \
                        --load_best_model_at_end True


# ----------------------------------------------------------------------------------------------- #




'
# # MNLI | SNLI
# # ----------------------------------------------------------------------------------------------- #
# # 17-17 (42 minutes)

# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/mnli/checkpoint-random-seed-17-earlystop \
#                         --dataset_name snli \
#                         --do_train \
#                         --do_eval \
#                         --seed 17 \
#                         --overwrite_output_dir \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/mnli-snli/checkpoint-17-random-seed-17-earlystop \
#                         --cache_dir /scratch/yk2516/cache \
#                         --evaluation_strategy steps \
#                         --eval_steps 500 \
#                         --num_train_epochs 3 \
#                         --greater_is_better False \
#                         --metric_for_best_model eval_loss \
#                         --load_best_model_at_end True


# # ----------------------------------------------------------------------------------------------- #

# SNLI | MNLI
# ----------------------------------------------------------------------------------------------- #
# 17-17 (1 minute)
python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/snli/checkpoint-random-seed-17-earlystop \
                        --dataset_name mnli \
                        --do_train \
                        --do_eval \
                        --seed 17 \
                        --overwrite_output_dir \
                        --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/snli-mnli/checkpoint-17-random-seed-17-earlystop \
                        --cache_dir /scratch/yk2516/cache \
                        --evaluation_strategy steps \
                        --eval_steps 500 \
                        --num_train_epochs 3 \
                        --greater_is_better False \
                        --metric_for_best_model eval_loss \
                        --load_best_model_at_end True

# ----------------------------------------------------------------------------------------------- #

# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-2.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# mlm_model_path=/scratch/yk2516/UDA_Text_Generation/pretrain_output
# mlm_output_path=/scratch/yk2516/UDA_Text_Generation/target_pretrain_output

# # RTE | MNLI (4 hrs)
# # ----------------------------------------------------------------------------------------------- #
# # 17-17 (Completed in 5 mins)

# for VARIABLE_I in 17 42 83
# do
#     for VARIABLE_J in 17 42 83
#     do
#         python run_mlm_target.py --model_name_or_path $mlm_model_path/rte/checkpoint-random-seed-$VARIABLE_I \
#                             --dataset_name mnli \
#                             --do_train --do_eval --seed $VARIABLE_J \
#                             --overwrite_output_dir \
#                             --output_dir $mlm_output_path/rte-mnli/checkpoint-$VARIABLE_I-random-seed-$VARIABLE_J \
#                             --cache_dir /scratch/yk2516/cache
#     done
# done
# # ----------------------------------------------------------------------------------------------- #

# # MNLI | RTE
# # ----------------------------------------------------------------------------------------------- #
# # 17-17 (30 seconds)
# for VARIABLE_I in 17 42 83
# do
#     for VARIABLE_J in 17 42 83
#     do
#         python run_mlm_target.py --model_name_or_path $mlm_model_path/mnli/checkpoint-random-seed-$VARIABLE_I \
#                                 --dataset_name rte \
#                                 --do_train --do_eval --seed $VARIABLE_J \
#                                 --overwrite_output_dir \
#                                 --output_dir $mlm_output_path/mnli-rte/checkpoint-$VARIABLE_I-random-seed-$VARIABLE_J \
#                                 --cache_dir /scratch/yk2516/cache
#     done
# done
# # ----------------------------------------------------------------------------------------------- #

# # SNLI | MNLI
# # ----------------------------------------------------------------------------------------------- #
# # 17-17 (1 minute)
# for VARIABLE_I in 17 42 83
# do
#     for VARIABLE_J in 17 42 83
#     do
#         python run_mlm_target.py --model_name_or_path $mlm_model_path/snli/checkpoint-random-seed-$VARIABLE_I \
#                                 --dataset_name mnli \
#                                 --do_train --do_eval --seed $VARIABLE_J \
#                                 --overwrite_output_dir \
#                                 --output_dir $mlm_output_path/snli-mnli/checkpoint-$VARIABLE_I-random-seed-$VARIABLE_J \
#                                 --cache_dir /scratch/yk2516/cache
#     done
# done
# # ----------------------------------------------------------------------------------------------- #

# # MNLI | SNLI
# # ----------------------------------------------------------------------------------------------- #
# # 17-17 (42 minutes)
# for VARIABLE_I in 17 42 83
# do
#     for VARIABLE_J in 17 42 83
#     do
#         python run_mlm_target.py --model_name_or_path $mlm_model_path/mnli/checkpoint-random-seed-$VARIABLE_I \
#                                 --dataset_name snli \
#                                 --do_train --do_eval --seed $VARIABLE_J \
#                                 --overwrite_output_dir \
#                                 --output_dir $mlm_output_path/mnli-snli/checkpoint-$VARIABLE_I-random-seed-$VARIABLE_J \
#                                 --cache_dir /scratch/yk2516/cache
#     done
# done
# # ----------------------------------------------------------------------------------------------- #


# IMDB | Yelp
# ----------------------------------------------------------------------------------------------- #
# 42-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-42-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-42-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-42-random-seed-83 --cache_dir /scratch/yk2516/cache

# 17-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-17-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-17-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-17-random-seed-83 --cache_dir /scratch/yk2516/cache

# 83-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-83-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-83-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/imdb-yelp/checkpoint-83-random-seed-83 --cache_dir /scratch/yk2516/cache


# ----------------------------------------------------------------------------------------------- #


# SST-2 | Yelp
# ----------------------------------------------------------------------------------------------- #
# 17-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-17 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-83 --cache_dir /scratch/yk2516/cache

# 42-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-42 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-83 --cache_dir /scratch/yk2516/cache


# 83-x
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-17 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-42 --cache_dir /scratch/yk2516/cache
# python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/sst2/checkpoint-random-seed-83 --dataset_name yelp_polarity --do_train --do_eval --seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-83 --cache_dir /scratch/yk2516/cache

# ----------------------------------------------------------------------------------------------- #


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



