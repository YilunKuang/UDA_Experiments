#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --job-name=finetune_bert
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-4.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Finetuning on Source / Target Datasets"
source /ext3/env.sh
conda activate

# Finetune on MNLI (Works!) - 10 Hrs per scripts
# ---------------------------------------------------------------------------------------------------------------- #
python run_benchmark.py --do_train \
                        --do_eval \
                        --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/mnli/17/bert_mnli_finetune \
                        --dataset_name mnli \
                        --random_seed 17 \
                        --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/mnli/17  \
                        --cache_dir /scratch/yk2516/cache
# ---------------------------------------------------------------------------------------------------------------- #


'
# # Finetune on MNLI (Works!) - 10 Hrs per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path bert-base-uncased \
#                         --dataset_name mnli \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/mnli/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on SNLI (Works!) - 15 Hrs per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/snli/17/bert_snli_finetune \
#                         --dataset_name snli \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/snli/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #


# # Finetune on RTE (Works!) - 5 minutes per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path bert-base-uncased \
#                         --dataset_name rte \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/rte/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on QQP (Works!) - 10 Hrs per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path bert-base-uncased \
#                         --dataset_name qqp \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/qqp/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on SNLI (Works!) - 15 Hrs per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path bert-base-uncased \
#                         --dataset_name snli \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/snli/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on MNLI (Works!) - 10 Hrs per scripts
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --model_and_tokenizer_path bert-base-uncased \
#                         --dataset_name mnli \
#                         --random_seed 17 \
#                         --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/mnli/17  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #


# # SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# # Finetune on IMDB
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_finetune.py --dataset_name imdb --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/17 
# python run_finetune.py --dataset_name imdb --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/42 
# python run_finetune.py --dataset_name imdb --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/imdb/83
# # ---------------------------------------------------------------------------------------------------------------- #

# ***** Target ***** #

# IMDB | SST-2 
# # ---------------------------------------------------------------------------------------------------------------- #
# # Seed 17-17
# python run_finetune.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_finetune_output/imdb-sst2/17-17
# # ---------------------------------------------------------------------------------------------------------------- #

# IMDB | Yelp (Every one of the script below takes roughly 18hrs - Just use one seed then)
# # ---------------------------------------------------------------------------------------------------------------- #
# # Seed 17-x
# python run_finetune.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_finetune_output/imdb-yelp/17-17
# # ---------------------------------------------------------------------------------------------------------------- #

# SST2 | Yelp (Every one of the script below takes roughly 18hrs - Just use one seed then)
# # ---------------------------------------------------------------------------------------------------------------- #
# # Seed 17-17
# python run_finetune.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_finetune_output/sst2-yelp/17-17
# # ---------------------------------------------------------------------------------------------------------------- #


# ***** Source ***** #

# # Finetune on Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_finetune.py --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/17 
# python run_finetune.py --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/42 
# python run_finetune.py --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/83
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on IMDB
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_finetune.py --dataset_name imdb --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17 
# python run_finetune.py --dataset_name imdb --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42 
# python run_finetune.py --dataset_name imdb --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on SST-2
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_finetune.py --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17 
# python run_finetune.py --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42 
# python run_finetune.py --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83
# # ---------------------------------------------------------------------------------------------------------------- #
