#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --job-name=self_training_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-1.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
source /ext3/env.sh
conda activate

model_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/logits_output
output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_self_train_output

# IMDB | Yelp
# ---------------------------------------------------------------------------------------------------------------- #
# 17-17

python run_benchmark.py --do_train \
                        --do_eval \
                        --self_training \
                        --indices_dir $zeroshot_path/imdb-yelp \
                        --model_and_tokenizer_path $model_path/imdb/17/bert_imdb_finetune \
                        --model_seed 17 \
                        --dataset_name yelp_polarity \
                        --random_seed 17 \
                        --logits_dir $logits_path/benchmark3/imdb-yelp/17-17 \
                        --output_dir $output_path/imdb-yelp/17-17 \
                        --cache_dir /scratch/yk2516/cache \
                        --sanity_logit_file_name logits17_17.txt
# ---------------------------------------------------------------------------------------------------------------- #


'

# # SST2 | Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17

# python run_benchmark.py --do_train \
#                         --do_eval \
#                         --self_training \
#                         --indices_dir $zeroshot_path/sst2-yelp \
#                         --model_and_tokenizer_path $model_path/sst2/17/bert_sst2_finetune \
#                         --model_seed 17 \
#                         --dataset_name yelp_polarity \
#                         --random_seed 17 \
#                         --logits_dir $logits_path/benchmark3/sst2-yelp/17-17 \
#                         --output_dir $output_path/sst2-yelp/17-17 \
#                         --cache_dir /scratch/yk2516/cache \
#                         --sanity_logit_file_name logits17_17.txt
# # ---------------------------------------------------------------------------------------------------------------- #


# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash


