#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_imdb
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/adapter_trained_on_source/17-17/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - Run BERT MLM Training on SST-2 datasets"
source /ext3/env.sh
conda activate
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-17 --random_seed 17 --dataset_name imdb --cache_dir /scratch/yk2516/cache'

