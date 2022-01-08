#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_imdb
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Adapter Training on IMDB datasets"
source /ext3/env.sh
conda activate

python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 17 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/ --cache_dir /scratch/yk2516/cache
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 42 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/ --cache_dir /scratch/yk2516/cache
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-42 --random_seed 83 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/ --cache_dir /scratch/yk2516/cache

python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 17 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/ --cache_dir /scratch/yk2516/cache
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 42 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/ --cache_dir /scratch/yk2516/cache
python run_adapter_training.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-random-seed-83 --random_seed 83 --dataset_name imdb --output_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/ --cache_dir /scratch/yk2516/cache

'

