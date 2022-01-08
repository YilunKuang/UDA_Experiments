#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_sst2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-17-17/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-1.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Adapter Inference on SST-2 datasets"
source /ext3/env.sh
conda activate
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --random_seed 17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --dataset_name sst2 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-17-17 --cache_dir /scratch/yk2516/cache'

