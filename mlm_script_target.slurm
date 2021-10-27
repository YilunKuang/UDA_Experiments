#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=mlm_bert_sst2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/target_pretrain_output/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - Run BERT MLM Training on SST-2 datasets"
source /ext3/env.sh
conda activate
python run_mlm_target.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-final --dataset_name sst --dataset_config_name default --do_train --do_eval --output_dir /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-final --cache_dir /scratch/yk2516/cache'


