#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_sst2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/adapter_output/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - Run BERT-Adapter Training on SST-2 datasets"
source /ext3/env.sh
conda activate
python run_glue.py --model_name_or_path /scratch/yk2516/UDA_Text_Generation/pretrain_output/checkpoint-final --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 1e-4 --num_train_epochs 3 --output_dir /scratch/yk2516/UDA_Text_Generation/adapter_output --train_adapter --adapter_config pfeiffer --cache_dir /scratch/yk2516/cache'



