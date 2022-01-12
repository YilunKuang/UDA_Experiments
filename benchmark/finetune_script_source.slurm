#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=finetune_bert_source
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-7.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Finetuning on Source datasets"
source /ext3/env.sh
conda activate

# Finetune on Yelp
# ---------------------------------------------------------------------------------------------------------------- #
python run_source_finetune.py --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/17 
# ---------------------------------------------------------------------------------------------------------------- #
'

# # SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# # Finetune on Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_source_finetune.py --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/17 
# python run_source_finetune.py --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/42 
# python run_source_finetune.py --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/yelp_polarity/83
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on IMDB
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_source_finetune.py --dataset_name imdb --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17 
# python run_source_finetune.py --dataset_name imdb --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42 
# python run_source_finetune.py --dataset_name imdb --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83
# # ---------------------------------------------------------------------------------------------------------------- #

# # Finetune on SST-2
# # ---------------------------------------------------------------------------------------------------------------- #
# python run_source_finetune.py --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17 
# python run_source_finetune.py --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42 
# python run_source_finetune.py --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83
# # ---------------------------------------------------------------------------------------------------------------- #
