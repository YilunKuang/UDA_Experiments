#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --job-name=zeroshot_bert_sst2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - Run BERT Zeroshot on SST2 datasets"
source /ext3/env.sh
conda activate
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-17
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-42
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-83

python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-17
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-42
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-83

python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-17
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-42
python run_sst2_evaluate.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-83'

