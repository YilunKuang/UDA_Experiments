#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_sst2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Adapter Inference on SST-2 datasets"
source /ext3/env.sh
conda activate

python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-17-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-42-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-83-17

python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-17-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-42-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-83-17

python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-17-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-42-17
python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-83-17

python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-17-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-42-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-83-17

python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-17-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-42-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-83-17

python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-17-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-42-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-83-17


'

# # 17 Start
# # 17-17 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42  --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-83-17

# # 17-42 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-42-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-42-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-42-83-17

# # 17-83 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-83-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-83-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-83-83-17

# # 42 Start
# # 42-17 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-17-83-17

# # 42-42 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-42-83-17

# # 42-83 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-17-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-42-17
# python run_adapter_inference.py --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/42-83-83-17

# # 83 Start
# # 83-17 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-17-83-17

# # 83-42 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-42-83-17

# # 83-83 (the 4th random seed is kept at 17 as it doesn't change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/83-83-83-17


