#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --job-name=adapter_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-7.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
echo "Running - Run BERT Adapter Inference on Target datasets"
source /ext3/env.sh
conda activate

# ****** IMDB | Yelp ****** #

# 83 Start
# 83-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-17-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-42-17
python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-83-17

'
# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# ****** IMDB | Yelp ****** #
# ## -------------------------------------------------------------------------------------------------------------- ##
# # 17 Start
# # 17-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-17-83-17

# # 17-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-42-83-17

# # 17-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/17-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/17-83-83-17

# # 42 Start
# # 42-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-17-83-17

# # 42-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-42-83-17

# # 42-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/42-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/42-83-83-17

# # 83 Start
# # 83-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-17/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-17-83-17

# # 83-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-42/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-42-83-17

# # 83-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-42  --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/83-83/final_adapter --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp/83-83-83-17


# ****** SST-2 | Yelp ****** #
# ## -------------------------------------------------------------------------------------------------------------- ##
# # 17 Start
# # 17-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-17-83-17

# # 17-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-42-83-17

# # 17-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/17-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-17-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/17-83-83-17

# # 42 Start
# # 42-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-17-83-17

# # 42-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-42-83-17

# # 42-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/42-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-42-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/42-83-83-17

# # 83 Start
# # 83-17 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-17-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-17-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-17/sst2_source17/sst2_source17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-17-83-17

# # 83-42 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-42-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-42-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-42/sst2_source42/sst2_source42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-42-83-17

# # 83-83 (the 4th random seed is kept at 17 as it doesnt change the eval_result during the adapter inference phase)
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-17 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-83-17-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-42 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-83-42-17
# python run_adapter_inference.py --adapters_dir /scratch/yk2516/UDA_Text_Generation/source_adapter_output/sst2/83-83/sst2_source83/sst2_source83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/target_pretrain_output/sst2-yelp/checkpoint-83-random-seed-83 --dataset_name yelp_polarity --output_dir /scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp/83-83-83-17
# ## -------------------------------------------------------------------------------------------------------------- ##

# ****** IMDB | SST-2 ****** #
# ## -------------------------------------------------------------------------------------------------------------- ##
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
# ## -------------------------------------------------------------------------------------------------------------- ##


