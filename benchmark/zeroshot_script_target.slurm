#!/bin/bash
###SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --job-name=zeroshot_bert_target
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/UDA_Text_Generation/slurm_file/slurm_train_%j.out

singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
source /ext3/env.sh
conda activate

# MNLI | SNLI
# ---------------------------------------------------------------------------------------------------------------- #
# 17-17

zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output

for VARIABLE_I in 17 42 83
do
    python run_benchmark.py --do_eval \
                            --track_logits \
                            --model_seed 17 \
                            --logits_dir $logits_path/mnli-snli/17-$VARIABLE_I/logits \
                            --model_and_tokenizer_path $zeroshot_path/mnli/17/bert_mnli_finetune \
                            --dataset_name snli \
                            --random_seed $VARIABLE_I \
                            --output_dir $output_path/mnli-snli/17-$VARIABLE_I/results  \
                            --cache_dir /scratch/yk2516/cache
done
# ---------------------------------------------------------------------------------------------------------------- #

# SNLI | MNLI
# ---------------------------------------------------------------------------------------------------------------- #
# 17-17

# zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
# logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
# output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output

for VARIABLE_I in 17 42 83
do
    python run_benchmark.py --do_eval \
                            --track_logits \
                            --model_seed 17 \
                            --logits_dir $logits_path/snli-mnli/17-$VARIABLE_I/logits \
                            --model_and_tokenizer_path $zeroshot_path/snli/17/bert_snli_finetune \
                            --dataset_name mnli \
                            --random_seed $VARIABLE_I \
                            --output_dir $output_path/snli-mnli/17-$VARIABLE_I/results  \
                            --cache_dir /scratch/yk2516/cache
done
# ---------------------------------------------------------------------------------------------------------------- #


'

# # MNLI | SNLI
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17

# zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
# logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
# output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output

# for VARIABLE_I in 17 42 83
# do
#     python run_benchmark.py --do_eval \
#                             --track_logits \
#                             --model_seed 17 \
#                             --logits_dir $logits_path/mnli-snli/17-$VARIABLE_I/logits \
#                             --model_and_tokenizer_path $zeroshot_path/mnli/17/bert_mnli_finetune \
#                             --dataset_name snli \
#                             --random_seed $VARIABLE_I \
#                             --output_dir $output_path/mnli-snli/17-$VARIABLE_I/results  \
#                             --cache_dir /scratch/yk2516/cache
# done
# # ---------------------------------------------------------------------------------------------------------------- #

# # SNLI | MNLI
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17

# # zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
# # logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
# # output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output

# for VARIABLE_I in 17 42 83
# do
#     python run_benchmark.py --do_eval \
#                             --track_logits \
#                             --model_seed 17 \
#                             --logits_dir $logits_path/snli-mnli/17-$VARIABLE_I/logits \
#                             --model_and_tokenizer_path $zeroshot_path/snli/17/bert_snli_finetune \
#                             --dataset_name mnli \
#                             --random_seed $VARIABLE_I \
#                             --output_dir $output_path/snli-mnli/17-$VARIABLE_I/results  \
#                             --cache_dir /scratch/yk2516/cache
# done
# # ---------------------------------------------------------------------------------------------------------------- #

# # MNLI | RTE - Failed (See notion)
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17

# zeroshot_path=/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output
# logits_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output
# output_path=/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output

# python run_benchmark.py --do_eval \
#                         --track_logits \
#                         --model_seed 17 \
#                         --logits_dir $logits_path/mnli-rte/17-17/logits \
#                         --model_and_tokenizer_path $zeroshot_path/mnli/17/bert_mnli_finetune \
#                         --dataset_name rte \
#                         --random_seed 17 \
#                         --output_dir $output_path/mnli-rte/17-17/results  \
#                         --cache_dir /scratch/yk2516/cache
# # ---------------------------------------------------------------------------------------------------------------- #




# SINGULARITY
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
# singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-0.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# *********** Standard UDA *********** #

# SST2 | Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17-x

# for VARIABLE_J in 17 42 83
# do
#     python run_benchmark.py --model_seed 17 \
#                                 --model_and_tokenizer_path $mlm_model_path/sst2-yelp/17-17 \
#                                 --logits_dir $uda_result_path/sst2-yelp \
#                                 --dataset_name yelp_polarity \
#                                 --random_seed $VARIABLE_J \
#                                 --output_dir $uda_result_path/sst2-yelp/17-17-$VARIABLE_J
# done

# # ---------------------------------------------------------------------------------------------------------------- #

# IMDB | Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-42-x

# for VARIABLE_J in 17 42 83
# do
#     python run_benchmark.py --model_seed 17 \
#                                 --model_and_tokenizer_path $mlm_model_path/imdb-yelp/17-42 \
#                                 --logits_dir $uda_result_path/imdb-yelp \
#                                 --dataset_name yelp_polarity \
#                                 --random_seed $VARIABLE_J \
#                                 --output_dir $uda_result_path/imdb-yelp/17-42-$VARIABLE_J
# done

# # ---------------------------------------------------------------------------------------------------------------- #

# IMDB | SST-2
# # ---------------------------------------------------------------------------------------------------------------- #
# # 17-17-x

# for VARIABLE_J in 17 42 83
# do
#     python run_benchmark.py --model_seed 17 \
#                                 --model_and_tokenizer_path $mlm_model_path/imdb-sst2/17-17 \
#                                 --logits_dir $uda_result_path/imdb-sst2 \
#                                 --dataset_name sst2 \
#                                 --random_seed $VARIABLE_J \
#                                 --output_dir $uda_result_path/imdb-sst2/17-17-$VARIABLE_J
# done

# # ---------------------------------------------------------------------------------------------------------------- #

# *********** Zeroshot *********** #

# SST2 | Yelp
# # ---------------------------------------------------------------------------------------------------------------- #
# 17-x
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/17-17
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/17-42
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/17/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/17-83

# 42-x
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/42-17
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/42-42
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/42/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/42-83

# 83-x
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/83-17
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/83-42
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/sst2/83/bert_sst2_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp/83-83
# # ---------------------------------------------------------------------------------------------------------------- #

# IMDB | Yelp
# ---------------------------------------------------------------------------------------------------------------- #
# 17-x
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-17
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-42
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/17-83

# 42-x
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-17
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-42
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/42-83

# 83-x
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-17
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-42
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp --dataset_name yelp_polarity --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp/83-83
# ---------------------------------------------------------------------------------------------------------------- #

# IMDB | SST-2
# ---------------------------------------------------------------------------------------------------------------- #
# 17-x
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-17
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-42
# python run_benchmark.py --model_seed 17 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17-83

# 42-x
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-17
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-42
# python run_benchmark.py --model_seed 42 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/42/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/42-83

# 83-x
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 17 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-17
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 42 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-42
# python run_benchmark.py --model_seed 83 --model_and_tokenizer_path /scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/83/bert_imdb_finetune --logits_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output --dataset_name sst2 --random_seed 83 --output_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/83-83
# ---------------------------------------------------------------------------------------------------------------- #

