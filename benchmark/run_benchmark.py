'''
Author: Yilun Kuang
Date: Jan 12, 2021
status: IN-PROGRESS

This script is used to evaluated the bert-base-uncased model that is already finetuned on the in-domain source datasets (i.i.d)
using the classification objective on the out-of-domain target dataset. 

Baseline 1
    Standard fine-tuning on source domain, zero-shot test on target domain.

Baseline 2
    Standard fine-tuning on source domain, MLM fine-tuning on target domain, then test on target domain.

Baseline 3
    Standard fine-tuning on source domain, zeroshot test on target domain. 
    Use the confident labels as pseudo labels then standard finetune the model on the target domain with pseudo labels, 
    test on target domain

'''
import math
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from datasets import Dataset
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, set_seed
from transformers import EarlyStoppingCallback

def main(args):
    column_dict = {'imdb':'text','sst2':'sentence','yelp_polarity':'text'}
    column_dict_label = {'yelp_polarity':'label','imdb':'label','sst2':'label'}

    glue_lst = ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    
    nli_dict = {'mnli':['premise','hypothesis'],
                'snli':['premise','hypothesis'],
                'rte':['sentence1','sentence2'],
                'qqp':['question1','question2']}

    triple_label_lst = ['mnli','snli']
    if args.dataset_name in triple_label_lst:
        num_labels = 3
    else:
        num_labels = 2

    def tokenize_function(examples):
        if args.dataset_name in nli_dict:
            return tokenizer(examples[nli_dict[args.dataset_name][0]], examples[nli_dict[args.dataset_name][1]], padding="max_length", truncation=True)
        else:
            return tokenizer(examples[column_dict[args.dataset_name]], padding="max_length", truncation=True)
    def compute_metrics_eval(eval_pred):
        softmax_func = nn.Softmax(dim=1)
        logits, labels = eval_pred

        if args.track_logits:
            for i in range(len(logits)):
                f_logits.writelines(str(logits[i])+'\n')
            f_labels.writelines(str(labels))
            
            logits_tensor = torch.tensor(logits)
            logits_prob = softmax_func(logits_tensor)
            for i in range(len(logits_prob)):
                f_logits_prob.writelines(str(logits_prob[i])+'\n')

        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def compute_metrics_train(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    set_seed(args.random_seed)

    # Initialize model and dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir, num_labels=num_labels)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')

    if args.dataset_name in glue_lst:
        raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    else:
        raw_datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    
    raw_keys = raw_datasets.keys()

    if args.self_training:
        print(f"The new training dataset size is {raw_datasets['train'].num_rows}")
        ind_high_confidence = np.load(args.indices_dir+"/conf_indices.npy").tolist()
        pseudo_label = np.load(args.indices_dir+"/pseudolabels.npy").tolist()

        if "validation" not in raw_keys:
            raw_train_datasets = raw_datasets['test'].select(ind_high_confidence)
            eval_num = raw_datasets['test'].num_rows
        else:
            raw_train_datasets = raw_datasets['validation'].select(ind_high_confidence)
            eval_num = raw_datasets['validation'].num_rows
        
        raw_eval_datasets = raw_datasets['train'][0:eval_num]
        df_eval = pd.DataFrame(raw_eval_datasets)
        raw_eval_datasets = Dataset.from_pandas(df_eval)

        df = pd.DataFrame({column_dict[args.dataset_name]:raw_train_datasets[column_dict[args.dataset_name]],\
                           column_dict_label[args.dataset_name]:pseudo_label})
        raw_train_datasets = Dataset.from_pandas(df)

        # Now the original training set becomes the eval set, the eval set becomes the slices of the original training set.
        if "validation" not in raw_keys:
            raw_datasets['test'] = raw_eval_datasets
        else:
            raw_datasets['validation'] = raw_eval_datasets
        raw_datasets['train'] = raw_train_datasets

        print(f"The new training dataset size after selection is {raw_datasets['train'].num_rows}")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if "validation" not in raw_keys:
        if args.dataset_name == 'mnli':
            # Use MNLI (Train) & MNLI (Dev)
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["validation_mismatched"]
        else: 
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["test"]
    else:
        if args.dataset_name == 'snli':
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["validation"]

            full_train_dataset = full_train_dataset.filter(lambda example: example['label']!=-1)
            full_eval_dataset = full_eval_dataset.filter(lambda example: example['label']!=-1)
        else:
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["validation"]    


    metric = load_metric("accuracy")


    if args.do_train:
        # training_args = TrainingArguments(output_dir=args.output_dir+"/bert_"+args.dataset_name+"_finetune", \
        #                                   overwrite_output_dir=True,
        #                                   load_best_model_at_end=True
        #                                   metric_for_best_model="eval_loss",
        #                                   greater_is_better=False,)
        training_args = TrainingArguments(output_dir=args.output_dir+"/bert_"+args.dataset_name+"_finetune", \
                                        overwrite_output_dir=True,
                                        evaluation_strategy ='steps',
                                        eval_steps = 500, 
                                        save_total_limit = 5,
                                        num_train_epochs=3,
                                        greater_is_better=True,
                                        metric_for_best_model = 'accuracy',
                                        load_best_model_at_end=True)

        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=full_train_dataset, 
            eval_dataset=full_eval_dataset,
            compute_metrics=compute_metrics_train,
            tokenizer=tokenizer
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )

        train_result = trainer.train()
        trainer.save_model()
        try:
            trainer.save_state()
        except:
            print(" *** Save state failed ***")

    if args.do_eval:
        if args.track_logits:
            f_logits = open(args.logits_dir+"/logits"+args.model_seed+"_"+str(args.random_seed)+".txt", "a")
            f_logits_prob = open(args.logits_dir+"/logits_prob"+args.model_seed+"_"+str(args.random_seed)+".txt", "a")
            f_labels = open(args.logits_dir+"/gold_label"+args.model_seed+"_"+str(args.random_seed)+".txt","a")

        # Evaluation
        training_args = TrainingArguments(output_dir=args.output_dir+"/"+args.dataset_name+"_test", overwrite_output_dir=True)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=full_train_dataset,
            eval_dataset=full_eval_dataset,
            compute_metrics=compute_metrics_eval,
        )
        metrics = trainer.evaluate()
        
        if args.track_logits:
            f_logits.close()
            f_logits_prob.close()
            f_labels.close()

        # save results
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # ------------------------------------------------------------ #
    parser.add_argument("--self_training",action="store_true")
    parser.add_argument("--indices_dir",type=str,default="None")
    # ------------------------------------------------------------ #

    # ------------------------------------------------------------ #
    parser.add_argument("--track_logits",action="store_true")
    parser.add_argument("--model_seed",type=str,default='17')
    parser.add_argument("--logits_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output')
    # ------------------------------------------------------------ #

    parser.add_argument("--model_and_tokenizer_path", type=str)
    parser.add_argument("--dataset_name", type=str, default='sst2')
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--output_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17')
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')
    args = parser.parse_args()

    main(args)
