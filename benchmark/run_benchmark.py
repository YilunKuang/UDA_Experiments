'''
Author: Yilun Kuang
Date: Jan 12, 2021
status: IN-PROGRESS

This script is used to evaluated the bert-base-uncased model that is already finetuned on the in-domain source datasets (i.i.d)
using the classification objective on the out-of-domain target dataset. 
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

def sanity_check(args, ind_high_confidence, pseudo_label, raw_datasets):
    with open(args.indices_dir+"/"+args.sanity_logit_file_name) as f:
        logits_prob = f.readlines()
        logits_prob = list(map(lambda x: x[7:],logits_prob))
        logits_prob = list(map(lambda x: x[:-2],logits_prob))
        logits_prob = list(map(lambda x: x[:-1].strip('][').split(', '),logits_prob))
        logits_prob = list(map(lambda x: [float(x[0]),float(x[1])],logits_prob))

        lst_largest_prob = list(map(lambda x: max(x), logits_prob))
        ind_lst_largest_prob = np.argsort(lst_largest_prob)[::-1]
        ind_lst_largest_prob = ind_lst_largest_prob[:int(len(ind_lst_largest_prob)*0.90)]

        predictions = np.array(np.argmax(logits_prob, axis=-1))
        pseudo_labels = predictions[ind_lst_largest_prob]

        zip_dataset = zip(ind_lst_largest_prob,pseudo_labels)
        zip_dict = dict(zip_dataset)
        dict_items = zip_dict.items()
        sorted_dict = dict(sorted(dict_items))

        for i in range(len(pseudo_label)):
            assert(sorted_dict[ind_high_confidence[i]]==pseudo_label[i])
        assert(raw_datasets['train']['label']==pseudo_label)
        

def main(args):
    column_dict = {'imdb':'text','sst2':'sentence','yelp_polarity':'text'}
    column_dict_label = {'yelp_polarity':'label','imdb':'label','sst2':'label'}

    glue_lst = ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    def tokenize_function(examples):
        return tokenizer(examples[column_dict[args.dataset_name]], padding="max_length", truncation=True)
    def compute_metrics_eval(eval_pred):
        softmax_func = nn.Softmax(dim=1)
        logits, labels = eval_pred

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_and_tokenizer_path, num_labels=2, cache_dir=args.cache_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')

    if args.dataset_name in glue_lst:
        raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    else: 
        raw_datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    
    if args.self_training:
        print(f"The new training dataset size is {raw_datasets['train'].num_rows}")
        ind_high_confidence = np.load(args.indices_dir+"/conf_indices.npy").tolist()
        pseudo_label = np.load(args.indices_dir+"/pseudolabels.npy").tolist()

        raw_train_datasets = raw_datasets['train'].select(ind_high_confidence)
        # df = pd.DataFrame({column_dict[args.dataset_name]:raw_train_datasets[column_dict[args.dataset_name]],\
        #                    column_dict_label[args.dataset_name]:pseudo_label})
        # raw_train_datasets = Dataset.from_pandas(df)
        raw_datasets['train'] = raw_train_datasets

        # sanity check
        # try:
        #     sanity_check(args, ind_high_confidence, pseudo_label, raw_datasets)
        # except:
        #     print(" *** sanity check failed! Check your data *** ")

        print(f"The new training dataset size after selection is {raw_datasets['train'].num_rows}")
        
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if "validation" not in raw_datasets.keys():
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["test"]
    else:
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["validation"]
    
    metric = load_metric("accuracy")

    if args.do_train:
        training_args = TrainingArguments(output_dir=args.output_dir+"/bert_"+args.dataset_name+"_finetune", overwrite_output_dir=True)
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=full_train_dataset, 
            eval_dataset=full_eval_dataset,
            compute_metrics=compute_metrics_train,
            tokenizer=tokenizer,
        )
        train_result = trainer.train()
        trainer.save_model()

    if args.do_eval:
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
    parser.add_argument("--self_training",action="store_true")
    parser.add_argument("--indices_dir",type=str,
            default="/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp")
    parser.add_argument("--model_and_tokenizer_path", type=str)
    parser.add_argument("--model_seed",type=str)
    parser.add_argument("--dataset_name", type=str, default='sst2')
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--logits_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output')
    parser.add_argument("--output_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17')
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')
    parser.add_argument("--sanity_logit_file_name",type=str,default='logits17_17.txt')
    args = parser.parse_args()

    main(args)
