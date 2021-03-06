'''
Author: Yilun Kuang
Date: Jan 12, 2021
status: IN-PROGRESS

This script is used to finetune the bert-base-uncased model on the source domain datasets (i.i.d) 
using the classification objective. 
'''
import math
import argparse
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, set_seed

def main(args):
    column_dict = {'imdb':'text','sst2':'sentence','yelp_polarity':'text'}
    glue_lst = ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    def tokenize_function(examples):
        return tokenizer(examples[column_dict[args.dataset_name]], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        
    set_seed(args.random_seed)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')

    if args.dataset_name in glue_lst:
        raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    else: 
        raw_datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if "validation" not in raw_datasets.keys():
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["test"]
    else:
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["validation"]

    metric = load_metric("accuracy")
    training_args = TrainingArguments(output_dir=args.output_dir+"/bert_"+args.dataset_name+"_finetune", overwrite_output_dir=True)
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=full_train_dataset, 
        eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    trainer.save_model()

    try:
        metrics = trainer.evaluate()
        # save results
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    except:
        print(" *** Evaluation cannot be done due to errors! *** ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')
    parser.add_argument("--model_and_tokenizer_path", type=str, default='bert-base-uncased')

    args = parser.parse_args()

    main(args)
