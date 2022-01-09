'''
Author: Yilun Kuang
Date: Jan 2, 2021
status: COMPLETE

This script is used to finetune the bert-base-uncased model on the imdb datasets (i.i.d) 
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
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        
    set_seed(args.random_seed)

    raw_datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, cache_dir=args.cache_dir)

    training_args = TrainingArguments(output_dir=args.output_dir+"/bert_imdb_finetune", overwrite_output_dir=True)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset
    )
    train_result = trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='imdb')
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--output_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17')
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')

    args = parser.parse_args()

    main(args)
