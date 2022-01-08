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

    raw_datasets = load_dataset("imdb", cache_dir='/scratch/yk2516/cache')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir='/scratch/yk2516/cache')
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, cache_dir='/scratch/yk2516/cache')

    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset
    )
    train_result = trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=17)
    args = parser.parse_args()

    main(args)
