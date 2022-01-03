'''
Author: Yilun Kuang
Date: Jan 2, 2021
status: COMPLETE

This script is used to evaluated the bert-base-uncased model that is already finetuned on the imdb datasets (i.i.d)
using the classification objective on the sst2 dataset from the GLUE benchmark. 
'''
import math
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

def main():
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        print(logits)
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Initialize model and dataset
    model = AutoModelForSequenceClassification.from_pretrained('/scratch/yk2516/UDA_Text_Generation/benchmark/vanilla_bert_finetuned_on_imdb/test_trainer',
                                                                num_labels=2, cache_dir='/scratch/yk2516/cache')
    raw_datasets = load_dataset("glue", "sst2", cache_dir='/scratch/yk2516/cache')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir='/scratch/yk2516/cache')
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["validation"]

    # Evaluation
    training_args = TrainingArguments("sst2_test")
    metric = load_metric("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()

    # save results
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    main()
