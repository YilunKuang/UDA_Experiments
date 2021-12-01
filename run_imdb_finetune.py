import math
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

# --do_train --do_eval --output_dir /scratch/yk2516/UDA_Text_Generation/source_finetune/checkpoint-final

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    raw_datasets = load_dataset("imdb", cache_dir='/scratch/yk2516/cache')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir='/scratch/yk2516/cache')
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir='/scratch/yk2516/cache')
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset
    )
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    metric = load_metric("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
