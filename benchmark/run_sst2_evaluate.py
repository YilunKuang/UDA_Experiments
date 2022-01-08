'''
Author: Yilun Kuang
Date: Jan 2, 2021
status: COMPLETE

This script is used to evaluated the bert-base-uncased model that is already finetuned on the imdb datasets (i.i.d)
using the classification objective on the sst2 dataset from the GLUE benchmark. 
'''
import math
import torch
import numpy as np
import torch.nn as nn
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, set_seed

def main(args):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
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

    set_seed(args.random_seed)

    # Initialize model and dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.model_and_tokenizer_path, num_labels=2, cache_dir=args.cache_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')

    raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["validation"]

    f_logits = open(args.logits_dir+"/logits"+str(args.random_seed)+".txt", "a")
    f_logits_prob = open(args.logits_dir+"/logits_prob"+str(args.random_seed)+".txt", "a")
    f_labels = open(args.logits_dir+"/gold_label"+str(args.random_seed)+".txt","a")

    # Evaluation
    training_args = TrainingArguments(args.output_dir+"/sst2_test")
    metric = load_metric("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics,
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
    parser.add_argument("--model_and_tokenizer_path", type=str, 
                            default='/scratch/yk2516/UDA_Text_Generation/benchmark/source_finetune_output/17/bert_imdb_finetune')
    parser.add_argument("--dataset_name", type=str, default='sst2')
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--logits_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output')
    parser.add_argument("--output_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/17')
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')
    args = parser.parse_args()

    main(args)
