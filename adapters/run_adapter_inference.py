import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithHeads, set_seed
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

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

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')
    
    model = AutoModelWithHeads.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    head_name = model.load_head(args.adapters_dir)
    adapter_name = model.load_adapter(args.adapters_dir)
    model.set_active_adapters(adapter_name)

    raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["validation"]

    f_logits = open(args.output_dir+"/logits.txt", "a")
    f_logits_prob = open(args.output_dir+"/logits_prob.txt", "a")
    f_labels = open(args.output_dir+"/gold_label.txt","a")

    training_args = TrainingArguments(output_dir=args.output_dir+"/sst2_target",
                                        overwrite_output_dir=True)
    trainer = AdapterTrainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset,compute_metrics=compute_metrics
    )

    metric = load_metric("accuracy")
    metrics = trainer.evaluate()
    
    f_logits.close()
    f_logits_prob.close()
    f_labels.close()

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
                            default='/scratch/yk2516/UDA_Text_Generation/target_pretrain_output/checkpoint-17-random-seed-17')
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--adapters_dir", type=str, default='none')
    parser.add_argument("--dataset_name", type=str, default='sst2')
    parser.add_argument("--output_dir",type=str,default='/scratch/yk2516/UDA_Text_Generation/target_adapter_output/17-17-17-17')
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')

    args = parser.parse_args()

    main(args)
