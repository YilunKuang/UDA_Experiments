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
    column_dict = {'imdb':'text','sst2':'sentence','yelp_polarity':'text'}
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
    model = AutoModelWithHeads.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')
    
    head_name = model.load_head(args.adapters_dir)
    adapter_name = model.load_adapter(args.adapters_dir)
    model.set_active_adapters(adapter_name)

    if args.dataset_name in glue_lst:
        raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    else: 
        raw_datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if "validation" not in raw_datasets.keys():
        if args.dataset_name == 'mnli':
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
            # snli_filtered = snli['validation'].filter(lambda example: example['label']!=-1)
        else:
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["validation"]    

    f_logits = open(args.output_dir+"/logits.txt", "a")
    f_logits_prob = open(args.output_dir+"/logits_prob.txt", "a")
    f_labels = open(args.output_dir+"/gold_label.txt","a")
    
    training_args = TrainingArguments(output_dir=args.output_dir+"/"+args.dataset_name+"_target",
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
    # Required
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters_dir", type=str)
    parser.add_argument("--model_and_tokenizer_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--output_dir",type=str)
    
    # Default
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')

    args = parser.parse_args()

    main(args)
