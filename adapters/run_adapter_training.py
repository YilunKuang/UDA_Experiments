import math
import argparse
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithHeads, set_seed
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import EarlyStoppingCallback

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
        #return tokenizer(examples["text"], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    set_seed(args.random_seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
        print('*** The tokenizer in use is bert-base-uncased ***')
    model = AutoModelWithHeads.from_pretrained(args.model_and_tokenizer_path, cache_dir=args.cache_dir)
    
    if args.dataset_name in glue_lst:
        dataset = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    else: 
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    if "validation" not in dataset.keys():
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
        else:
            full_train_dataset = tokenized_datasets["train"]
            full_eval_dataset = tokenized_datasets["validation"]    

    model.add_adapter(args.dataset_name+"_source"+str(args.random_seed))
    model.add_classification_head(args.dataset_name+"_source"+str(args.random_seed), num_labels=num_labels)
    model.train_adapter(args.dataset_name+"_source"+str(args.random_seed))
    training_args = TrainingArguments(output_dir=args.output_dir+args.dataset_name+"_source"+str(args.random_seed), 
                                      overwrite_output_dir=True, 
                                      evaluation_strategy='steps', 
                                      eval_steps=500, 
                                      save_total_limit=5, 
                                      num_train_epochs=3, 
                                      greater_is_better=True, 
                                      metric_for_best_model='eval_accuracy', 
                                      load_best_model_at_end=True)
    trainer = AdapterTrainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset,compute_metrics=compute_metrics,callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    train_result = trainer.train()
    trainer.save_model()

    
    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    try:
        model.save_adapter(args.output_dir+"final_adapter", args.dataset_name+"_source"+str(args.random_seed))
    except:
        model.save_adapter("./final_adapter", args.dataset_name+"_source"+str(args.random_seed))
        print("*** The output is saved in ./final adapter ***")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_and_tokenizer_path", type=str)
    parser.add_argument("--cache_dir", type=str, default='/scratch/yk2516/cache')
    
    
    args = parser.parse_args()

    main(args)

