import math
import argparse
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithHeads, set_seed
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

def main(args):
    column_dict = {'imdb':'text','sst2':'sentence'}
    glue_lst = ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    def tokenize_function(examples):
        return tokenizer(examples[column_dict[args.dataset_name]], padding="max_length", truncation=True)
        #return tokenizer(examples["text"], padding="max_length", truncation=True)
    def compute_metrics(eval_pred):
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
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    model.add_adapter(args.dataset_name+"_source"+str(args.random_seed))
    model.add_classification_head(args.dataset_name+"_source"+str(args.random_seed), num_labels=2)
    model.train_adapter(args.dataset_name+"_source"+str(args.random_seed))
    training_args = TrainingArguments(output_dir=args.output_dir+args.dataset_name+"_source"+str(args.random_seed), overwrite_output_dir=True)
    trainer = AdapterTrainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset,compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    trainer.save_model()

    metric = load_metric("accuracy")
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

