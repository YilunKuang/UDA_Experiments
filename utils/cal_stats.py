import json
import argparse

# adapter | file_dir
# '/scratch/yk2516/UDA_Text_Generation/target_adapter_output/sst2-yelp'
# benchmark | file_dir
# '/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp'
parser = argparse.ArgumentParser()
parser.add_argument("--file_dir", type=str, 
                    default='/scratch/yk2516/UDA_Text_Generation/target_adapter_output/imdb-yelp')
parser.add_argument("--dataset_name", type=str, default='yelp_polarity')
parser.add_argument("--type", type=str, default='adapter')
args = parser.parse_args()

eval_acc = 0
eval_loss = 0
eval_ppl = 0
seed_lst = [17, 42, 83]
counter = 0

if args.type == 'adapter':
    for i_1 in seed_lst:
        for i_2 in seed_lst:
            for i_3 in seed_lst:
                try:
                    with open(args.file_dir+'/'+str(i_1)+'-'+str(i_2)+'-'+str(i_3)+'-'+str(17)+'/'+args.dataset_name+'_target/eval_results.json') as f:
                        data = json.load(f)
                        eval_acc += data['eval_accuracy']
                        eval_loss += data['eval_loss']
                        eval_ppl += data['perplexity']
                    counter += 1
                except:
                    print(f"the for loop break. counter is equal to {counter}.")
                    break
else:
    for i_1 in seed_lst:
        for i_2 in seed_lst:
            with open(args.file_dir+'/'+str(i_1)+'-'+str(i_2)+'/'+args.dataset_name+'_test/eval_results.json') as f:
                data = json.load(f)
                eval_acc += data['eval_accuracy']
                eval_loss += data['eval_loss']
                eval_ppl += data['perplexity']
            counter += 1

eval_acc = eval_acc/counter
eval_loss = eval_loss/counter
eval_ppl = eval_ppl/counter

print(f'eval_acc={eval_acc}')
print(f'eval_loss={eval_loss}')
print(f'eval_ppl={eval_ppl}')

