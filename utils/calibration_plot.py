import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.calibration import calibration_curve

def main(args):
    adapter_path = args.adapter_path
    benchmark_path = args.benchmark_path
    dataset_pair = args.dataset_pair
    
    file_name_dict = {"mnli-snli": {"adapter":['logits_prob.txt','gold_label.txt'],
                                   "benchmark":['logits_prob17_17.txt','gold_label17_17.txt']},
                      "snli-mnli": None}

    with open(adapter_path+'/'+file_name_dict[dataset_pair]['adapter'][0]) as f_adapter_prob:
        lines_adapter_prob = f_adapter_prob.readlines()
        lines_adapter_prob = list(map(lambda x: x[:-1].strip('][').split(', '),lines_adapter_prob))
        if "mnli" in dataset_pair or "snli" in dataset_pair:
            lines_adapter_prob = list(map(lambda x: [x[0].split('tensor([')[1] if 'tensor' in x[0] else x[0], x[1], x[2]],lines_adapter_prob))
            lines_adapter_prob = list(map(lambda x: [x[0], x[1], x[2].split('])')[0] if '])' in x[2] else x[2]],lines_adapter_prob))        
            lines_adapter_prob = list(map(lambda x: [float(x[0]),float(x[1]),float(x[2])],lines_adapter_prob))

    with open(benchmark_path+'/'+file_name_dict[dataset_pair]['benchmark'][0]) as f_benchmark_prob:
        lines_benchmark_prob = f_benchmark_prob.readlines()
        lines_benchmark_prob = list(map(lambda x: x[:-1].strip('][').split(', '),lines_benchmark_prob))
        if "mnli" in dataset_pair or "snli" in dataset_pair:
            lines_benchmark_prob = list(map(lambda x: [x[0].split('tensor([')[1] if 'tensor' in x[0] else x[0], x[1], x[2]],lines_benchmark_prob))
            lines_benchmark_prob = list(map(lambda x: [x[0], x[1], x[2].split('])')[0] if '])' in x[2] else x[2]],lines_benchmark_prob))        
            lines_benchmark_prob = list(map(lambda x: [float(x[0]),float(x[1]),float(x[2])],lines_benchmark_prob))
            # print(lines_benchmark_prob)
    
    pred_adapter = np.argmax(lines_adapter_prob, axis=-1)
    pred_benchmark = np.argmax(lines_benchmark_prob, axis=-1)

    
    # print(pred_adapter)
    print(len(pred_adapter))
    # print(len(lines_adapter_prob))
        
    # print(pred_benchmark)
    print(len(pred_benchmark))
    # print(len(lines_benchmark_prob))

    conf_adapter = list(map(lambda x,y: x[y], lines_adapter_prob, pred_adapter.tolist()))
    conf_benchmark = list(map(lambda x,y: x[y], lines_benchmark_prob, pred_benchmark.tolist()))

    inference_set = load_dataset('snli',cache_dir='/scratch/yk2516/cache')
    inference_set['validations']['labels']
    
    with open(adapter_path+'/'+file_name_dict[dataset_pair]['adapter'][1]) as f_label:
        lines_label = ""
        for readline in f_label:
            line_strip = readline.rstrip('\n')
            lines_label += line_strip

        print(len(lines_label))

        lines_label = list(map(lambda x: int(x),lines_label.split()))
        # print(lines_label)

    sys.exit()

    prob_true_adapter, prob_pred_adapter = calibration_curve(np.array(lines_label), np.array(lines_adapter_prob)[:,1], n_bins=10)
    prob_true_benchmark, prob_pred_benchmark = calibration_curve(np.array(lines_label), np.array(lines_benchmark_prob)[:,1], n_bins=10)

    plt.figure()
    plt.plot(prob_pred_adapter,prob_true_adapter,marker='s',label='adapter')
    plt.plot(prob_pred_benchmark,prob_true_benchmark,marker='s',label='benchmark')
    plt.plot([0,1],[0,1],'--',color='0.8')
    plt.title(f'Calibration Plot for {dataset_pair}')
    plt.xlabel('the mean predicted probability for elements of the bin')
    plt.ylabel('the fraction of elements in the bin \n for which the event actually occurs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pair", type=str)
    parser.add_argument("--adapter_path", type=str)
    parser.add_argument("--benchmark_path", type=str)
    args = parser.parse_args()
    main(args)

# ******************** mnli-snli ******************** #
# --------------------------------------------------- #
# python calibration_plot.py --dataset_pair mnli-snli --adapter_path /scratch/yk2516/UDA_Text_Generation/target_adapter_output/mnli-snli/17-17-17-17 --benchmark_path /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/mnli-snli/17-17/logits
                           