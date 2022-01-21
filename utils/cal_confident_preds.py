import argparse
import numpy as np

def main(args):
    seed_lst = [17, 42, 83]

    if "dataset_pair" == "imdb-sst2":
        for i in seed_lst:
            file_name = "logits_prob"+str(i)+".txt"
            with open(args.zeroshot_dir+"/"+file_name) as f:
                logits_prob = f.readlines()
                # convert logits_prob into a list of probability vectors. 
                logits_prob = list(map(lambda x: x[7:],logits_prob))
                logits_prob = list(map(lambda x: x[:-2],logits_prob))
                logits_prob = list(map(lambda x: x[:-1].strip('][').split(', '),logits_prob))
                logits_prob = list(map(lambda x: [float(x[0]),float(x[1])],logits_prob))
                
                # check logits statistics
                lst_largest_prob = list(map(lambda x: max(x), logits_prob))
                ind_lst_largest_prob = np.argsort(lst_largest_prob)[::-1]
                ind_lst_largest_prob = ind_lst_largest_prob[:int(len(ind_lst_largest_prob)*0.90)]

                predictions = np.array(np.argmax(logits_prob, axis=-1))
                pseudo_labels = predictions[ind_lst_largest_prob]

                zip_dataset = zip(ind_lst_largest_prob,pseudo_labels)
                zip_dict = dict(zip_dataset)
                dict_items = zip_dict.items()
                sorted_dict = dict(sorted(dict_items))
                
                key_lst = list(sorted_dict.keys())
                value_lst = list(sorted_dict.values())

                for k in range(10):
                    print(f"{sorted_dict[key_lst[k]]}=={logits_prob[key_lst[k]]} & {lst_largest_prob[ind_lst_largest_prob[k]]}")
                
                with open(args.zeroshot_dir+'/'+args.output_file_name, 'wb') as f_ind:
                    np.save(f_ind, np.array(key_lst))
                with open(args.zeroshot_dir+'/'+args.output_label_name, 'wb') as f_label:
                    np.save(f_label, np.array(value_lst))


                with open(args.zeroshot_dir+'/'+args.output_file_name, 'wb') as f_ind:
                    np.save(f_ind, ind_lst_largest_prob)

    else:
        for i in seed_lst:
            for j in seed_lst:
                file_name = "logits_prob"+str(i)+"_"+str(j)+".txt"
                with open(args.zeroshot_dir+"/"+file_name) as f:
                    logits_prob = f.readlines()
                    # convert logits_prob into a list of probability vectors. 
                    logits_prob = list(map(lambda x: x[7:],logits_prob))
                    logits_prob = list(map(lambda x: x[:-2],logits_prob))
                    logits_prob = list(map(lambda x: x[:-1].strip('][').split(', '),logits_prob))
                    logits_prob = list(map(lambda x: [float(x[0]),float(x[1])],logits_prob))
                    
                    # check logits statistics
                    lst_largest_prob = list(map(lambda x: max(x), logits_prob))
                    ind_lst_largest_prob = np.argsort(lst_largest_prob)[::-1]
                    ind_lst_largest_prob = ind_lst_largest_prob[:int(len(ind_lst_largest_prob)*0.90)]
                    
                    predictions = np.array(np.argmax(logits_prob, axis=-1))
                    pseudo_labels = predictions[ind_lst_largest_prob]

                    zip_dataset = zip(ind_lst_largest_prob,pseudo_labels)
                    zip_dict = dict(zip_dataset)
                    dict_items = zip_dict.items()
                    sorted_dict = dict(sorted(dict_items))
                    
                    key_lst = list(sorted_dict.keys())
                    value_lst = list(sorted_dict.values())

                    for k in range(10):
                        print(f"{sorted_dict[key_lst[k]]}=={logits_prob[key_lst[k]]} & {lst_largest_prob[ind_lst_largest_prob[k]]}")
                    
                    with open(args.zeroshot_dir+'/'+args.output_file_name, 'wb') as f_ind:
                        np.save(f_ind, np.array(key_lst))
                    with open(args.zeroshot_dir+'/'+args.output_label_name, 'wb') as f_label:
                        np.save(f_label, np.array(value_lst))

                break
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pair",type=str,default="imdb-yelp")
    parser.add_argument("--zeroshot_dir", type=str, 
                        default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp')
    parser.add_argument("--output_file_name",type=str,default='conf_indices.npy')
    parser.add_argument("--output_label_name",type=str,default='pseudolabels.npy')
    args = parser.parse_args()
    main(args)

# imdb-yelp
# python cal_confident_preds.py

# sst2-yelp
# python cal_confident_preds.py --dataset_pair sst2-yelp --zeroshot_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/sst2-yelp

# imdb-sst2
# python cal_confident_preds.py --dataset_pair imdb-sst2 --zeroshot_dir /scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output