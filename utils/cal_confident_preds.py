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
                print(len(ind_lst_largest_prob))
                print(ind_lst_largest_prob)

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

                    output_label_name = 'pseudolabels.npy'
                    with open(args.zeroshot_dir+'/'+output_label_name, 'wb') as f_label:
                        np.save(f_label, pseudo_labels)

                    # with open(args.zeroshot_dir+'/'+args.output_file_name, 'wb') as f_ind:
                    #     np.save(f_ind, ind_lst_largest_prob)

                break
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pair",type=str,default="imdb-yelp")
    parser.add_argument("--zeroshot_dir", type=str, 
                        default='/scratch/yk2516/UDA_Text_Generation/benchmark/target_zeroshot_output/imdb-yelp')
    parser.add_argument("--output_file_name",type=str,default='conf_indices.npy')
    args = parser.parse_args()
    main(args)

def change_label(examples):
    examples['label']=new_label