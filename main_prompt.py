
import numpy as np
import json
import torch
from collections import  defaultdict
from pathlib import Path
import os
import copy
from math import *
import random
from modelinit import *
import numpy as np
from office_dataset import prepare_data
from utils import *
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='ViT-B_16', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='SGPT',  help='strategy')
    parser.add_argument('--comm_round', type=int, default=60, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--test_round', type=int, default=50)
    parser.add_argument('--keyepoch', type=int, default=5, help='number of epoch to update key')
    parser.add_argument('--moment', type=float, default=0.5, help='momentum key')
    parser.add_argument('--moment_p', type=float, default=0.5, help='momentum prompt')
    parser.add_argument('--domain_query', action='store_true')   
    parser.add_argument('--leaky', action='store_true') 
    parser.add_argument('--all_moment', action='store_true') 
    parser.add_argument('--root_path', type=str, default='', help='Noise type: None/increasng/space')
    """
    Used for model 
    """
    parser.add_argument('--model_type', type=str, default='ViT-B_16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained_dir', type=str, default=None, help='The pretrain model path')
    parser.add_argument('--n_prompt', type=int, default=1, help='The length of shared prompt')
    parser.add_argument("--key_prompt", type=int, default=0, help='cluster numbers')   
    parser.add_argument('--avg_key', action='store_true')   
    parser.add_argument('--cls_num', type=int, default=10) 
    parser.add_argument('--initial_g', action='store_true',help='xavier initial group prompts elase 0') 
    parser.add_argument('--share_blocks', nargs='+', type=int, default=[], help="shared transformer set 6 ")
    parser.add_argument('--share_blocks_g', nargs='+', type=int,  default=[], help="shared transformer set 6 ")
    
    args = parser.parse_args()
    return args
args = get_args()
if args.dataset == 'CIFAR-100':
    cls_coarse = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
# args=Args()
save_path = args.model_type+"-"+str(args.n_parties)+"-"+args.dataset+'-'+str(args.cls_num)+"-"+args.partition+'-'+'avg_key_'+str(args.avg_key)+'-'+args.alg +'p_num_' +str(args.key_prompt) + '_momentk_' + str(args.moment) +'_momentp_' + str(args.moment_p) + 'keyepoch_'+str(args.keyepoch)
root_path = args.logdir

save_path = Path(os.path.join(root_path,save_path))
save_path.mkdir(parents=True, exist_ok=True)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
print(save_path)

with open(os.path.join(save_path,'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.dataset not in ['office']:
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
                args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir,args=args)

arr = np.arange(args.n_parties)

if args.dataset == 'office':
    data_loader_dict,net_dataidx_map_train = prepare_data(args)
    num_classes = 10
else:
    ###### Data Set related ###### 
    data_loader_dict = {}
    for net_id in arr:
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        data_loader_dict[net_id] = {}
        train_dl_local, test_dl_local, _, _ ,_,_ = get_divided_dataloader(args, dataidxs_train, dataidxs_test,traindata_cls_counts=traindata_cls_counts[net_id])
        num_classes = 100
        data_loader_dict[net_id]['train_dl_local'] = train_dl_local
        data_loader_dict[net_id]['test_dl_local'] = test_dl_local
            
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = args.device
###### Model related ###### 
config = CONFIGS[args.model_type]
net = VisionTransformer_m(config, args.img_size, num_classes=num_classes,vis = True,args= args)
net.load_from(np.load(args.pretrained_dir))
net.freeze()
net.to(device)
global_para = {k: copy.deepcopy(v) for k, v in net.state_dict().items() if 'head' in k or 'prompt' in k }
keys_dict = {}
cluster_dict = {}
cluster_dict_all = {}
for net_id in arr:
    keys_dict[net_id] = keys_dict[net_id] = {k:copy.deepcopy(v) for k, v in global_para.items()}
    cluster_dict[net_id] = {}
    cluster_dict_all[net_id] = {i:0 for i in range(args.key_prompt)}
global_select =  {i:0 for i in range(args.key_prompt)}
embedding_dicts= {i:{} for i in range(args.n_parties) } 
dict_loss = {}
results_dict = defaultdict(list)
#### clustering for classifiers
groups_list = []
pr_label_pr = {}
if args.avg_key:
    net.selection = True
for round in range(args.comm_round):
    # if round >= 30:
    #     net.args.domain_query = True
    print('########### Now is the round {} ######'.format(round))
    arr = np.arange(args.n_parties)
    np.random.shuffle(arr)
    selected = arr[:int(args.n_parties * args.sample)]
    for ix in range(len(selected)):
        param_dict = {}
        idx = selected[ix]
        print('Now is the client {}'.format(idx))
        #### copy with global parameters
        keys_dict[idx] = {k:copy.deepcopy(v) for k, v in global_para.items()}
        net.load_state_dict(keys_dict[idx],strict = False)
        net.cluster_size = {i:0 for i in range(args.key_prompt)}
        net.cluster_size_g = global_select
        net.cluster_size_l = cluster_dict_all[idx]
        avg_acc = 0.0
        n_epoch = args.epochs
        train_dl_local = data_loader_dict[idx]['train_dl_local']
        test_dl_local = data_loader_dict[idx]['test_dl_local'] 
        param_dict['train_dataloader'] = train_dl_local
        param_dict['test_dataloader'] = test_dl_local
        param_dict['dict_loss'] = dict_loss
        param_dict['round'] = round
        param_dict['group_label'] = pr_label_pr
        param_dict['embedding_dict'] = embedding_dicts[idx]
        embedding_dict = train_local_twostage(net,args,param_dict)
        embedding_dicts[idx] = copy.deepcopy(embedding_dict)
        for k, v in net.state_dict().items():
            if 'head' in k or 'prompt' in k:
                keys_dict[idx][k] =  copy.deepcopy(v)   
        cluster_dict[idx]['cluster_size'] = copy.deepcopy(net.cluster_size)   
        for key in cluster_dict[idx]['cluster_size'].keys():
            global_select[key] += cluster_dict[idx]['cluster_size'][key]
            cluster_dict_all[idx][key] = cluster_dict[idx]['cluster_size'][key]
        print(cluster_dict[idx]['cluster_size'])
    print(global_select)
    total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
    fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]
    if net.selection:
        group_ratio = group_ratio_func(cluster_dict,selected)
    else:
        group_ratio = None
    global_para = aggregation_func(keys_dict,global_para,selected,fed_avg_freqs,group_ratio,args)
    net.load_state_dict(global_para,strict = False)
    if (round+1)>=args.test_round:
        test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_our(net,data_loader_dict,args)
        print('>> Mean Local Test accuracy: %f' % local_mean_acc)
        print('>> Min Local Test accuracy: %f' % local_min_acc)
        print('>> Global Model Test accuracy: %f' % test_avg_acc)
        print('>> Test avg loss: %f' %test_avg_loss)

        results_dict['test_avg_loss'].append(test_avg_loss)
        results_dict['test_avg_acc'].append(test_avg_acc)
        results_dict['local_mean_acc'].append(local_mean_acc)
        results_dict['local_min_acc'].append(local_min_acc)

        outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
        torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)

json_file_opt = "results.json"
with open(os.path.join(save_path,json_file_opt), "w") as file:
    json.dump(results_dict, file, indent=4)