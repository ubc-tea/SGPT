import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import copy
from collections import  defaultdict
from sklearn.metrics import confusion_matrix
from datasets import  CIFAR100_truncated
import torch.nn as nn
import random
from constants import *
import copy
from collections import OrderedDict, defaultdict
import torch.optim as optim
import torch, torch.nn as nn, torch.nn.functional as F
# from pypapi import events, papi_high as high

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

cls_coarse = np.array([
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

def fine_to_coarse(fine_label,args):
    label_np = fine_label.cpu().detach().numpy()
    label_change = np.zeros_like(label_np)
    for i in range(label_change.shape[0]):
        label_change[i]= coarse_labels[label_np[i]]
    return torch.Tensor(label_change)
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.

def compute_accuracy_our(global_model,data_loader_dict,args):
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        global_model.eval()
        if net_id not in data_loader_dict.keys():
            continue
        test_dl_local = data_loader_dict[net_id]['test_dl_local'] 
        # traindata_cls_count = data_loader_dict[net_id]['traindata_cls_count'] 
        test_correct, test_total, test_avg_loss = compute_accuracy_loss_our(global_model, test_dl_local, device=args.device,args = args)
        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total
        global_model.cluster_size = {i:0 for i in range(args.key_prompt)}
    
    #### global performance
    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    ### local performance
    local_mean_acc = np.mean([val['correct']/val['total'] for val in test_results.values()])
    local_min_acc = np.min([val['correct']/val['total'] for val in test_results.values()])

    return  test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc

def compute_accuracy_loss_our(model, dataloader, device="cpu",prototype = None,args=None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                output = model(x)
                out = output['logits']       
                _, pred_label = torch.max(out.data, 1)
                loss = criterion(out, target)
                correct += (pred_label == target.data).sum().item()
                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_simple_our(model, dataloader, get_confusion_matrix=False, args = None):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(args.device), target.to(args.device,dtype=torch.int64)
                output = model(x)
                out = output['logits']
                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                if args.device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def aggregation_func(keys_dict,global_para,selected,fed_avg_freqs,group_ratio,args):
    unique_dict = {}
    # unique_dict['prompt_keys'] = copy.deepcopy(global_para['prompt_keys'])
    for idx,r in enumerate(selected):
        net_para = keys_dict[r]
        ## momentum key ##
        if args.all_moment:
            if idx == 0:
                for key in net_para:
                    if  'prompt_embeddings' in key or 'prompt_keys' in key :
                        unique_dict[key] = copy.deepcopy(global_para[key])
                    if ('head' in key  or 'prompt_common' in key or 'running_mean' in key) and ('prompt_common_g' not in key) :
                        global_para[key] = copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
                    elif ( 'prompt_keys' in key ) and group_ratio is not None:
                        for ii, gs in enumerate(group_ratio[r].keys()):
                            global_para[key][ii:ii+1] = net_para[key][ii:ii+1]*group_ratio[r][gs]
                    elif ('prompt_embeddings' in key) and group_ratio is not None:
                        global_para[key] = net_para[key]*group_ratio[r][int(key.split('.')[-1])]
            else:
                for key in net_para:
                    if ('head' in key  or 'prompt_common' in key or 'running_mean' in key)  and ('prompt_common_g' not in key):
                        global_para[key] += copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
                    elif ( 'prompt_keys' in key ) and group_ratio is not None:
                        for ii, gs in enumerate(group_ratio[r].keys()):
                            global_para[key][ii:ii+1] += net_para[key][ii:ii+1]*group_ratio[r][gs]
                    elif ('prompt_embeddings' in key) and group_ratio is not None:
                        global_para[key] = net_para[key]*group_ratio[r][int(key.split('.')[-1])]
        else:
            if idx == 0:
                for key in net_para:
                    if  'prompt_embeddings' in key or 'prompt_keys' in key :
                        unique_dict[key] = copy.deepcopy(global_para[key])
                    if ('head' in key or 'prompt_embeddings' in key  or 'prompt_common' in key or 'running_mean' in key) and ('prompt_common_g' not in key) :
                        global_para[key] = copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
                    elif ( 'prompt_keys' in key ) and group_ratio is not None:
                        for ii, gs in enumerate(group_ratio[r].keys()):
                            global_para[key][ii:ii+1] = net_para[key][ii:ii+1]*group_ratio[r][gs]
            else:
                # or 'prompt_embeddings' in key 
                for key in net_para:
                    if ('head' in key  or 'prompt_embeddings' in key  or 'prompt_common' in key or 'running_mean' in key)  and ('prompt_common_g' not in key):
                        global_para[key] += copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
                    elif ( 'prompt_keys' in key ) and group_ratio is not None:
                        for ii, gs in enumerate(group_ratio[r].keys()):
                            global_para[key][ii:ii+1] += net_para[key][ii:ii+1]*group_ratio[r][gs]
    for key in unique_dict.keys():
        #### momentum 
        if  'prompt_embeddings' in key:
            global_para[key] = args.moment_p*unique_dict[key] + (1-args.moment_p)*global_para[key]
        if  'prompt_keys' in key :
            global_para[key] = args.moment*unique_dict[key] + (1-args.moment)*global_para[key]

    return global_para

def group_ratio_func(keys_dict,selected):
    group_ratio = {}
    group_total = {}
    for r in selected:
        cluster_r = keys_dict[r]['cluster_size']
        for key in cluster_r.keys():
            if key in group_total:
                group_total[key] += cluster_r[key]
            else:
                group_total[key] = cluster_r[key]

    for r in selected:
        group_ratio[r] = {}
        cluster_r = keys_dict[r]['cluster_size']
        for key,val in cluster_r.items():
            if group_total[key] <= 50:
                group_ratio[r][key] = 1/len(selected)
            else:
                group_ratio[r][key] = val/group_total[key]
    return group_ratio



def network_training_base(net,optimizer,args,train_dataloader,test_dataloader):
    criterion = nn.CrossEntropyLoss().to(args.device)
    cnt = 0
    for epoch in range(args.epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target,_) in enumerate(train_dataloader):
            x, target = x.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            target = target.long()
            output = net(x)   
            out = output['logits']
            loss = criterion(out, target)    
            epoch_loss_collector.append(loss.item())   
            loss.backward()
            optimizer.step()
            cnt += 1
            if batch_idx % 40 == 0:
                print('Training loss is {}'.format(sum(epoch_loss_collector) / len(epoch_loss_collector)))
                epoch_loss_collector = []
        # if (epoch+1) % 3 == 0:
        #     test_acc, conf_matrix = compute_accuracy_simple_our(net, test_dataloader, get_confusion_matrix=True,args = args)
        #     print('###### The Test ACC is {}'.format(test_acc))
    return net
def train_local_twostage(net,args,param_dict):
    train_dataloader = param_dict['train_dataloader']
    test_dataloader = param_dict['test_dataloader']
    dict_loss = param_dict['dict_loss']
    embedding_dict = param_dict['embedding_dict']
    round = param_dict['round']
    lr = args.lr
    net.train()
    net.selection = False
    optimizer = optim.SGD([p for k,p in net.named_parameters() if p.requires_grad  and  ('head' in k or 'common' in  k )], lr=lr, momentum=args.rho, weight_decay=args.reg)
    net = network_training_base(net,optimizer,args,train_dataloader,test_dataloader)
    net.selection = True
    net.train()
    optimizer = optim.SGD([p for k,p in net.named_parameters() if p.requires_grad  and 'prompt_keys_pr' not in k and 'prompt_common_g' not in k  and 'common' not in k ], lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(args.device)
    cnt = 0
    for epoch in range(args.epochs):
        epoch_loss_collector = []
        epoch_loss_collector2 = []
        for batch_idx, (x, target,index) in enumerate(train_dataloader):
            x, target = x.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            target = target.long()
            output = net(x,index,embedding_dict)
            out = output['logits']
            if out.shape[0] != target.shape[0]:
                target = torch.cat([target]*net.topk,dim=0)
            loss = criterion(out, target)
            if args.avg_key and epoch <= args.keyepoch:
                reduced_sim = output['reduced_sim']
                loss += reduced_sim
                epoch_loss_collector2.append(reduced_sim.item())       
            # #          
            epoch_loss_collector.append(loss.item())
            loss.backward()
            optimizer.step()
            cnt += 1
            if batch_idx % 40 == 0:
                if args.avg_key and epoch <= args.keyepoch:
                    print('Key loss is {}'.format(sum(epoch_loss_collector2) / len(epoch_loss_collector2)))
                print('Training loss is {}'.format(sum(epoch_loss_collector) / len(epoch_loss_collector)))
                dict_loss["train/step"] = cnt
                dict_loss["train/train_loss"] = sum(epoch_loss_collector) / len(epoch_loss_collector)
                epoch_loss_collector = []
        # if (epoch+1) % 3 == 0:
        ### do validation if needed
        #     test_acc, conf_matrix = compute_accuracy_simple_our(net, test_dataloader, get_confusion_matrix=True,args = args)
        #     print('###### The Test ACC is {}'.format(test_acc))
        #     dict_loss["val/step"] = epoch
        #     dict_loss["val/test_acc_epoch"] = test_acc    
    return output['embedding_dict']


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
def save_promptparams(nets):
    nets_list = {}
    for client in nets.keys():
        nets_list[client] = {k: v for k, v in nets[client].state_dict().items() if 'prompt' in k}
    return nets_list
def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights

#     return protos
def fine_to_coarse(fine_label):
    label_np = fine_label.cpu().detach().numpy()
    label_change = np.zeros_like(label_np)
    coarse_labels = \
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
    for i in range(label_change.shape[0]):
        label_change[i]= coarse_labels[label_np[i]]
    return label_change

def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None,args= None):

    if dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    if partition == "noniid-labeluni":
        if "cifar100" in dataset:
            num = args.cls_num
        if "cifar100" in dataset:
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu",args = None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                if args is not None and args.uppper_coarse:
                    output = model(x,prompt_index = fine_to_coarse(target))
                else:
                    output = model(x)
                out = output['logits']
                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def compute_accuracy_loss(model, dataloader, device="cpu",args=None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                if args is not None  and args.uppper_coarse:
                    output = model(x,prompt_index = fine_to_coarse(target))
                    out = output['logits']
                else:
                    output = model(x)
                    out = output['logits']
                _, pred_label = torch.max(out.data, 1)
                loss = criterion(out, target)
                correct += (pred_label == target.data).sum().item()
              
                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count

def compute_accuracy_local(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = 0
        _, test_dl_local, _, _ = get_divided_dataloader(args, dataidxs_train, dataidxs_test, noise_level)
        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):

        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 255)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(args, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0, apply_noise=False):
    
    dataset = args.dataset
    datadir = args.datadir
    train_bs = args.batch_size

    if dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        transform_train = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_divided_dataloader(args, dataidxs_train, dataidxs_test,noise_level=0, drop_last=False, apply_noise=False,traindata_cls_counts=None):
    dataset = args.dataset
    datadir = args.datadir
    train_bs = args.batch_size
    test_bs = 4*args.batch_size

    if 'cifar100' in dataset:
        dl_obj = CIFAR100_truncated
        if apply_noise:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                GaussianNoise(0., noise_level)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                GaussianNoise(0., noise_level)
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    traindata_cls_counts = traindata_cls_counts
    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False,return_index=True)
    test_ds = dl_obj(datadir, dataidxs= dataidxs_test ,train=False, transform=transform_test, download=False,return_index=False)

    conditioned_loader = {}
    if traindata_cls_counts is not None:
        for cls_id in traindata_cls_counts.keys():
            cnd_ds =  dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False,cls_condition= cls_id)
            train_dl = data.DataLoader(dataset=cnd_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
            conditioned_loader[cls_id]  = train_dl


    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds,conditioned_loader,traindata_cls_counts