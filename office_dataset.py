
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None,retrun_index =False):
        if train:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'
        self.retrun_index = retrun_index
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.retrun_index:
            return image, label,idx
        else:
            return image, label


   
def prepare_data(args):
    net_dataidx_map_train = {}
    data_loader_dict = {}
    data_base_path = 'data'
    transform_office = transforms.Compose([
            transforms.Resize([224, 224]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),            
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office,retrun_index=True)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office,retrun_index=True)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office,retrun_index=True)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office,retrun_index=True)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch_size, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch_size, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch_size, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch_size, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch_size, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch_size, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch_size, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch_size, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch_size, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch_size, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch_size, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch_size, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    for i in range(len(train_loaders)):
        data_loader_dict[i] = {'train_dl_local': train_loaders[i], 'val_dl_local' :val_loaders[i],'test_dl_local' : test_loaders[i]}
        net_dataidx_map_train[i] = list(range(len(train_loaders[i].dataset.dataset)))
    return data_loader_dict, net_dataidx_map_train
