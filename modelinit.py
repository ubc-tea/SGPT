import torch
from models.vit_models_pr import VisionTransformer_m,CONFIGS
from utils import *

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):

        config = CONFIGS[args.model_type]
        if args.dataset == "cifar100":
            net = VisionTransformer_m(config, 224, zero_head=True, num_classes=100,args= args)
            net.load_from(np.load(args.pretrained_dir))
            net.freeze()
        if args.device == 'cuda':
            net = nn.DataParallel(net)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    # for (k, v) in nets[0].state_dict().items():
    #     model_meta_data.append(v.shape)
    #     layer_type.append(k)

    return nets, model_meta_data, layer_type