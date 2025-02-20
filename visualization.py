import numpy as np
import torch
from spikingjelly.activation_based import encoding,functional
from model_spiking import VisionTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from data_loader import get_loader
from spikingjelly import visualizing
import torchvision




def visualize(args, spikes_per_layer,folder):
    if args.out_dir is not None and args.out_dir != '':
        vs_dir = os.path.join(args.out_dir, 'visualization',folder)
        if not os.path.exists(vs_dir):
            os.mkdir(vs_dir)
        for (layer, tensor_list) in spikes_per_layer[:-1]:
            print(f'saving sample with L={layer}...')
            spike_seq = torch.cat(tensor_list)
            print(spike_seq.shape)
            visualizing.plot_2d_feature_map(spike_seq,  nrows=args.T, ncols=spike_seq.shape[0]//args.T, space=2, title='Spiking Feature Maps', dpi=200)
            plt.savefig(os.path.join(vs_dir, f's_{layer}.png'), pad_inches=0.02)
            plt.clf()
