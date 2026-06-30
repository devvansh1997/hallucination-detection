import random

import torch
from torch.utils.data import Dataset


class RawActivations(Dataset):
    def __init__(self, checkpoint_pth: str) -> None:
        # load raw tensor
        ckpt = torch.load(checkpoint_pth)
        all_emb = ckpt["all_emb"]
        flags = ckpt["all_hallucination_flag"]

        # get N, L, D
        self.N = len(flags)
        # For L we need to sort first
        layer_keys = sorted(all_emb.keys(), key=lambda k: int(k.split(".")[-1]))
        self.L = len(layer_keys)
        self.D = all_emb[list(all_emb.keys())[0]][0].shape[-1]

        self.raw_activation_list = []
        for i in range(self.N):
            x_i = torch.stack([all_emb[key][i] for key in layer_keys], dim=0)
            # shape: (L, T_i, D)
            self.raw_activation_list.append(x_i)

        # Labels: True→1, False→0
        self.y_train = torch.tensor(flags, dtype=torch.long)

    def __len__(self):
        return len(self.raw_activation_list)

    def __getitem__(self, index):
        return self.raw_activation_list[index], self.y_train[index]

    def token_dim_reduce(self):
        x_pooled_list = []
        for i in range(len(self.raw_activation_list)):
            x_i = self.raw_activation_list[i]
            x_pooled_list.append(x_i.mean(dim=1))
        X_population = torch.stack(x_pooled_list)
        return X_population


class SyntheticRaw(Dataset):
    def __init__(self, L=32, D=4096, N=100, R_L=5, R_D=64):

        # ANOTOMICAL CONSTRAINTS
        self.L = L
        self.D = D
        self.N = N
        self.R_L = R_L
        self.R_D = R_D

        # reproducibility
        torch.manual_seed(42)
        random.seed(42)

        self.raw_activation_list = []
        self.T_min, self.T_max = 4, 16

        for i in range(self.N):
            # variable seq len
            T_i = random.randint(self.T_min, self.T_max)
            # create a tensor with shape: [L, T_i, D]
            x_i = torch.randn(self.L, T_i, self.D)
            self.raw_activation_list.append(x_i)

            # now entire tensor shape: [N, L, T_i, D]

        # now create binaray hallucination ground truth labels
        self.y_train = [0] * 50 + [1] * 50
        random.shuffle(self.y_train)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)

    def __len__(self):
        return len(self.raw_activation_list)

    def __getitem__(self, index):
        return self.raw_activation_list[index], self.y_train[index]

    def token_dim_reduce(self):
        x_pooled_list = []
        for i in range(len(self.raw_activation_list)):
            x_i = self.raw_activation_list[i]
            x_pooled_list.append(x_i.mean(dim=1))
        X_population = torch.stack(x_pooled_list)
        return X_population
