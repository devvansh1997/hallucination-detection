import torch
from data import SyntheticRaw
from utils import get_g_pop, gram_factor_matrices, project_to_core

# create data
data = SyntheticRaw()

# create x_pop
x_pop = data.token_dim_reduce()

# get U_L, U_D
U_L, U_D = gram_factor_matrices(x_pop, data.R_L, data.R_D)

# get g_pop
G_population = get_g_pop(x_pop, U_L, U_D)

# get projected truths and lies
C_truth, C_lie = project_to_core(G_population, data.y_train)

# testing
x_test_raw = torch.randn(32, 11, 4096)  # raw mock
x_test_pool = x_test_raw.mean(dim=1)  # (32, 4096)
G_test = U_L.T @ x_test_pool @ U_D  # reuse TRAINING factor matrices

# compute features
sub_dC = torch.norm(G_test - C_truth)  # how far from truth centroid
sub_dH = torch.norm(G_test - C_lie)  # how far from lie centroid
sud_diff = sub_dC - sub_dH  # positive = hallucination | negative = truth
core_norm = torch.norm(G_test)  # activation magnitude in subspace

features = {
    "sub_dC": sub_dC.item(),
    "sub_dH": sub_dH.item(),
    "sud_diff": sud_diff.item(),
    "core_norm": core_norm.item(),
}

print(features)
