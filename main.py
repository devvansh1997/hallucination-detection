from utils import evaluate_classifiers

R_L = 5
R_D = 64
CKPT = "../res2_train_32_layers_tensor.pt"

print("=" * 58)
print("  NO-LEAKAGE SUPERVISED CLASSIFICATION")
print("  (U_L, U_D computed from training set only)")
print("=" * 58)

results = evaluate_classifiers(CKPT, R_L, R_D, test_size=0.2, random_state=42)
