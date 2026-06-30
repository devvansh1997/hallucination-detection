"""Quick probe of attention tensor files."""
import torch

for fname in ["res2_attn_train_pad.pt", "res2_attn_val_pad.pt"]:
    path = f"C:/Users/devan/Hallucination-Detection/{fname}"
    d = torch.load(path, weights_only=False)
    emb = d["all_emb"]
    flags = d["all_hallucination_flag"]
    keys = sorted(emb.keys(), key=lambda k: int(k.split(".")[-1]))
    k0 = keys[0]
    v0 = emb[k0]

    print(f"=== {fname} ===")
    print(f"  layers: {len(keys)}  ({keys[0]} .. {keys[-1]})")
    print(f"  samples: {len(flags)}")
    print(f"  all hallucination: {all(flags) if isinstance(flags, list) else flags.all().item()}")
    print(f"  layer[0] type: {type(v0).__name__},  len: {len(v0)}")
    print(f"  entry[0] type: {type(v0[0]).__name__},  shape: {v0[0].shape}")
    if len(v0) > 1:
        print(f"  entry[1] shape: {v0[1].shape}")
        print(f"  entry[2] shape: {v0[2].shape}")
    # Check if shapes are uniform across samples
    shapes = set()
    for i in range(min(20, len(v0))):
        shapes.add(v0[i].shape)
    print(f"  unique shapes in first 20 samples: {shapes}")
    print()
