import pandas as pd, torchvision
from .utils import OUT_DIR, set_seed, nowtag
from .models import build_resnet18
from .data import get_cifar10
from .robust_eval import eval_with_corruptions
from .train import train_model

def run_experiment(dataset="cifar10", epochs=5, batch_size=128):
    train_loader, test_loader = get_cifar10(batch_size=batch_size)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    rows = []
    set_seed(42); model_base = build_resnet18(num_classes=10, use_drags=False)
    t_base = train_model(model_base, train_loader, test_loader, epochs=epochs)
    base_res = eval_with_corruptions(model_base, test_set); base_res.update({"model":"Baseline","train_time_sec":t_base}); rows.append(base_res)
    set_seed(42); model_drags = build_resnet18(num_classes=10, use_drags=True, drags_layers=("layer1","layer2"))
    t_drags = train_model(model_drags, train_loader, test_loader, epochs=epochs)
    drags_res = eval_with_corruptions(model_drags, test_set); drags_res.update({"model":"DRAGS","train_time_sec":t_drags}); rows.append(drags_res)
    df = pd.DataFrame(rows); out = OUT_DIR/"csv"/f"drags_results_{nowtag()}.csv"; df.to_csv(out, index=False); print(f"Results saved to {out}"); return df

def run_multi_seed_experiment(seeds=[42,43,44], epochs=5, batch_size=128):
    train_loader, test_loader = get_cifar10(batch_size=batch_size); test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    results=[]
    for seed in seeds:
        print(f"\n--- Running seed {seed} ---"); set_seed(seed)
        m_base = build_resnet18(num_classes=10, use_drags=False); t_base = train_model(m_base, train_loader, test_loader, epochs=epochs)
        r = eval_with_corruptions(m_base, test_set); r.update({"model":"Baseline","train_time_sec":t_base,"seed":seed}); results.append(r)
        m_d = build_resnet18(num_classes=10, use_drags=True, drags_layers=("layer1","layer2")); t_d = train_model(m_d, train_loader, test_loader, epochs=epochs)
        r = eval_with_corruptions(m_d, test_set); r.update({"model":"DRAGS","train_time_sec":t_d,"seed":seed}); results.append(r)
    df_all = pd.DataFrame(results); csv = OUT_DIR/"csv"/f"multi_seed_raw_{nowtag()}.csv"; df_all.to_csv(csv, index=False); print(f"Saved raw multi-seed results to {csv}"); return df_all

def aggregate_results(df_raw):
    corr = ["clean","gaussian","motion_blur","occlusion"]; rows=[]
    for model in ["Baseline","DRAGS"]:
        sub = df_raw[df_raw["model"]==model]; row={"model":model}
        for c in corr:
            row[c] = f"{sub[c].mean():.2f} ± {sub[c].std():.2f}"
        row["train_time_sec"] = f"{sub['train_time_sec'].mean():.2f} ± {sub['train_time_sec'].std():.2f}"; rows.append(row)
    import pandas as pd; df = pd.DataFrame(rows); csv = OUT_DIR/"csv"/f"multi_seed_summary_{nowtag()}.csv"; df.to_csv(csv, index=False); print(f"Saved aggregated results table → {csv}"); return df

def run_ablation_study(topk_list=[0.05,0.10,0.15], layers_list=[("layer1",),("layer1","layer2")], seeds=[42,43,44], epochs=2):
    import pandas as pd
    train_loader, test_loader = get_cifar10(); test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    results=[]
    for topk in topk_list:
        for layers in layers_list:
            for seed in seeds:
                set_seed(seed); m = build_resnet18(num_classes=10, use_drags=True, drags_layers=layers, pool_ks=3, topk_ratio=topk)
                from .train import train_model as tm; t = tm(m, train_loader, test_loader, epochs=epochs)
                res = eval_with_corruptions(m, test_set); res.update({"topk_ratio":topk, "layers_wrapped":"+".join(layers), "train_time_sec":t, "seed":seed}); results.append(res)
    df = pd.DataFrame(results); csv = OUT_DIR/"csv"/f"drags_ablation_{nowtag()}.csv"; df.to_csv(csv, index=False); print(f"Saved ablation study results → {csv}"); return df
