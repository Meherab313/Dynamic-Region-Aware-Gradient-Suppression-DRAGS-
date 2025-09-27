import torch, torch.nn as nn, torch.nn.functional as F
class DRAGSLayerWrapper(nn.Module):
    def __init__(self, module, pool_ks=3, topk_ratio=0.1):
        super().__init__(); self.module=module; self.pool_ks=pool_ks; self.topk_ratio=float(topk_ratio)
    def forward(self, x):
        out = self.module(x)
        if self.training and out.dim()==4:
            pooled = F.avg_pool2d(out, self.pool_ks, stride=1, padding=self.pool_ks//2)
            score = (out - pooled).pow(2)
            B,C,H,W = score.shape; k = max(1, int(H*W*self.topk_ratio))
            mask = torch.zeros_like(score, dtype=torch.bool)
            for b in range(B):
                vals = score[b].mean(0).flatten()
                top_idx = vals.topk(k).indices
                mask_2d = torch.zeros(H*W, dtype=torch.bool, device=out.device)
                mask_2d[top_idx] = True
                mask[b] = mask_2d.view(H,W).unsqueeze(0).expand(C,-1,-1)
            out = out.clone(); out[mask] = out[mask].detach()
        return out
