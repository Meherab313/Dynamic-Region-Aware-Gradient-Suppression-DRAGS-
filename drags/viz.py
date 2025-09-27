import numpy as np, matplotlib.pyplot as plt, torch.nn.functional as F, torchvision.transforms as T
from .layers import DRAGSLayerWrapper
from .utils import device, OUT_DIR
def drags_heatmap_pdf(model, dataset, num_images=5, save_path=None):
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    images, labels = zip(*[(transform(dataset[i][0]), dataset[i][1]) for i in range(num_images)])
    fig_width=7.16; fig_height=fig_width*0.35; fig, axes = plt.subplots(1, num_images, figsize=(fig_width, fig_height))
    if num_images==1: axes=[axes]
    for i,(img,label) in enumerate(zip(images,labels)):
        x=img.unsqueeze(0).to(device); masks=[]
        def hook_fn(module, input, output):
            mask=(output!=output.detach()).float()
            mask_resized=F.interpolate(mask, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
            masks.append(mask_resized.squeeze(0).mean(0).cpu().numpy())
        hooks=[]; 
        for _,m in model.named_modules():
            if isinstance(m, DRAGSLayerWrapper): hooks.append(m.register_forward_hook(hook_fn))
        _=model(x)
        for h in hooks: h.remove()
        import numpy as np
        heatmap=np.mean(np.stack(masks),axis=0); heatmap=(heatmap-heatmap.min())/(heatmap.max()-heatmap.min()+1e-6)
        img_np=img.permute(1,2,0).cpu().numpy(); axes[i].imshow(img_np); im=axes[i].imshow(heatmap, cmap='jet', alpha=0.55, vmin=0, vmax=1); axes[i].axis("off")
        axes[i].text(0.5, -0.08, f"Label: {label}", fontsize=8, ha='center', va='top', transform=axes[i].transAxes, fontweight='bold', color='black')
    fig.subplots_adjust(right=0.88); cbar_ax=fig.add_axes([0.90,0.25,0.015,0.5]); cbar=fig.colorbar(im, cax=cbar_ax); cbar.set_label('Suppression Intensity', fontsize=8); cbar.ax.tick_params(labelsize=8)
    import pathlib; 
    import matplotlib
    plt.tight_layout(rect=[0,0,0.88,1])
    if save_path is None: save_path = OUT_DIR / "figures" / "drags_heatmap_ieee.pdf"
    plt.savefig(save_path, dpi=900, bbox_inches='tight'); print(f"Saved IEEE two-column DRAGS heatmap PDF → {save_path}"); plt.close(fig)
