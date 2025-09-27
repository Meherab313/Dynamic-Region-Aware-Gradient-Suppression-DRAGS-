import random, numpy as np, torch
from datetime import datetime
from pathlib import Path
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def nowtag(): return datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("drags_outputs"); (OUT_DIR/"figures").mkdir(parents=True, exist_ok=True); (OUT_DIR/"csv").mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
