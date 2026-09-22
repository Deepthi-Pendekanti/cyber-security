import numpy as np, torch
from amln.config import AMLNConfig
from amln.data import ATTACK_FAMILIES, CyberEventGenerator
from amln.data.preprocess import StandardScaler
from amln.models.encoder import BehaviorAutoencoder
from amln.memory.landscape import AdaptiveMemoryLandscape
from amln.training import pretrain_autoencoder, populate_landscape

cfg = AMLNConfig()
torch.manual_seed(42); np.random.seed(42)
zero=[5,7]; known=[i for i in range(len(ATTACK_FAMILIES)) if i not in zero]
gen=CyberEventGenerator(cfg.encoder.num_features, seed=42)
tr=gen.sample(12000, families=known)
tek=gen.sample(1500, families=known)
tez=gen.sample(1500, families=zero)
sc=StandardScaler().fit(tr.features)
xtr=sc.transform(tr.features); xk=sc.transform(tek.features); xz=sc.transform(tez.features)
m=BehaviorAutoencoder(cfg.encoder); pretrain_autoencoder(m,xtr,cfg)
ls=AdaptiveMemoryLandscape(cfg.encoder.latent_dim,cfg.landscape)
populate_landscape(m,ls,xtr,tr.labels,tr.severity,cfg)

def nearest_hill_sim(feats):
    emb=m.embed(torch.from_numpy(feats)).numpy()
    out=[]; labs=[]
    for e in emb:
        n,s=ls.retrieve(e,hills_only=True); out.append(s); labs.append(n.label)
    return np.array(out), np.array(labs)

sk,lk = nearest_hill_sim(xk)
sz,lz = nearest_hill_sim(xz)
print("KNOWN  nearest-hill sim: mean=%.3f  p05=%.3f  min=%.3f"%(sk.mean(),np.percentile(sk,5),sk.min()))
print("ZERO   nearest-hill sim: mean=%.3f  p95=%.3f  max=%.3f"%(sz.mean(),np.percentile(sz,95),sz.max()))
benign=ATTACK_FAMILIES.index("benign")
print("ZERO matched-hill label dist:", {ATTACK_FAMILIES[l]:int((lz==l).sum()) for l in set(lz)})
# what threshold separates?
for th in [0.99,0.98,0.97,0.95,0.9]:
    fk=(sk<th).mean(); fz=(sz<th).mean()
    print(f"th={th}: known flagged(novel)={fk:.3f}  zero flagged(novel)={fz:.3f}")
