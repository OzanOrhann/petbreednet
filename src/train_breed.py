import argparse, time, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def loaders(root, img=224, bs=32, w=4):
    ttr = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    tte = transforms.Compose([
        transforms.Resize(int(img*1.15)), transforms.CenterCrop(img),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    tr = datasets.ImageFolder(Path(root) / "train", ttr)
    va = datasets.ImageFolder(Path(root) / "val",   tte)

    train_ld = DataLoader(
        tr, batch_size=bs, shuffle=True,
        num_workers=w, pin_memory=True, persistent_workers=(w > 0)
    )
    val_ld = DataLoader(
        va, batch_size=bs, shuffle=False,
        num_workers=w, pin_memory=True, persistent_workers=(w > 0)
    )
    return train_ld, val_ld, tr.classes


def build(nc, finetune=True):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if finetune:
        for p in m.parameters(): p.requires_grad=False
        for p in m.layer4.parameters(): p.requires_grad=True
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m

def acc(o,y): return (o.argmax(1)==y).float().mean().item()

@torch.no_grad()
def evaluate(m,ld,dev,crit):
    m.eval(); L=A=N=0
    for x,y in ld:
        x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
        o=m(x); l=crit(o,y); L+=l.item()*x.size(0); A+=acc(o,y)*x.size(0); N+=x.size(0)
    return L/N, A/N

def train_one(m, ld, opt, sc, dev, crit):
    m.train(); L = A = N = 0
    for x, y in ld:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        # Güncel PyTorch API ile autocast + GradScaler kullanımı
        with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            o = m(x)
            l = crit(o, y)
        sc.scale(l).backward()      
        sc.step(opt)                
        sc.update()                 
        L += l.item() * x.size(0)
        A += (o.argmax(1) == y).float().sum().item()
        N += x.size(0)
    return L / N, A / N


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",default="data/processed")
    ap.add_argument("--img_size",type=int,default=224)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--out",default="checkpoints/resnet50_breed.pth")
    a=ap.parse_args()

    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr,va,classes = loaders(a.data_root,a.img_size,a.batch_size,a.workers)
    m=build(len(classes)).to(dev)
    crit=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    sc = torch.amp.GradScaler('cuda', enabled=(dev=="cuda"))


    best=0.0; Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    for e in range(1,a.epochs+1):
        t=time.time()
        tl,ta=train_one(m,tr,opt,sc,dev,crit)
        vl,vaacc=evaluate(m,va,dev,crit)
        print(f"[{e:02d}] train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{vaacc:.3f}  ({time.time()-t:.1f}s)")
        if vaacc>best:
            best=vaacc; torch.save({"model":m.state_dict(),"classes":classes}, a.out)
            print(f"  ↳ saved best to {a.out} (acc={best:.3f})")

if __name__=="__main__": main()
