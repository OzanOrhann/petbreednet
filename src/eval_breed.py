import argparse, torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def get_loader(root, img=224, bs=64, workers=0):
    tte = transforms.Compose([
        transforms.Resize(int(img*1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(Path(root)/"test", tte)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return ds, ld

def build(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(m, ld, device):
    m.eval()
    correct = total = 0
    for x, y in ld:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = m(x).argmax(1)
        correct += (out == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)   # Windows için 0 güvenli
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, ld = get_loader(args.data_root, args.img_size, args.batch_size, args.workers)

    model = build(len(ds.classes)).to(device)
    # weights_only=True güvenli; ckpt sadece state_dict içeriyor
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])

    acc = evaluate(model, ld, device)
    print(f"Test accuracy: {acc:.4f}  ({int(acc*100)}%)")

if __name__ == "__main__":  # <-- Windows multiprocessing koruması
    main()
