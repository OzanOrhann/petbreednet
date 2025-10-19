import argparse, shutil, random
from pathlib import Path

def copy_subset(files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in files: shutil.copy2(p, dst_dir / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data/raw/oxford_pets gibi; alt klasörler sınıf isimleri")
    ap.add_argument("--dst", default="data/processed")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src); assert src.exists(), f"Kaynak yok: {src}"
    random.seed(args.seed)

    classes = [d for d in src.iterdir() if d.is_dir()]
    for cls in classes:
        imgs = sorted([p for p in cls.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
        if not imgs: 
            continue
        random.shuffle(imgs)
        n = len(imgs); ntr=int(n*args.train); nva=int(n*args.val)
        tr, va, te = imgs[:ntr], imgs[ntr:ntr+nva], imgs[ntr+nva:]
        copy_subset(tr, Path(args.dst)/"train"/cls.name)
        copy_subset(va, Path(args.dst)/"val"/cls.name)
        copy_subset(te, Path(args.dst)/"test"/cls.name)
        print(f"[{cls.name}] total={n} train={len(tr)} val={len(va)} test={len(te)}")

if __name__=="__main__":
    main()
