import argparse, shutil, random
from pathlib import Path

def copy_subset(files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in files: shutil.copy2(p, dst_dir / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageFolder kök klasörü (alt klasörler sınıf adları)")
    ap.add_argument("--dst", default="data/processed")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--ensure_min_test", type=int, default=1,
                    help="Her sınıf için test split’inde minimum görüntü sayısı (>=1). n==1 ise test’e zorlanmaz.")
    args = ap.parse_args()

    src = Path(args.src)
    assert src.exists(), f"Kaynak yok: {src}"

    random.seed(args.seed)

    classes = [d for d in src.iterdir() if d.is_dir()]
    for cls in classes:
        imgs = sorted([p for p in cls.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])
        if not imgs:
            print(f"[WARN] sınıf boş atlandı: {cls.name}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        ntr = int(n * args.train)
        nva = int(n * args.val)
        # kalan otomatik test
        tr = imgs[:ntr]
        va = imgs[ntr:ntr+nva]
        te = imgs[ntr+nva:]

        # n==1 ise test/val'e dağıtmak yerine train'de bırak (eğitimi bozmasın)
        if n == 1:
            te = []
            va = []
            tr = imgs

        # Test en az ensure_min_test olsun (args.test>0 ise)
        if args.test > 0 and n > 1:
            need = max(0, args.ensure_min_test - len(te))
            while need > 0:
                if va:
                    te.append(va.pop())      
                elif tr:
                    te.append(tr.pop())      
                else:
                    break
                need -= 1

        
        copy_subset(tr, Path(args.dst)/"train"/cls.name)
        copy_subset(va, Path(args.dst)/"val"/cls.name)
        copy_subset(te, Path(args.dst)/"test"/cls.name)

        print(f"[{cls.name}] total={n}  train={len(tr)}  val={len(va)}  test={len(te)}")

if __name__ == "__main__":
    main()
