# tools/oxford_to_imagefolder.py
# Oxford-IIIT Pets: images/*.jpg -> dst/<class_name>/*.jpg
import argparse, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data/raw/oxford_pets/images")
    ap.add_argument("--dst", required=True, help="data/raw/oxford_pets_imagefolder")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    assert src.exists(), f"Kaynak yok: {src}"
    dst.mkdir(parents=True, exist_ok=True)

    pics = list(src.glob("*.jpg"))
    print(f"[INFO] {len(pics)} görüntü bulundu.")
    moved = 0
    for p in pics:
        cls = p.stem.split("_")[0]   # 'Abyssinian_12.jpg' -> 'Abyssinian'
        outdir = dst / cls
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, outdir / p.name)  # istersen move için shutil.move
        moved += 1
        if moved % 500 == 0:
            print(f"[PROG] {moved} kopyalandı...")
    print(f"[DONE] Toplam {moved} dosya {dst} altındaki sınıf klasörlerine kopyalandı.")

if __name__ == "__main__":
    main()
