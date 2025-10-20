import argparse, torch
from pathlib import Path
from torchvision import models

def load_model(ckpt_path:str):
    ck = torch.load(ckpt_path, map_location="cpu")
    m = models.resnet50()
    m.fc = torch.nn.Linear(m.fc.in_features, len(ck["classes"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["classes"]

def export_torchscript(m, out_path):
    ex = torch.randn(1,3,224,224)
    ts = torch.jit.trace(m, ex)
    ts.save(out_path)

def export_onnx(m, out_path):
    ex = torch.randn(1,3,224,224)
    torch.onnx.export(
        m, ex, out_path,
        input_names=["input"], output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoints/resnet50_breed.pth")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--no_ts", action="store_true", help="TorchScript üretme")
    ap.add_argument("--no_onnx", action="store_true", help="ONNX üretme")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    m, classes = load_model(args.ckpt)

    if not args.no_ts:
        ts_path = str(Path(args.out_dir)/"resnet50_breed.ts")
        export_torchscript(m, ts_path)
        print(f"✔ TorchScript: {ts_path}")

    if not args.no_onnx:
        onnx_path = str(Path(args.out_dir)/"resnet50_breed.onnx")
        export_onnx(m, onnx_path)
        print(f"✔ ONNX:       {onnx_path}")

    (Path(args.out_dir)/"classes.txt").write_text("\n".join(classes), encoding="utf-8")
    print("✔ classes.txt yazıldı")
