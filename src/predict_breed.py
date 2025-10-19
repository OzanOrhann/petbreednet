import argparse, torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def build(nc):
    m=models.resnet50(); m.fc=nn.Linear(m.fc.in_features,nc); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="checkpoints/resnet50_breed.pth")
    ap.add_argument("--img",required=True)
    ap.add_argument("--img_size",type=int,default=224)
    a=ap.parse_args()

    ck=torch.load(a.ckpt,map_location="cpu")
    classes=ck["classes"]; m=build(len(classes)); m.load_state_dict(ck["model"]); m.eval()

    tf=transforms.Compose([transforms.Resize(int(a.img_size*1.15)),
                           transforms.CenterCrop(a.img_size),
                           transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    x=tf(Image.open(a.img).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p=torch.softmax(m(x),1)[0]; i=int(p.argmax())
    print(f"Pred: {classes[i]}  (p={float(p[i]):.3f})")

if __name__=="__main__": main()
