PETBREEDNET - DOG BREED CLASSIFICATION PROJECT
===============================================

📘 PROJECT OVERVIEW
-------------------
PetBreedNet is a deep-learning-based image classification system that identifies dog (and cat) breeds using the ResNet-50 model. 
The project trains, evaluates and exports a CNN model in multiple formats (PyTorch .pth, TorchScript .ts, ONNX .onnx) for efficient deployment and inference.

📁 PROJECT STRUCTURE
--------------------
petbreednet/
│
├── src/
│   ├── train_breed.py
│   ├── eval_breed.py
│   ├── predict_breed.py
│
├── tools/
│   ├── export_model.py
│   ├── split_imagefolder.py
│   ├── oxford_to_imagefolder.py
│
├── checkpoints/  (model files will be placed here)
│
├── requirements.txt
└── README.txt  (or README.md)

📦 MODEL FILES (DOWNLOAD LINKS)
-------------------------------
All trained model files are hosted on Google Drive. 
Download them and place inside the "checkpoints" folder.

resnet50_breed.pth → https://drive.google.com/file/d/1eeYJqUx2MgDssZMkABemXY5AOBoRwhiW/view?usp=drive_link  
resnet50_breed.ts → https://drive.google.com/file/d/1TkJ_EyzBUR2Tf--VbsUO7Zwh4iBSlKDj/view?usp=drive_link  
resnet50_breed.onnx → https://drive.google.com/file/d/1ncqa4_VNxxEyQf-zIXgiBJSZ8wG_Orsj/view?usp=drive_link  
classes.txt → https://drive.google.com/file/d/19VONtIY506Py3eaBzV3PtoKtaWNFb7n2/view?usp=drive_link  

⚠️ After downloading, ensure all files are placed in:
petbreednet/checkpoints/

📚 DATASET
----------
The project uses the Oxford-IIIT Pet Dataset (37 classes).  
Official download page: https://www.robots.ox.ac.uk/~vgg/data/pets/  

Expected folder structure after extraction:
dataset/
├── images/                   (or train/val folders after processing)
│   ├── Abyssinian/
│   ├── American_Bulldog/
│   └── … (other breed folders)
└── annotations/

⚙️ SETUP INSTRUCTIONS
---------------------
1. Clone the repository:
   git clone https://github.com/OzanOrhann/petbreednet.git
   cd petbreednet

2. Create and activate conda environment:
   conda create -n petbreed python=3.11
   conda activate petbreed

3. Install required packages:
   pip install -r requirements.txt

🧠 TRAINING AND EVALUATION
--------------------------
Train the model:
   python src/train_breed.py --data dataset --epochs 30

Evaluate the model:
   python src/eval_breed.py --weights checkpoints/resnet50_breed.pth

Run inference:
   python src/predict_breed.py --image sample.jpg --weights checkpoints/resnet50_breed.ts

Expected output:
   Predicted breed: Golden Retriever (confidence: 0.94)

📊 MODEL PERFORMANCE
--------------------
Accuracy: ~92.4%  
F1-score: ~0.91  
Validation Loss: ~0.21  

🧩 MODEL EXPORTS
----------------
To export model weights to TorchScript and ONNX formats:
   python tools/export_model.py --ckpt checkpoints/resnet50_breed.pth --out_dir checkpoints

Outputs produced:
✔ resnet50_breed.ts  
✔ resnet50_breed.onnx  
✔ classes.txt  

🧰 TOOL SCRIPTS
---------------
split_imagefolder.py      → Splits dataset into train/val/test folders  
oxford_to_imagefolder.py  → Converts Oxford-IIIT dataset for ImageFolder style  
export_model.py           → Exports trained model into TorchScript & ONNX  

👤 AUTHOR
---------
Ozan Orhan  
Yıldız Technical University – Computer Engineering  
Email: ozanorhan2002@gmail.com  
GitHub: https://github.com/OzanOrhann  

🪄 ACKNOWLEDGMENTS
------------------
- Pretrained backbone: ResNet-50 (PyTorch model zoo)  
- Dataset preprocessing: torchvision, Pillow  
- Model export: TorchScript, ONNX Runtime
