# ADNI CODE
## Quick Start
1. Download pretrained extractor weights and put the .pth file into ./checkpoint/extractor/ dir. Link: https://pan.baidu.com/s/1JxjxsbN1M7xSbQ9ZPlMdTA?pwd=fcra, extract code: fcra

2. Download dataset file and unzip it into ./dataset/ dir.Link: https://pan.baidu.com/s/1EqD2MY-zXmvgUsTWe-2kVA?pwd=dkyx, extract code: dkyx

Your dir structure should be like this: 
```
ADNI/
├── README.md
├── ourMethod_order1_train.py
├── ourMethod_order2_train.py
├── checkpoint/
│   └── extractor
│       └──model-ds128-AUC8346.pth
│
├── dataset/
│   ├── frfmwp1024_S_2239_ADNI1_GO-Month-6_I252342.npy
│   ├── frfmwp1024_S_2239_ADNI2-Initial-Visit-Cont-Pt_I270233.npy
│   ├── ...
│   └── frfmwp1041_S_4629_ADNI2-Month-3-MRI-New-Pt_I315574.npy
│   
├── model/
└── data/
```
3. run scripts
```
python ourMethod_order1_train.py
```
## Train with your own dataset
1. Preprocess AD data with your own method and convert it to npy format.
2. Create your own .xls file follow our method in /data/ directory.
3. Our method will first downsample the metadata from 181\*217\*181 to 128\*128\*128 in size. If you want to use data of other sizes, please retrain the pre-trained extractor and modify the code of the generator and discriminator.