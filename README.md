## Environment Settings

The code has been tested with

* python 3.9
* torch 1.12
* cuda 11.3

Some dependencies:

```
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```

## Data Processing

### NOCS dataset

* Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
* Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under `PROJ\_DIR/data`and the final file structure is as follows:

```
data
├── camera
│   ├── train
│   ├── val
│   ├── train\_list\_all.txt
│   ├── train\_list.txt
│   ├── val\_list\_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train\_list.txt
│   ├── train\_list\_all.txt
│   └── test\_list\_all.txt
├── segmentation\_results
│   ├── CAMERA25
│   └── REAL275
├── camera\_full\_depths
├── gts
└── obj\_models
```

### HouseCat6D

Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:

```
HOUSECAT6D\_DIR
├── scene\*\*
├── val\_scene\*
├── test\_scene\*
└── obj\_models\_small\_size\_final
```

## Train

### Training on NOCS

```
python train.py --config config/REAL/camera\_real.yaml
```

### Training on HouseCat6D

```
python train\_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluate

* Evaluate on NOCS:

```
python test.py --config config/REAL/camera\_real.yaml --test\_epoch 30
```

* Evaluate on HouseCat6D:

```
python test\_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test\_epoch 150
```

## Visualization

For visualization, please run

```
python visualize.py --config config/REAL/camera\_real.yaml --test\_epoch 30
```

## 

