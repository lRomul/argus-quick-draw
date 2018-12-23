# Argus solution Quick, Draw! Doodle Recognition Challenge

Source code of 50th place solution for [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition) by Argus team ([Ruslan Baikulov](https://www.kaggle.com/romul0212), [Nikolay Falaleev](https://www.kaggle.com/nikolasent)).

## Solution 

We used PyTorch 1.0.0 with framework [Argus](https://github.com/lRomul/argus) and CNN architectures from [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune).

Key points: 
* Train SE-ResNet-50
* Use simplified data
* Encode time to RGB with color map from pyplot
* Country embeddings
* Gradient accumulation

## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 9, cuDNN >= 7
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided dockerfile is supplied to build image with cuda support and cudnn.


### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/lRomul/argus-quick-draw.git
    cd argus-quick-draw/docker 
    ./build.sh
    ```

* Download and extract [dataset](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)

### Run

* Run docker container 
```bash
cd docker
./run.sh
```

* Train model
```bash
python train.py
```

* Predict test and make submission 
```bash
python predict.py
```
