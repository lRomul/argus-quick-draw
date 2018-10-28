FROM floydhub/pytorch:0.4.1-gpu.cuda9cudnn7-py3.34

RUN pip3 install --upgrade pip &&\
    pip3 install \
    torchsummary\
    pretrainedmodels\
    pytorch-argus==0.0.5\
    cnn-finetune==0.5.1\
    tqdm

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
