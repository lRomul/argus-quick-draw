FROM floydhub/pytorch:1.0.0-gpu.cuda9cudnn7-py3.37

RUN pip3 install --upgrade pip &&\
    pip3 install \
    torchsummary\
    pretrainedmodels\
    cnn-finetune==0.5.1\
    tqdm

RUN git clone -b data_parallel --single-branch https://github.com/lRomul/argus.git
RUN cd argus $$ python setup.py install

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
