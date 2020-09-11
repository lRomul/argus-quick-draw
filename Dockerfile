FROM floydhub/pytorch:1.0.0-gpu.cuda9cudnn7-py3.37

# Install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
