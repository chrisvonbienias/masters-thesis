FROM nvidia/cuda:10.1-devel-ubuntu18.04

COPY . /app

RUN apt-get -y update && apt-get install -y python3.8 && apt-get install -y python3-pip && pip3 install virtualenv && apt-get install -y python3.8-dev
RUN virtualenv -p python3.8 app/.venv

# Requirements
RUN . app/.venv/bin/activate && pip3 install torch==1.7+cu101 -f https://download.pytorch.org/whl/torch_stable.html