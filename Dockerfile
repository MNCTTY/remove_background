FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y \
    apt-utils \
    wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev liblzma-dev 
RUN wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz &&\
    tar -xf Python-3.8.0.tgz &&\
    cd Python-3.8.0 &&\
    ./configure --enable-optimizations &&\
    make -j8 &&\
    make altinstall
    
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 3 &&\
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 3 &&\
    pip3 install --upgrade pip

RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/

ARG PROJECT=removebg
ARG PROJECT_DIR=/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR
RUN apt-get update --fix-missing && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD gunicorn --access-logfile - -w 1 --bind 0.0.0.0:5000 app:app --timeout 15000
