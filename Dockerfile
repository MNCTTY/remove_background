FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y \
    apt-utils \
    python3-pip 
    
RUN pip3 install --upgrade pip
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/

ARG PROJECT=removebg
ARG PROJECT_DIR=/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR
RUN pip3 install --upgrade scikit-image
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD gunicorn --access-logfile - -w 1 --bind 0.0.0.0:5000 app:app --timeout 15000
