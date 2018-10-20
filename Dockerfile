FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN python3 -m pip install numpy sklearn myo-python
COPY ./src /src
COPY ./data /data
CMD cd /src && python3 train.py