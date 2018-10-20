#!/bin/bash
docker rm -f myo-train-servable
docker run -p 8501:8501 \
--mount type=bind,source=$(pwd)/../tf_export,target=/models/myo-train \
-e MODEL_NAME=myo-train -t --name myo-train-servable tensorflow/serving &