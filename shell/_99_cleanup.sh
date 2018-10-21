#!/bin/bash
docker rmi -f myo-train
docker rm -f myo-train
sudo chmod u+w ../tf_export/*
sudo rm -r ../tf_export/1
