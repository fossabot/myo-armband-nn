#!/bin/bash
docker rmi -f myo-train
docker rm -f myo-train
rm -r ../tf_export/1
