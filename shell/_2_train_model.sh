#!/bin/bash
docker rm -f myo-train
docker run --runtime=nvidia -it \
--mount type=bind,source="$(pwd)"/../tf_export,target=/tf_export \
--mount type=bind,source="$(pwd)"/../tf_session,target=/tf_session \
--name myo-train -p 8888:8888 myo-train
