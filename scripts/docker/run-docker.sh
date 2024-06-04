#!/bin/bash

docker run -it --rm \
	--gpus='"device=0"' \
	-e NVIDIA_API_KEY=${NVIDIA_API_KEY} \
	-e HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
	-v ${PWD}:/workspace \
	--name geo-llm \
	nvcr.io/nvidia/nemo:24.01.01.framework \
	/bin/bash
