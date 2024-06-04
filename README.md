# Hands-on LoRA fine-tuning of Llama-2-7B with NeMo Framework
This repository contains step-by-step instructions on how to generate a synthetic dataset for supervised fine-tuning from a collection of EAGE abstracts released for EAGE Annual Hackathon 2023 (now we are in 2024).

### Setup environment

1. Register [NVIDIA Developer account](https://developer.nvidia.com/developer-program)
2. Access [NGC](https://catalog.ngc.nvidia.com/) using your developer account
3. Create private `NGC_API_KEY` on NGC
4. Login `docker login nvcr.io` and pull NeMo Framework container 
```
docker pull nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.05
```

5. Launch container
```
docker run -it --rm \
	--gpus='"device=0"' \
	-e NVIDIA_API_KEY=${NVIDIA_API_KEY} \
	-e HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
	-v ${PWD}:/workspace \
	--name geo-llm \
	nvcr.io/nvidia/nemo:24.01.01.framework \
	/bin/bash
```



### Run experiment
1. Prepare training dataset formatted following `prepare_datasets.ipynb`
2. Run [LoRA fine-tuning on Llama-2-7B](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/lora.ipynb)
```
./scripts/run-lora-training.sh
```

3. Run evaluation with [llm-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```
./scripts/run-evaluation.sh
```

-------------
Prepared by Oleg Ovcharenko and Oleg Sudakov
