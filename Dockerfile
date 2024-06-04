FROM nvcr.io/nvidia/nemo:24.01.01.framework

RUN pip install --upgrade pip && pip install lm-eval gdown langchain langchain_nvidia_ai_endpoints pypdf

CMD [/bin/bash]
