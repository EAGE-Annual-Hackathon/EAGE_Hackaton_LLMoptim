#!/bin/bash
# https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html

MODEL_DIR='/workspace/models/'
MODEL="${MODEL_DIR}Llama-2-7b-chat.nemo"
TRAIN_DS="[/workspace/data/data_train.jsonl]"
CONCAT_SAMPLING_PROBS="[1.0]"

VALID_DS="[/workspace/data/data_val.jsonl]"
TEST_DS="[/workspace/data/data_test.jsonl]"
TEST_NAMES="[geo-llm]"

# Set $SCHEME="ptuning" for ptuning instead of lora.
SCHEME="lora"
# SCHEME="ptuning"

TP_SIZE=1
PP_SIZE=1

torchrun --nproc_per_node=1 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=20 \
    trainer.max_steps=50 \
    model.megatron_amp_O2=False \
    ++model.mcore_gpt=True \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.micro_batch_size=4 \
    model.global_batch_size=4 \
    model.restore_from_path=${MODEL} \
    model.data.train_ds.num_workers=0 \
    model.data.validation_ds.num_workers=0 \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
    model.data.validation_ds.file_names=${VALID_DS} \
    model.peft.peft_scheme=${SCHEME}
