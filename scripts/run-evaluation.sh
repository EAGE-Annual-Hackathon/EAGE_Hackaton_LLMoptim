# The number of total GPUs available
GPU_COUNT=1

# Change this to the nemo model you want to use
# MODEL="./llama2-7b.nemo"
MODEL_DIR='/workspace/models/'
MODEL="${MODEL_DIR}Llama-2-7b-chat.nemo"

# This will live in whatever $OUTPUT_DIR was set to in the training script above
# The filename will match whichever peft scheme was used during training
# PATH_TO_TRAINED_MODEL="./results/checkpoints/megatron_gpt_peft_lora_tuning.nemo"
OUTPUT_DIR="/workspace/nemo_experiments/megatron_gpt_peft_lora_tuning/"
PATH_TO_TRAINED_MODEL="${OUTPUT_DIR}checkpoints/megatron_gpt_peft_lora_tuning.nemo"

# The test dataset
TEST_DS="/workspace/data/data_test.jsonl"
TEST_NAMES="geo-test"

# This is the prefix, including the path and filename prefix, for the accuracy file output
# This will be combined with TEST_NAMES to create the file ./results/peft_results_test_pubmedqa_inputs_preds_labels.jsonl
# OUTPUT_PREFIX="./results/peft_results"
OUTPUT_PREFIX="${OUTPUT_DIR}/peft_results"

# TOKENS_TO_GENERATE=20

# # This is the tensor parallel size (splitting tensors among GPUs horizontally)
# TP_SIZE=1

# # This is the pipeline parallel size (splitting layers among GPUs vertically)
# PP_SIZE=1

# Execute
# python \
# torchrun --nproc_per_node=8 \
# /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
#     model.restore_from_path=${MODEL} \
#     model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
#     trainer.devices=${GPU_COUNT} \
#     model.data.test_ds.file_names=[${TEST_DS}] \
#     model.data.test_ds.names=[${TEST_NAMES}] \
#     model.data.test_ds.global_batch_size=4 \
#     model.data.test_ds.micro_batch_size=1 \
#     model.data.test_ds.tokens_to_generate=${TOKENS_TO_GENERATE} \
#     model.tensor_model_parallel_size=${TP_SIZE} \
#     model.megatron_amp_O2=True \
#     model.pipeline_model_parallel_size=${PP_SIZE} \
#     inference.greedy=True \
#     model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
#     model.answer_only_loss=True \
#     model.data.test_ds.write_predictions_to_file=True

TOKENS_TO_GENERATE=64

GPU_COUNT=1
TP_SIZE=1
PP_SIZE=1

torchrun --nproc_per_node=1 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    trainer.num_nodes=1 \
    model.restore_from_path=${MODEL} \
    model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=${GPU_COUNT} \
    model.data.test_ds.file_names=[${TEST_DS}] \
    model.data.test_ds.names=[${TEST_NAMES}] \
    model.data.test_ds.global_batch_size=4 \
    model.data.test_ds.micro_batch_size=4 \
    model.micro_batch_size=4 \
    model.global_batch_size=4 \
    model.data.test_ds.tokens_to_generate=${TOKENS_TO_GENERATE} \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.megatron_amp_O2=True \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
    model.answer_only_loss=True \
    model.data.test_ds.write_predictions_to_file=True