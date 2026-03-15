set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0


export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}
BENCHMARK=${BENCHMARK:-AIME24}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-math_qwen3_1p7b_lora}
GPU_num=${GPU_num:-4}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
TRAIN_SAMPLE_NUM=${TRAIN_SAMPLE_NUM:-8}
VALIDATE_SAMPLE_NUM=${VALIDATE_SAMPLE_NUM:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
VAL_FREQ=${VAL_FREQ:-10}

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"


python3 -m pettingllms.trainer.train --config-path ../config/math --config-name math_L2_lora \
    $model_0_resource \
    base_models.policy_0.path="$BASE_MODEL_PATH"\
    training.experiment_name="$EXPERIMENT_NAME"\
    training.total_training_steps="$TOTAL_TRAINING_STEPS"\
    training.train_batch_size="$TRAIN_BATCH_SIZE"\
    training.train_sample_num="$TRAIN_SAMPLE_NUM"\
    training.validate_sample_num="$VALIDATE_SAMPLE_NUM"\
    training.max_prompt_length="$MAX_PROMPT_LENGTH"\
    training.max_response_length="$MAX_RESPONSE_LENGTH"\
    training.val_freq="$VAL_FREQ"\
    env.dataset=polaris\
    env.benchmark="$BENCHMARK"\
