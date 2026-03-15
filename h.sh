BASE_MODEL_PATH=Qwen/Qwen3-1.7B \
BENCHMARK=AIME24 \
EXPERIMENT_NAME=math_qwen3_1p7b_L3_4xA100 \
CHECKPOINT_DIR=./checkpoints \
MODEL_CHECKPOINTS_DIR=./checkpoints \
PPO_DEFAULT_LOCAL_DIR=./checkpoints \
bash scripts/train/math/run_qwen3_1p7b_L3_4xA100.sh