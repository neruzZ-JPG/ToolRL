export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export ROLLOUT_TP_SIZE=1
# export VLLM_ATTENTION_BACKEND=XFORMERS // only for vllm<=0.6.3
# All the env variables below are set to 0 by default
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0
export VLLM_USE_V1=1
export WANDB_API_KEY="ea5ca61fbc3e4fbe1822b941b9e06a38dab60933"
# 
export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="Qwen3-0.6B" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="grpo-qwen3-0.6B" # e.g., "grpo-qwen2.5-3b"
bash ./examples/grpo_trainer/run_grpo.sh