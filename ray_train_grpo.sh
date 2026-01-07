# for python path in 10.176.44.14
export PYTHONPATH=""
export TMPDIR=/home/whr/tmp
mkdir -p /home/whr/tmp
export HF_DATASETS_CACHE=/home/whr/hf_datasets_cache
export HF_TRANSFORMERS_CACHE=/home/whr/hf_transformers_cache
export TRANSFORMERS_CACHE=/home/whr/hf_transformers_cache
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_TRANSFORMERS_CACHE


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
export DATA_DIR="./dataset/chatops"
export BASE_MODEL="/chatops_models/Qwen3-4B-Instruct-2507" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="grpo-qwen3-1.7B-1_3" # e.g., "grpo-qwen2.5-3b"
bash ./examples/grpo_trainer/ray_run_grpo.sh