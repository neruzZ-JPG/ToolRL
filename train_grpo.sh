ray stop --force
# for python path in 10.176.44.14
export PYTHONPATH=""
export TMPDIR=/home/whr/tmp
mkdir -p /home/whr/tmp
export HF_DATASETS_CACHE=/home/whr/hf_datasets_cache
export HF_TRANSFORMERS_CACHE=/home/whr/hf_transformers_cache
export TRANSFORMERS_CACHE=/home/whr/hf_transformers_cache
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_TRANSFORMERS_CACHE

# ==========================================
# 0. CUDA 环境锁定 (非常重要！)
# ==========================================
# 请修改为你实际安装新版 CUDA 的路径
# 如果是按我之前的推荐安装的 runfile，路径应该是：
export CUDA_HOME="/home/whr/cuda-12.1"

# 强制将新版 CUDA 的 bin 和 lib 放到最前面，覆盖系统的旧版本
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 验证一下（可选，方便在日志里确认用了对的版本）
echo "Using CUDA Version:"
nvcc -V || echo "Warning: nvcc not found in specified CUDA_HOME"

export RAY_ADDRESS="auto"

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
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
export DATA_DIR="./dataset/chatops/union"
export BASE_MODEL="Qwen3-0.6B" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="grpo-qwen3-0.6B" # e.g., "grpo-qwen2.5-3b"
ray start --head --disable-usage-stats --port=6379
bash ./examples/grpo_trainer/run_grpo.sh 