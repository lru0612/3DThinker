#!/bin/bash
###############################################################################
# 3DThinker 一键复现脚本
# 功能: 环境配置 → 模型下载 → 数据下载 → Eval → 点云恢复
# 用法: bash run_all.sh
###############################################################################
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="3DThinker-eval"
MODEL_DIR="$REPO_ROOT/models"
MINDCUBE_DIR="$REPO_ROOT/MindCube-main"
EVAL_RESULTS_DIR="$REPO_ROOT/results/eval"
RECON_RESULTS_DIR="$REPO_ROOT/tests/recon_results"

MODEL_NAME="3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12"
EVAL_JSONL="$MINDCUBE_DIR/data/prompts/general/MindCube_tinybench_raw_qa.jsonl"

log_step() {
    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] $1"
    echo "========================================"
}

###############################################################################
# Step 1: 环境配置
###############################################################################
log_step "Step 1/7: 配置 conda 环境"

# 初始化 conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook 2>/dev/null)" || {
        echo "ERROR: 找不到 conda，请先安装 Anaconda/Miniconda"
        exit 1
    }
fi

# 创建环境（已存在则跳过）
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "conda 环境 $CONDA_ENV 已存在，跳过创建"
else
    conda create -n "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

echo "当前 Python: $(which python)"
echo "当前环境: $CONDA_ENV"

###############################################################################
# Step 2: 安装依赖
###############################################################################
log_step "Step 2/7: 安装依赖"

pip install -r "$REPO_ROOT/envs/requirements_stage1.txt"

# 安装自定义 transformers（支持 latent token 生成逻辑，覆盖标准版）
pip install -e "$REPO_ROOT/3dthinker/stage1/transformers/"

# 点云恢复额外依赖
pip install trimesh scipy
pip install pycolmap 2>/dev/null || echo "WARNING: pycolmap 安装失败，COLMAP 格式输出将跳过，PLY 正常保存"

###############################################################################
# Step 3: 下载模型
###############################################################################
log_step "Step 3/7: 下载模型"

mkdir -p "$MODEL_DIR"

# 3DThinker checkpoint
if [ -f "$MODEL_DIR/$MODEL_NAME/config.json" ]; then
    echo "3DThinker 模型已存在，跳过下载"
else
    echo "下载 3DThinker-Mindcube 模型..."
    huggingface-cli download jankin123/3DThinker-Mindcube \
        --local-dir "$MODEL_DIR/$MODEL_NAME"
fi

# VGGT-1B
if [ -f "$MODEL_DIR/vggt/model.pt" ]; then
    echo "VGGT 模型已存在，跳过下载"
else
    echo "下载 VGGT-1B 模型..."
    mkdir -p "$MODEL_DIR/vggt"
    huggingface-cli download facebook/VGGT-1B model.pt \
        --local-dir "$MODEL_DIR/vggt"
fi

###############################################################################
# Step 4: 下载 MindCube 数据
###############################################################################
log_step "Step 4/7: 下载 MindCube 数据"

# 克隆 MindCube 仓库（含 eval JSONL）
if [ -f "$EVAL_JSONL" ]; then
    echo "MindCube 仓库已存在，跳过克隆"
else
    echo "克隆 MindCube 仓库..."
    git clone https://github.com/mll-lab-nu/MindCube.git "$MINDCUBE_DIR" || {
        echo "ERROR: 克隆 MindCube 仓库失败"
        exit 1
    }
fi

# 下载图片数据
if [ -d "$REPO_ROOT/data/other_all_image" ]; then
    echo "MindCube 图片数据已存在，跳过下载"
else
    echo "下载 MindCube 图片数据..."
    TMP_DATA_DIR="$REPO_ROOT/MindCube-data-tmp"
    huggingface-cli download MLL-Lab/MindCube --repo-type dataset --local-dir "$TMP_DATA_DIR" || {
        echo ""
        echo "ERROR: MindCube 图片数据下载失败"
        echo "请手动下载并放置到 $REPO_ROOT/data/other_all_image/"
        echo "下载地址: https://huggingface.co/datasets/MLL-Lab/MindCube"
        exit 1
    }
    # 将图片移动到正确位置
    if [ -d "$TMP_DATA_DIR/data/other_all_image" ]; then
        cp -r "$TMP_DATA_DIR/data/other_all_image" "$REPO_ROOT/data/other_all_image"
    elif [ -d "$TMP_DATA_DIR/other_all_image" ]; then
        cp -r "$TMP_DATA_DIR/other_all_image" "$REPO_ROOT/data/other_all_image"
    else
        echo "WARNING: 下载完成但未找到 other_all_image 目录，请检查 $TMP_DATA_DIR 的内容"
        echo "手动将图片放到 $REPO_ROOT/data/other_all_image/ 后重新运行"
        exit 1
    fi
    echo "图片数据已就位"
fi

# 最终检查
if [ ! -d "$REPO_ROOT/data/other_all_image" ]; then
    echo "ERROR: 图片数据不存在于 $REPO_ROOT/data/other_all_image/"
    echo "请手动下载 MindCube 图片数据"
    exit 1
fi

###############################################################################
# Step 5: 运行 Eval Inference
###############################################################################
log_step "Step 5/7: 运行 Eval 推理"

mkdir -p "$EVAL_RESULTS_DIR"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

cd "$REPO_ROOT/eval"
python scripts/run_inference_clean.py \
    --model-type qwen2.5vl \
    --model-path "$MODEL_DIR/$MODEL_NAME" \
    --input-file "$EVAL_JSONL" \
    --output-dir "$EVAL_RESULTS_DIR"

###############################################################################
# Step 6: 运行 Eval 评估
###############################################################################
log_step "Step 6/7: 计算评估指标"

# 找到推理输出文件
INFERENCE_OUTPUT=$(ls -t "$EVAL_RESULTS_DIR"/*.jsonl 2>/dev/null | head -1)
if [ -z "$INFERENCE_OUTPUT" ]; then
    echo "ERROR: 未找到推理输出文件"
    exit 1
fi
echo "评估文件: $INFERENCE_OUTPUT"

cd "$REPO_ROOT/eval"
python scripts/run_evaluation.py \
    -t basic \
    -i "$INFERENCE_OUTPUT" \
    -o "$EVAL_RESULTS_DIR/eval_results.json"

echo ""
echo "Eval 结果已保存到: $EVAL_RESULTS_DIR/eval_results.json"
cat "$EVAL_RESULTS_DIR/eval_results.json"

###############################################################################
# Step 7: 点云恢复
###############################################################################
log_step "Step 7/7: 运行点云恢复"

# 设置 PYTHONPATH（vggt 模块在 preprocessing/feature/ 下）
export PYTHONPATH="$REPO_ROOT/preprocessing/feature:${PYTHONPATH:-}"

cd "$REPO_ROOT/tests"
python visual_decoder.py

echo ""
echo "========================================"
echo "全部完成!"
echo "========================================"
echo ""
echo "Eval 结果:    $EVAL_RESULTS_DIR/eval_results.json"
echo "点云输出:     $RECON_RESULTS_DIR/"
echo ""
echo "每个样本目录 (sparse_N/) 包含:"
echo "  - points.ply          3DThinker 恢复的点云"
echo "  - vggt_points.ply     VGGT 原始点云（对比用）"
echo "  - prompt.txt          输入问题"
echo "  - gt_answer.txt       标准答案"
echo "  - generated_answer.txt 模型生成的回答"
echo ""
echo "可用 MeshLab 或 Open3D 查看 PLY 文件"
