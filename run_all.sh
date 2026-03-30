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
log_step "Step 1/8: 配置 conda 环境"

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
log_step "Step 2/8: 安装依赖"

pip install -r "$REPO_ROOT/envs/requirements_stage1.txt"

# 安装自定义 transformers（支持 latent token 生成逻辑，覆盖标准版）
pip install -e "$REPO_ROOT/3dthinker/stage1/transformers/"

# 点云恢复额外依赖
pip install trimesh scipy
pip install pycolmap 2>/dev/null || echo "WARNING: pycolmap 安装失败，COLMAP 格式输出将跳过，PLY 正常保存"

###############################################################################
# Step 3: 下载模型
###############################################################################
log_step "Step 3/8: 下载模型"

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
log_step "Step 4/8: 下载 MindCube 数据"

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
    # 如果下载的是 zip 文件则先解压
    if [ -f "$TMP_DATA_DIR/data.zip" ]; then
        echo "检测到 data.zip，正在解压..."
        unzip -q "$TMP_DATA_DIR/data.zip" -d "$TMP_DATA_DIR"
        echo "解压完成"
    fi

    # 将图片移动到正确位置
    if [ -d "$TMP_DATA_DIR/data/other_all_image" ]; then
        cp -r "$TMP_DATA_DIR/data/other_all_image" "$REPO_ROOT/data/other_all_image"
    elif [ -d "$TMP_DATA_DIR/other_all_image" ]; then
        cp -r "$TMP_DATA_DIR/other_all_image" "$REPO_ROOT/data/other_all_image"
    else
        echo "WARNING: 解压后未找到 other_all_image 目录，请检查 $TMP_DATA_DIR 的内容"
        echo "目录结构如下:"
        ls "$TMP_DATA_DIR"
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
# Step 5: 运行 Eval Inference（3D 想象模式）
###############################################################################
log_step "Step 5/7: 运行 Eval 推理（3DThinker，含 3D 想象）"

mkdir -p "$EVAL_RESULTS_DIR"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

cd "$REPO_ROOT/eval"
python scripts/run_inference_clean.py \
    --model-type qwen2.5vl \
    --model-path "$MODEL_DIR/$MODEL_NAME" \
    --input-file "$EVAL_JSONL" \
    --output-dir "$EVAL_RESULTS_DIR/3d" \
    --prompt-mode 3d

###############################################################################
# Step 6: 运行 Eval Inference（普通 CoT，不做 3D 想象）
###############################################################################
log_step "Step 6/7: 运行 Eval 推理（普通 CoT，不做 3D 想象）"

cd "$REPO_ROOT/eval"
python scripts/run_inference_clean.py \
    --model-type qwen2.5vl \
    --model-path "$MODEL_DIR/$MODEL_NAME" \
    --input-file "$EVAL_JSONL" \
    --output-dir "$EVAL_RESULTS_DIR/plain" \
    --prompt-mode plain

###############################################################################
# Step 7: 计算两种模式的评估指标并对比
###############################################################################
log_step "Step 7/7: 计算评估指标并对比"

cd "$REPO_ROOT/eval"

# 3D 想象模式
INFERENCE_3D=$(ls -t "$EVAL_RESULTS_DIR/3d/"*.jsonl 2>/dev/null | head -1)
if [ -z "$INFERENCE_3D" ]; then
    echo "ERROR: 未找到 3D 模式推理输出"
    exit 1
fi
python scripts/run_evaluation.py \
    -t basic \
    -i "$INFERENCE_3D" \
    -o "$EVAL_RESULTS_DIR/3d/eval_results.json"

# 普通 CoT 模式
INFERENCE_PLAIN=$(ls -t "$EVAL_RESULTS_DIR/plain/"*.jsonl 2>/dev/null | head -1)
if [ -z "$INFERENCE_PLAIN" ]; then
    echo "ERROR: 未找到 plain 模式推理输出"
    exit 1
fi
python scripts/run_evaluation.py \
    -t basic \
    -i "$INFERENCE_PLAIN" \
    -o "$EVAL_RESULTS_DIR/plain/eval_results.json"

# 合并对比结果
export EVAL_RESULTS_DIR
python3 - <<'PYEOF'
import json, os, sys

results_dir = os.environ.get("EVAL_RESULTS_DIR", "results/eval")

def load_acc(path):
    with open(path) as f:
        d = json.load(f)
    return d["results"]["gen_cogmap_accuracy"]

acc_3d    = load_acc(f"{results_dir}/3d/eval_results.json")
acc_plain = load_acc(f"{results_dir}/plain/eval_results.json")

summary = {
    "3d_thinking_accuracy":    round(acc_3d,    4),
    "plain_cot_accuracy":      round(acc_plain, 4),
    "delta (3d - plain)":      round(acc_3d - acc_plain, 4),
}
print("\n===== 结果对比 =====")
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(f"{results_dir}/comparison.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n对比结果已保存到: {results_dir}/comparison.json")
PYEOF

###############################################################################
# Step 8: 点云恢复
###############################################################################
log_step "Step 8/8: 运行点云恢复"

# 设置 PYTHONPATH（vggt 模块在 preprocessing/feature/ 下）
export PYTHONPATH="$REPO_ROOT/preprocessing/feature:${PYTHONPATH:-}"

cd "$REPO_ROOT/tests"
python visual_decoder.py

echo ""
echo "========================================"
echo "全部完成!"
echo "========================================"
echo ""
echo "3D 想象 Eval:  $EVAL_RESULTS_DIR/3d/eval_results.json"
echo "普通 CoT Eval: $EVAL_RESULTS_DIR/plain/eval_results.json"
echo "对比结果:      $EVAL_RESULTS_DIR/comparison.json"
echo "点云输出:      $RECON_RESULTS_DIR/"
echo ""
echo "每个样本目录 (sparse_N/) 包含:"
echo "  - points.ply          3DThinker 恢复的点云"
echo "  - vggt_points.ply     VGGT 原始点云（对比用）"
echo "  - prompt.txt          输入问题"
echo "  - gt_answer.txt       标准答案"
echo "  - generated_answer.txt 模型生成的回答"
echo ""
echo "可用 MeshLab 或 Open3D 查看 PLY 文件"
