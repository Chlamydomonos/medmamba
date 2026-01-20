#!/bin/bash

# 训练脚本 - 训练6个模型（3种模型 × 2个数据集）

# 数据集路径
DATASET1="/root/train/dataset/gen_images"
DATASET2="/root/train/dataset/sussex_output_dir"

# 输出路径
OUTPUT1="./results/gen_images"
OUTPUT2="./results/sussex_output"

# 训练参数
EPOCHS=200
BATCH_SIZE=32
NUM_CLASSES=6
LR=0.0001

echo "========================================"
echo "开始训练所有模型"
echo "========================================"
echo ""

# ========== 训练 gen_images 数据集 ==========
echo "========================================"
echo "数据集: gen_images"
echo "========================================"

# 1. ResNet50 + gen_images
echo ""
echo "[1/6] 训练 ResNet50 - gen_images"
echo "----------------------------------------"
python trainer.py \
    --model resnet50 \
    --data_path "$DATASET1" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT1"

if [ $? -ne 0 ]; then
    echo "错误: ResNet50 (gen_images) 训练失败"
    exit 1
fi

# 2. DenseNet169 + gen_images
echo ""
echo "[2/6] 训练 DenseNet169 - gen_images"
echo "----------------------------------------"
python trainer.py \
    --model densenet169 \
    --data_path "$DATASET1" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT1"

if [ $? -ne 0 ]; then
    echo "错误: DenseNet169 (gen_images) 训练失败"
    exit 1
fi

# 3. MedMamba + gen_images
echo ""
echo "[3/6] 训练 MedMamba - gen_images"
echo "----------------------------------------"
python trainer.py \
    --model medmamba \
    --data_path "$DATASET1" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT1"

if [ $? -ne 0 ]; then
    echo "错误: MedMamba (gen_images) 训练失败"
    exit 1
fi

# ========== 训练 sussex_output_dir 数据集 ==========
echo ""
echo "========================================"
echo "数据集: sussex_output_dir"
echo "========================================"

# 4. ResNet50 + sussex_output_dir
echo ""
echo "[4/6] 训练 ResNet50 - sussex_output_dir"
echo "----------------------------------------"
python trainer.py \
    --model resnet50 \
    --data_path "$DATASET2" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT2"

if [ $? -ne 0 ]; then
    echo "错误: ResNet50 (sussex_output_dir) 训练失败"
    exit 1
fi

# 5. DenseNet169 + sussex_output_dir
echo ""
echo "[5/6] 训练 DenseNet169 - sussex_output_dir"
echo "----------------------------------------"
python trainer.py \
    --model densenet169 \
    --data_path "$DATASET2" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT2"

if [ $? -ne 0 ]; then
    echo "错误: DenseNet169 (sussex_output_dir) 训练失败"
    exit 1
fi

# 6. MedMamba + sussex_output_dir
echo ""
echo "[6/6] 训练 MedMamba - sussex_output_dir"
echo "----------------------------------------"
python trainer.py \
    --model medmamba \
    --data_path "$DATASET2" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir "$OUTPUT2"

if [ $? -ne 0 ]; then
    echo "错误: MedMamba (sussex_output_dir) 训练失败"
    exit 1
fi

# ========== 完成 ==========
echo ""
echo "========================================"
echo "所有模型训练完成！"
echo "========================================"
echo ""
echo "结果保存位置:"
echo "  gen_images 数据集: $OUTPUT1"
echo "  sussex_output_dir 数据集: $OUTPUT2"
echo ""
