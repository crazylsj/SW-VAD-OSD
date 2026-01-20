#!/bin/bash

# 配置文件路径
CONF_FILE="/mnt/sdb2/lsj/OSDC-mamba/egs/local/test.yml"

# 日志目录
LOG_DIR="log_dir"

# 模型检查点文件
MODEL_CHECKPOINT="/mnt/sdb2/lsj/OSDC-mamba/egs/exp/tcn/tcn_encoder_2.pth"

# 测试数据目录
TEST_DATA="/mnt/sdb2/lsj/datasets/AMI/media/sam/bx500/amicorpus/fa_labels/eval/"
# CSD="Frequency-Selected Diagonal CSD"

# 运行测试
python local/test_right.py "$CONF_FILE" "$LOG_DIR" "$MODEL_CHECKPOINT" "$TEST_DATA" 
# python local/test_csd.py "$CONF_FILE" "$LOG_DIR" "$MODEL_CHECKPOINT" "$TEST_DATA" "$CSD"
