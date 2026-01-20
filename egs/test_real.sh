#!/bin/bash

# 配置文件路径
CONF_FILE="/mnt/sdb1/lsj/OSDC-mamba/egs/local/test.yml"

# 日志目录
LOG_DIR="log_dir"

# 模型检查点文件
MODEL_CHECKPOINT="/mnt/sdb1/lsj/OSDC-mamba/egs/exp/tcn/case7.pth"

# 测试数据目录
TEST_DATA="/mnt/sdb1/lsj/datasets/AMI/media/sam/bx500/amicorpus/fa_labels/eval_s/"
# CSD="Frequency-Selected Diagonal CSD"

# 运行测试
python local/test_csdopp.py "$CONF_FILE" "$LOG_DIR" "$MODEL_CHECKPOINT" "$TEST_DATA" 
# python local/test_csd.py "$CONF_FILE" "$LOG_DIR" "$MODEL_CHECKPOINT" "$TEST_DATA" "$CSD"
