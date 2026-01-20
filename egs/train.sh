gpus=2
export CUDA_VISIBLE_DEVICES=$gpus
EXP_DIR=local/exp/
CONFS=/mnt/sdb2/lsj/Interspeech2025/egs/local/train.yml

# mkdir -p $EXP_DIR
# cp -r local $EXP_DIR/code
python local/train.py $CONFS $EXP_DIR $gpus
 