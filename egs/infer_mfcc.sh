gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
WAVS=/datasets/AMI/media/sam/bx500/amicorpus/audio_eval
# CKPT=
# OUT=$EXP_DIR/preds/${CKPT}
OUT=$EXP_DIR/preds_tcn_single_array1_eval/
python local/infer.py $EXP_DIR $CKPT $WAVS $OUT $gpus --regex Array1-01