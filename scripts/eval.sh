
export CUDA_VISIBLE_DEVICES=0

DOWNSTREAM_OUTPUT_DIR='eval_results/epoch_400/p-0.15-0.25#c-0.3-0.7#f-0.15-0.25/pt-0.8#ft-0.8#pf-20/block-1#base_lr-0.002#lr_mul-0.1#epoch-400'
PRETRAINED_WEIGHT_DIR='save/vits16_kinetics/epoch_400/p-0.15-0.25#c-0.3-0.7#f-0.15-0.25/pt-0.8#ft-0.8#pf-20/block-1#base_lr-0.002#lr_mul-0.1#epoch-400'


python3 -m downstreams.propagation.start --davis \
    --backbone=vits \
    --patch_size=16 \
    --davis_file=downstreams/propagation/davis_vallist_480_880.txt \
    --davis_path=data/DAVIS/davis-2017/ \
    --config_file=/home/liuyang/dinov2/dinov2/configs/train/vits16_short.yaml \
    --downstream_output_dir=${DOWNSTREAM_OUTPUT_DIR} \
    --pretrained_weights=${PRETRAINED_WEIGHT_DIR}


python3 -m downstreams.propagation.start --jhmdb \
    --backbone=vits \
    --patch_size=16 \
    --jhmdb_file=downstreams/propagation/jhmdb_vallist.txt \
    --jhmdb_path=data/JHMDB/ \
    --config_file=/home/liuyang/dinov2/dinov2/configs/train/vits16_short.yaml \
    --downstream_output_dir=${DOWNSTREAM_OUTPUT_DIR} \
    --pretrained_weights=${PRETRAINED_WEIGHT_DIR}


python3 -m downstreams.propagation.start --vip \
    --backbone=vits \
    --patch_size=16 \
    --vip_file=downstreams/propagation/vip_vallist.txt \
    --vip_path=data/VIP/ \
    --config_file=/home/liuyang/dinov2/dinov2/configs/train/vits16_short.yaml \
    --downstream_output_dir=${DOWNSTREAM_OUTPUT_DIR} \
    --pretrained_weights=${PRETRAINED_WEIGHT_DIR}
