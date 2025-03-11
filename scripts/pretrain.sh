
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    base_model/train/train.py \
    --config-file base_model/configs/vits16_short.yaml \
    --output-dir save/vits16_kinetics/epoch_400/p-0.15-0.25#c-0.3-0.7#f-0.15-0.25/pt-0.8#ft-0.8#pf-20/block-1#base_lr-0.002#lr_mul-0.1#epoch-400 \
    student.arch=vit_small \
    student.patch_size=16 \
    auxiliary.use_auxiliary=true \
    auxiliary.lr_mul=0.1 \
    ibot.past_tea_ibot_loss_weight=0.8 \
    ibot.future_tea_ibot_loss_weight=0.8 \
    ibot.past_future_MSE_loss_weight=20 \
    optim.base_lr=0.002 \
    optim.epochs=400 \
    optim.warmup_epochs=20 \
    train.past_offset_range=[0.15,0.25] \
    train.current_range=[0.3,0.7] \
    train.future_offset_range=[0.15,0.25] \
    train.dataset=Kinetics \
    train.dataset_path=Kinetics:split=TRAIN:root=data/Kinetics-400/frames:extra=data/Kinetics-400/frames \
    train.batch_size_per_gpu=4 \
    train.OFFICIAL_EPOCH_LENGTH=936 \
    evaluation.eval_period_iterations=$((936*20))

