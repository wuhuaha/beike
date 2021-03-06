export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_qa"

python -u classify_demo.py \
                   --batch_size=25 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=5e-5 \
                   --weight_decay=0.01 \
                   --warmup_proportion=0.1 \
                   --max_seq_len=360\
                   --num_epoch=8 \
                   --use_data_parallel=True \
