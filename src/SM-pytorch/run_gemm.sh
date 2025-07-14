BASE=Megatron-LM # ML_Benchmarks_run # 
MAIN_DIR=/opt/tiger/$BASE/coll-test

nnodes=1
node_rank="${node_rank:=$ARNOLD_ID}"
master_port=12000
excute_file=$MAIN_DIR/gemm.py

nproc_per_node=8

args=(
    --fp16
    # --fp32
    # -m 2048 -k 5120 -n 20480
    --comm-size 256
    --comp-size 200
    --warmup 0
    --repeats 1000
    --fn all_to_all
    --hang
    --no-barrier
)

if [ $nproc_per_node -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ $nproc_per_node -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ $nproc_per_node -eq 16 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15
fi

torchrun \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --master_port=$master_port \
    $excute_file ${args[@]}
