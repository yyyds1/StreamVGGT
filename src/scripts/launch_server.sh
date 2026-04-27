START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-vga_robotwin}

save_root='visualization/'
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    va_server.py \
    --config-name $CONFIG_NAME \
    --port $START_PORT \
    --save_root $save_root


