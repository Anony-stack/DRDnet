# train
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 8882 --use-env train.py --config_file configs/ltcc/eva02_l_random.yml MODEL.DIST_TRAIN True