echo 'processing waymo'
source /home/yanconglin/miniconda3/etc/profile.d/conda.sh
conda activate sf_icp 
pwd

bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
         CUDA_VISIBLE_DEVICES=0 \
         python main.py --root '/media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset/scene_flow/eth_scene_flow' \
        --dataset='waymo'  --split='test' \
        --if_gpu  --gpu_idx=0 \
        --num_frames=5  --range_x=32.0  --range_y=32.0  --range_z=0.04  --ground_slack=0.3 \
        --if_hdbscan  --num_clusters=200  --min_cluster_size=30  --epsilon=0.25 \
        --speed=1.67  --thres_dist=0.1  --max_points=10000 \
        --thres_box=0.1  --thres_error=0.3  --thres_iou=0.2 \
        --batch_size=1  --num_workers=4
        " 

echo 'processing waymo done!'
