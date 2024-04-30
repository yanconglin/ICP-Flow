echo 'demo'
pwd
bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
         CUDA_VISIBLE_DEVICES=0 \
         python demo.py --root './' \
        --dataset='argo'  --split='val' \
        --if_gpu  --gpu_idx=0 \
        --if_save \
        --num_frames=2  --range_x=10000.0  --range_y=10000.0  --range_z=-10000.0  --ground_slack=0.0 \
        --if_hdbscan  --num_clusters=200  --min_cluster_size=20  --epsilon=0.25 \
        --speed=1.0  --thres_dist=0.1  --max_points=10000 \
        --thres_box=0.1  --thres_rot=0.1  --thres_error=0.2  --thres_iou=0.2 \
        --batch_size=1  --num_workers=0 \
        --if_show
        " 
