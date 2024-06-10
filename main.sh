pwd
echo 'processing waymo'
bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
         CUDA_VISIBLE_DEVICES=0 \
         python main.py --root '/path/to/pca/scene_flow' \
        --dataset='waymo'  --split='test' \
        --if_gpu  --gpu_idx=0 \
        --num_frames=5  --range_x=32.0  --range_y=32.0  --range_z=0.04  --ground_slack=0.3 \
        --if_hdbscan  --num_clusters=200  --min_cluster_size=30  --epsilon=0.25 \
        --speed=1.67  --thres_dist=0.1  --max_points=10000 \
        --thres_box=0.1  --thres_rot=0.1  --thres_error=0.3  --thres_iou=0.2 \
        --batch_size=1  --num_workers=4
        " 

echo 'processing waymo completed!'

echo 'processing nuscenes'
# better result than the paper
bash -c "python main.py --root '/path/to/pca/scene_flow' \
        --dataset='nuscene' \
        --if_gpu \
        --gpu_idx=0 \
        --if_save \
        --num_frames=11  --range_x=32.0  --range_y=32.0  --range_z=-1.84  --ground_slack=0.3 \
        --if_hdbscan  --num_clusters=200  --min_cluster_size=20  --epsilon=0.25 \
        --speed=0.833333  --thres_dist=0.1  --max_points=5000 \
        --thres_box=0.1  --thres_rot=0.1  --thres_error=0.2  --thres_iou=0.2 \
        --batch_size=1  --num_workers=16 \
        " 
echo 'processing nuscenes completed'

echo 'processing argo-v2'
bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
         CUDA_VISIBLE_DEVICES=0 \
            python main.py --root '/path/to/argo-v2/scene_flow' \
                --dataset='argo' \
                --split='val' \
                --num_frames=2  --range_x=10000.0  --range_y=10000.0  --range_z=-10000.0  --ground_slack=0.0 \
                --if_hdbscan  --num_clusters=200  --min_cluster_size=20  --epsilon=0.25 \
                --speed=1.67  --thres_dist=0.1  --max_points=10000 \
                --thres_box=0.1  --thres_rot=0.1  --thres_error=0.2  --thres_iou=0.2 \
                --num_workers=4
                "
echo 'processing argo-v2 completed!'
