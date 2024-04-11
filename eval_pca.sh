# # waymo
# # error
echo 'processing waymo'

# local debug: conda
source /home/yanconglin/miniconda3/etc/profile.d/conda.sh
conda activate sf_icp 
cd /home/yanconglin/Desktop/IV/scene_flow/SceneFlowICP
pwd
# evaluation:
# bash -c "python eval_pca.py --root '/media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset/scene_flow/eth_scene_flow' \
#         --dataset='waymo'  --if_save \
#         --num_frames=5  --range_x=32.0  --range_y=32.0  --range_z=0.04  --ground_slack=0.3 \
#         --num_workers=0 \
#         > test_pca_waymo.txt
#         " 
bash -c "python eval_pca.py --root '/media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset/scene_flow/eth_scene_flow' \
        --dataset='nuscene'  --if_save \
        --num_frames=11  --range_x=32.0  --range_y=32.0  --range_z=-1.84  --ground_slack=0.3 \
        --num_workers=0 \
        > test_pca_nuscene.txt
        " 






