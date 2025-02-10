#!/bin/bash
###### PREPORCESSING #######
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ma
# python script/waymo/waymo_converter --root_dir ../DATASETS/waymo/unprocessed/ --save_dir ../DATASETS/waymo/output/ --split_file script/waymo/waymo_splits/demo.txt --segment_file script/waymo/waymo_splits/segment_list_train.txt


#for ID in 0001 0002 0003 0004 0005; do
#  python script/waymo/generate_lidar_depth.py --datadir ../DATASETS/waymo/output/$ID/
#  python script/waymo/generate_sky_mask.py --datadir ../DATASETS/waymo/output/$ID/ --sam_checkpoint sam_vit_l_0b3195.pth
#  cp ../DATASETS/waymo/output/$ID/ data/waymo/training -r
#  cp configs/waymo_train_generic.yaml configs/waymo_train_$ID.yaml
#
#  END_FRAME=$($(( $(ls -1 data/waymo/training/$ID/ego_pose | wc -l) / 6 )) - 1)
#  START_FRAME=$(END_FRAME-100)
#  sed -i " s/PLACEHOLDER/$ID/g; s/START_FRAME/$START_FRAME/g; s/END_FRAME/$END_FRAME/g" configs/waymo_train_$ID.yaml
#  echo "Finished processing $ID"
#done

###### TRAINING #######

#for ID in 0001; do
#    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --config configs/waymo_train_$ID.yaml
#done

##### CHANGING TRAJECTORY #######
# python OWN_SCRIPTS/change_all.py --folder data/waymo/training/0001

##### RENDERING ########
python render.py --config configs/waymo_train_0001.yaml mode trajectory