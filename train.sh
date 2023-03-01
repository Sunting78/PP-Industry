### train detection
python3 tools/det/train.py -c configs/det/faster_rcnn/faster_rcnn_r101_1x_coco.yml 
### train segmentation
python3 tools/seg/train.py --config configs/topformer/seaformer_tiny_ade20k_512x512_160k.yml --do_eval  --use_vdl --save_interval 1000 --save_dir ./output/ --num_workers 4
### detection + roi segmentation 

