### train detection
python3 tools/det/train.py -c configs/det/faster_rcnn/faster_rcnn_r101_1x_coco.yml 
### train segmentation
python3 tools/seg/train.py --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_1024x512_160k.yml --do_eval  --use_vdl --save_interval 1000 --save_dir ./output/ --num_workers 4
### detection + roi segmentation 

python3 tools/convert_tools/convert_coco_to_RoI_mask.py --json_path /ssd3/sunting/PP-Industry/dataset/coco/annotations/instances_val2017.json --image_path /ssd3/sunting/PP-Industry/dataset/coco/val2017 --seg_classid 1 2

python3 tools/end2end/predict.py --config ./configs/end2end/e2e_det_RoI_seg.yml --input ./dataset/kolektor2/images/tmp