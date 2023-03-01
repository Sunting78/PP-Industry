# PP-Industry

## Introduciton
PaddleIndustry aims to create multilingual, awesome, leading, and practical industry models and tools that help users train better models and apply them into practice.
## Installation
  - 1.1 Install PaddlePaddle
  - 1.2 Install PaddleDetetion
  - 1.3 Install PaddleSegmentation

## Quick Start
### Train

* Detection

  ```bash
  python3 tools/det/train.py -c configs/det/faster_rcnn/faster_rcnn_r101_1x_coco.yml 
  ```
* Segmentation

  ```bash
  python3 tools/seg/train.py --config configs/seg/topformer/seaformer_tiny_ade20k_512x512_160k.yml --do_eval  --use_vdl --save_interval 1000
  ```

* Detection and RoI Segmentation
  First, deal with your data to generate coco.json format and RoI segmentation mask format.

  ```bash
  python3 tools/data_process/conver_coco.py --json_path xxx --output_path xxx
  ```
## Overall directory structure

The overall directory structure of PaddleOCR is introduced as follows:


```
PaddleIndustry
├── configs                                 // Configuration file, you can config the model structure and modify the hyperparameters through the yml file
│   ├── det                                 // Defect detection config files
│   │   ├── faster_rcnn.yml                  // Training config
│   │   ...  
│   └── seg                                 // defect segmentation config files
│       ├── hrnet_ocr.yml     // hrnet config
│       ...  
├── deploy                                  // Depoly
|   ...                                
├── doc                                     // Documentation and Tutorials
│   ...
├── ppindustry                               // Core code
│   ├── det                                 // detection code
│   │   ├── datasets                        // defect data loder code
│   │   ├── transforms                     // defect data augmentation
│   │   └── models                          // defect models
│   ├── seg                             // Metrics
│   │   ├── datasets                        // defect data loder code
│   │   ├── transforms                     // defect data augmentation
│   │   └── models                          // defect models
│   ├── postprocess                         // Post-processing
│   │   ├── det_postprocess.py              // class post-processing
│   │   └── seg_postprocess.py               // DB post-processing
│   └── utils                               // utils
├── tools
│   ├── end2end.py                           // Evaluation function end-to-end
│   ├── export_model.py                     // Export inference model
│   ├── det                                // Detection train predict and inference 
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── infer.py
|   |   ...
│   ├── seg                                // Segmentation train predict and inference 
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── infer.py
|   |   ...
├── README_ch.md                            // Chinese documentation
├── README_en.md                            // English documentation
├── README.md                               // Home page documentation
├── requirements.txt                         // Requirements
