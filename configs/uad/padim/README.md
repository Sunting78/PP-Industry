# PaDiM-Anomaly-Detection-Localization
This is an implementation of the paper [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).

## Results
### Implementation results on MVTec
* Image-level and Pixel level accuracy (ROCAUC)

|MvTec| R18-Rd100(Image-level) | R18-Rd100(Pixel-level) |
|:---:|:----------------------:|:----------------------:|
|Carpet|         0.996          |         0.990          | 
|Grid|         0.929          |         0.933          | 
|Leather|         0.996          |         0.989          |
|Tile|         0.948          |         0.862          | 
|Wood|         0.986          |         0.922          | 
|All texture classes|         0.971          |         0.939          |
|Bottle|         0.995          |         0.977          |
|Cable|         0.873          |         0.945          | 
|Capsule|         0.873          |         0.981          | 
|Hazelnut|         0.843          |         0.971          | 
|Metal nut|         0.976          |         0.962          |
|Pill|         0.923          |         0.937          |
|Screw|         0.750          |         0.971          |
|Toothbrush|         0.967          |         0.985          |
|Transistor|         0.971          |         0.972          |
|Zipper|         0.919          |         0.980          
|All object classes|         0.909          |         0.968          |
|All classes|         0.940          |         0.954          | 
