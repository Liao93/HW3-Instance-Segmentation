# HW3-Instance-Segmentation

Instance segmentation using Mask R-CNN and mmdetection.

# Environment

- Python 3.7.12
- mmcv-full 1.4.0
- mmdet 2.19.1
- openmim 0.1.5
- torch 1.10.0+cu111
- torchvision 0.11.1+cu111

# Training

Use pre-trained model 'mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth' from mmdetection.
(https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/README.md)
The path can be set in the config file.

These files should be specified a path. You can set the path on your own:
- 'train_dataset.zip' for 'unzip' command: including all the training images and COCO format annotation file. 
(https://drive.google.com/file/d/1oCu-06eYpDE7q2k2gMHWCiO00Ex2aOnw/view?usp=sharing)
- 'mask_rcnn_r101_fpn_config.py': config file for mmdetection.
- The data path and pre-trained model path also should be set properly in the config file ('data' block and 'load_from').

# Inference

Run 'inference.ipynb' on Colab.

These files should be specified a path. You can set the path on your own:
- 'dataset.zip' for 'unzip' command: including the testing images (downloaded from CodaLab).
- 'test_img_ids.json': included in 'dataset.zip'.
- 'mask_rcnn_r101_fpn_config.py': config file for mmdetection.
- 'mask_rcnn_2436.pth': the checkpoint file.
- 'answer.json': generated through this notbook, the answer file to be submitted.