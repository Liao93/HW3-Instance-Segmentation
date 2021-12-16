# The new config inherits a base config to highlight the necessary modification
_base_ = '/content/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
  backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
  roi_head=dict(
      bbox_head=dict(num_classes=1),
      mask_head=dict(num_classes=1)),
  rpn_head=dict(
      anchor_generator=dict(scales=[2,4])), # default is [8] # Reduce RPN anchor sizes
  test_cfg=dict(
      rcnn=dict(max_per_img=500, mask_thr_binary=0.35)), # Increase number of objects detected # Reduce object detection probability
)

albu_train_transforms = [
    dict(type='VerticalFlip', p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='RandomRotate90', p=0.5),
    dict(type='RandomCrop', height=360, width=360, p=1.0),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(720, 720), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])  
]

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nuclear',)
data = dict(
  train=dict(
      img_prefix='/content/train_dataset',
      classes=classes,
      ann_file='/content/train_dataset/nuclear_coco_format_rleMask_rleArea_1class.json',
      pipeline=train_pipeline),
  val=dict(
      img_prefix='/content/train_dataset',
      classes=classes,
      ann_file='/content/train_dataset/nuclear_coco_format_rleMask_rleArea_1class.json',
      pipeline=test_pipeline),
  test=dict(
      img_prefix='/content/train_dataset',
      classes=classes,
      ann_file='/content/train_dataset/nuclear_coco_format_rleMask_rleArea_1class.json',
      pipeline=test_pipeline))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth'
