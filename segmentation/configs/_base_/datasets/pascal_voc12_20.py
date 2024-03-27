_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "PascalVOCDataset20"
data_root = "./data/VOCdevkit/VOC2012"
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img"],
                 meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        pipeline=test_pipeline,

    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
