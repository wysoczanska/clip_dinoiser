_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "PascalContextDataset59"
data_root = "./data/VOCdevkit/VOC2010"
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
            # dict(type='Normalize', **img_norm_cfg),
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
        ann_dir="SegmentationClassContext",
        split="ImageSets/SegmentationContext/val.txt",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
# test_cfg = dict(mode="whole")
