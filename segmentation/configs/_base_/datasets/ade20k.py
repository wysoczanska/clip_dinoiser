_base_ = ["../custom_import.py"]
dataset_type = "ADE20KDataset"
data_root = "./data"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='ToRGB'),
            dict(type='Resize', img_scale=(2048, 448)),
            dict(type='RandomCrop', crop_size=(448, 448)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type='Collect', keys=['img'], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]

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
            dict(type="Collect", keys=["img"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="ADEChallengeData2016/images/training",
        ann_dir="ADEChallengeData2016/annotations/training",
        pipeline=train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="ADEChallengeData2016/images/validation",
        ann_dir="ADEChallengeData2016/annotations/validation",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
