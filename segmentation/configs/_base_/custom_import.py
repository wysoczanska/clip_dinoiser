# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ----------------------------------------------------------------------------------------------------
# GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
custom_imports = dict(
    imports=["segmentation.datasets.coco_object", "segmentation.datasets.pascal_voc",
             "segmentation.datasets.pascal_voc20",
             ],
    allow_failed_imports=False,
)
