# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ----------------------------------------------------------------------------------------------------
# Modified from TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------

custom_imports = dict(
    imports=["segmentation.datasets.coco_object", "segmentation.datasets.pascal_voc", "datasets.transforms", "segmentation.datasets.pascal_voc20"],
    allow_failed_imports=False,
)
