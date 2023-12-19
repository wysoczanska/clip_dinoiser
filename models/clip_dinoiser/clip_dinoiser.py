# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology & Oriane Simeoni, valeo.ai
# ---------------------------------------------------------------------------------------------------
import torch.nn as nn
from models.builder import MODELS
from models.builder import build_model
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
import torch.nn.functional as F

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@MODELS.register_module()
class CLIP_DINOiser(nn.Module):
    def __init__(self, clip_backbone, class_names, mask_th=None, found_th=0.5, certainty_th=0.9, apply_found=False,
                 in_dim=256, conv_kernel=3, feats_idx=-3):

        super(CLIP_DINOiser, self).__init__()
        self.mask_th = mask_th
        self.apply_found = apply_found
        self.found_th = found_th
        self.certainty_th = certainty_th
        self.sigmoid = nn.Sigmoid()
        maskclip_cfg = OmegaConf.load(f"configs/{clip_backbone}.yaml")
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)
        self.vit_patch_size = self.clip_backbone.patch_size
        self.feats_idx = feats_idx
        self.in_dim = [in_dim]
        in_size = 768 if self.feats_idx != 'final' else 512
        self.bkg_decoder = nn.Conv2d(in_size, 1, (1, 1))
        self.obj_proj = nn.Conv2d(in_size, in_dim, (conv_kernel, conv_kernel),
                                      padding=conv_kernel // 2, padding_mode='replicate')

        # setup clip feature for training
        if feats_idx != 'final':
            train_feats = {}
            def get_activation(name):
                def hook(model, input, output):
                    train_feats[name] = output.detach()
                return hook
            self.clip_backbone.backbone.layers[feats_idx].ln2.register_forward_hook(get_activation('clip_inter'))
            self.train_feats = train_feats


    def forward_pass(self, x):
        clip_feats = self.get_clip_map(x)[0]
        B, c_dim, h, w = clip_feats.shape
        _, _, H, W = x.shape
        if self.feats_idx != 'final':
            clip_feats = self.train_feats['clip_inter']
            c_dim = clip_feats.shape[-1]
            clip_feats = clip_feats[:, 1:, ].permute(0, 2, 1).reshape(B, c_dim, h, w)

        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim[-1], -1)
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)

        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(B,h*w, h, w)
        output = clip_feats / clip_feats.norm(dim=1, keepdim=True)
        output = self.bkg_decoder(output)

        return output, corrs

    def forward(self, x):
        preds, corrs = self.forward_pass(x)
        output, _, _ = self.get_clip_map(x)
        B, C, hf, wf = output.shape
        preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False )

        # Compute weighted pooling
        if self.mask_th:
             corrs[corrs < self.mask_th] = 0.0
        output = self.compute_weighted_pool(output, corrs)
        output = output.reshape(B, C, hf, wf)
        output = self.clip_backbone.decode_head.cls_seg(output)

        if self.apply_found:
            # Compute FOUND --------------------------------------------------
            soft_found = self.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > self.found_th).float()

            # TODO: make it work for Batch Size != 1
            uncertain = (output.max(dim=1)[0] < self.certainty_th).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class

        return output

    def predict(self, x):
        return self(x)

    @torch.no_grad()
    def get_clip_map(self, img):
        maskclip_map, feat, k = self.clip_backbone(img, return_feat=True)

        return feat, k, maskclip_map

    @torch.no_grad()
    def compute_weighted_pool(self, clipmap, corrs):
        # upsampling
        B = clipmap.shape[0]
        h_m, w_m = clipmap.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            clipmap = F.interpolate(clipmap, (h_w, w_w), mode="bilinear", align_corners=False )
            h_m, w_m = h_w, w_w

        corrs[corrs < 0.0] = 0.0   # B HW H W
        clipmap_refined = torch.einsum("bnij, bcij -> bcn", corrs, clipmap)  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None] # B 1 HW
        clipmap_refined = clipmap_refined / (norm_factor + 1e-6)

        # RESHAPE back to 2d
        clipmap_refined = clipmap_refined.reshape(B, -1, h_m, w_m)

        return clipmap_refined
