### modified code from MaskCLIP : https://github.com/chongzhou96/MaskCLIP/blob/master/tools/maskclip_utils/
import os

import torch
import argparse
from open_clip import create_model_and_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save the CLIP visual weights')
    parser.add_argument('--model', default='ViT-16-laion', choices=['ViT32', 'ViT16', 'ViT14', 'ViT-16-laion'], help='clip model name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    name_mapping = {'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16', 'ViT14': 'ViT-L/14', 'ViT-16-laion': 'ViT-B-16'}
    pretrained = 'laion2B-s34B-b88K'
    model, _, _ = create_model_and_transforms(name_mapping[args.model], pretrained=pretrained)

    state_dict = model.state_dict()

    result_model = {'meta': {}, 'state_dict': {}}
    all_model = dict()
    stem_mapping = {'conv1': 0, 'bn1': 1, 'conv2': 3, 'bn2': 4, 'conv3': 6, 'bn3':7}
    clip_keys = []
    prefix = 'visual'
    for key in state_dict.keys():
        if 'ViT' in args.model and prefix in key:
            new_key = key[len(f'{prefix}.'):]
            if new_key == 'proj':
                all_model['proj'] = {}
                all_model['proj']['weight'] = state_dict[key].float().t()
                continue
            if new_key == 'class_embedding':
                new_key = 'cls_token'
                state_dict[key] = state_dict[key][None, None, :]
            elif new_key == 'positional_embedding':
                new_key = 'pos_embed'
                state_dict[key] = state_dict[key][None, :, :]
            elif new_key == 'conv1.weight':
                new_key = 'patch_embed.projection.weight'
            elif 'ln_pre' in new_key:
                weight_or_bias = new_key.split('.')[-1]
                new_key = f'ln0.{weight_or_bias}'
            elif 'ln_post' in new_key:
                weight_or_bias = new_key.split('.')[-1]
                new_key = f'ln1.{weight_or_bias}'
            elif 'transformer' in new_key:
                new_key = 'layers.' + new_key[len('transformer.resblocks.'):]
                if 'mlp' in new_key:
                    new_key = new_key.replace('mlp', 'ffn.layers')
                if 'c_fc' in new_key:
                    new_key = new_key.replace('c_fc', '0.0')
                if 'c_proj' in new_key:
                    new_key = new_key.replace('c_proj', '1')
                if 'attn' in new_key:
                    new_key = new_key.replace('attn', 'attn.attn')
                elif 'ln_' in new_key:
                    new_key = new_key.replace('ln_', 'ln')
            clip_keys.append(new_key)
            result_model['state_dict'].update({new_key: state_dict[key].float()})
        elif prefix in key:
            if 'attnpool' in key:
                if 'proj' in key:
                    proj_name = key.split('.')[2]
                    weight_or_bias = key.split('.')[3]
                    if proj_name not in all_model:
                        all_model[proj_name] = {}
                    all_model[proj_name][weight_or_bias] = state_dict[key].float()
            else:
                new_key = key[len(f'{prefix}.'):]
                if 'layer' not in new_key:
                    print(new_key.split('.'))
                    layer_name, layer_type = new_key.split('.')
                    new_key = 'stem.{}.{}'.format(stem_mapping[layer_name], layer_type)
                if 'downsample' in new_key:
                    splits = new_key.split('.')
                    new_key = '{}.{}.{}.{}.{}'.format(splits[0], splits[1], splits[2],
                        int(splits[3])+1, splits[4])
                clip_keys.append(new_key)
                result_model['state_dict'].update({new_key: state_dict[key].float()})

    if not os.path.exists('pretrain'):
        os.mkdir('pretrain')

    torch.save(result_model, f'checkpoints/{args.model}_clip_backbone.pth')
    torch.save(all_model, f'checkpoints/{args.model}_clip_proj.pth')