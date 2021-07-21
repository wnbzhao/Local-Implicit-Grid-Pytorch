import torch
import os
import argparse
import numpy as np
import sys
sys.path.append('./')
from pipelines import config
from pipelines.utils.point_utils import read_point_ply

parser = argparse.ArgumentParser(description='Extract meshes from occupancy process.')
parser.add_argument('--config', default='configs/lig/lig_pretrained.yaml', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--input_ply', type=str, help='Input object file')
parser.add_argument('--output_ply', type=str, help='Output object file')
parser.add_argument('--gen', action='store_true', help='to generate mesh, no training')
parser.add_argument('--continue_training', action='store_true', help='whether to continue training')
parser.add_argument('--model', type=str, default='pretrained_models/lig/model_best.pt', help='pretrained model path')
parser.add_argument('--debug', action='store_true', help='whether it is debug mode')
parser.add_argument('--normalized', action='store_true', help='whether normalize the input')
args = parser.parse_args()
print(str(args))

cfg = config.load_config(args.config, 'configs/default.yaml')
assert not np.logical_and(args.gen, args.continue_training), "Cannot be generation mode and training mode at the same time"
if args.gen:
    assert args.model != '', "Pretrained model path shouldn't be empty in generation mode"
    cfg['generation']['optimizer_kwargs']['optim_steps'] = 0
if args.continue_training:
    assert args.model != '', "Pretrained model path shouldn't be empty in continue training mode"
cfg['generation']['optimizer_kwargs']['continue_training'] = args.continue_training
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Fix seed of numpy and torch to make results reproducable
np.random.seed(0)
torch.manual_seed(0)

# Model
model = config.get_model(cfg, device=device)

print('!!! Model Loaded !!! ')
out_dir = cfg['generation']['out_dir']

# Initialize generation directory
file_path, file_name = os.path.split(args.input_ply)
obj_name, ext = os.path.splitext(file_name)
generation_dir = os.path.join(out_dir, obj_name + '_debug' if args.debug else obj_name)
if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)

output_path = './'
# Set pretrained path and output path of the model
model_path = os.path.join(output_path, 'model')
cfg['model']['model_path'] = model_path
cfg['model']['pretrained_path'] = args.model

# Load pretrained weight
if args.model != '':
    pretrained_dict = torch.load(args.model)
    model_dict = model.state_dict()
    update_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    print(f"Total {len(model_dict)} parameters, updated {len(update_dict)} parameters")

# Generator
generator = config.get_generator(model, cfg, device=device, output_path=output_path)

v, n = read_point_ply(args.input_ply)
v = v.astype(np.float32)
n = n.astype(np.float32)

# Normalize to unit sphere
if args.normalized:
    v = v - v.mean(axis=0)
    v = v / (np.linalg.norm(v, ord=2, axis=1).max() + 1e-12)

mesh = generator.generate_single_obj_mesh(v, n)

# Write output
mesh.export(args.output_ply)