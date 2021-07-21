import os
import torch
import torch.distributions as dist
import torch.optim as optim
import torch.nn as nn
from pipelines import config
from pipelines.lig import models, generation
from pipelines.lig.models import LocalImplicitGrid
from pipelines.lig.models.layers import GridInterpolationLayer
from pipelines.lig.models.optimizer import LIGOptimizer
import ipdb


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Local Implicit Grid Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    if encoder is not None:
        encoder = models.encoder_dict[encoder](**encoder_kwargs).to(device)
    decoder = models.decoder_dict[decoder](**decoder_kwargs).to(device)
    grid_interp_layer = GridInterpolationLayer()
    method = 'linear' if cfg['model']['overlap'] else 'nn'
    x_location_max = 1.0 if cfg['model']['overlap'] else 2.0
    interp = not cfg['generation']['indep_pt_loss']

    model = LocalImplicitGrid(encoder, decoder, grid_interp_layer,
                              method, x_location_max, interp, device)

    return model


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    # Optimizer parameters
    optimizer_kwargs = cfg['generation']['optimizer_kwargs']
    latent_size = optimizer_kwargs['latent_size']
    alpha_lat = optimizer_kwargs['alpha_lat']
    num_optim_samples = optimizer_kwargs['num_optim_samples']
    init_std = optimizer_kwargs['init_std']
    learning_rate = optimizer_kwargs['learning_rate']
    optim_steps = optimizer_kwargs['optim_steps']
    print_every_n_steps = optimizer_kwargs['print_every_n_steps']
    indep_pt_loss = cfg['generation']['indep_pt_loss']

    optimizer = LIGOptimizer(model, latent_size=latent_size, alpha_lat=alpha_lat,
                             num_optim_samples=num_optim_samples, init_std=init_std,
                             learning_rate=learning_rate, optim_steps=optim_steps,
                             print_every_n_steps=print_every_n_steps,
                             indep_pt_loss=indep_pt_loss, device=device)

    # Generator parameters
    part_size = cfg['model']['part_size']
    res_per_part = cfg['model']['res_per_part']
    overlap = cfg['model']['overlap']
    points_batch = cfg['generation']['points_batch']
    conservative = cfg['generation']['conservative']
    postprocess = cfg['generation']['postprocess']

    generator = generation.Generator3D(
        model, optimizer, part_size=part_size, num_optim_samples=num_optim_samples,
        res_per_part=res_per_part, overlap=overlap, device=device,
        points_batch=points_batch, conservative=conservative,
        postprocess=postprocess
    )
    return generator
