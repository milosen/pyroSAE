import torchio as tio
from sae.util import Slicer, get_datasets
import torch
from torch.utils.data import DataLoader
from sae.sae import SAE
import yaml


if __name__ == '__main__':
    slicer = Slicer(dimension=2, dim_slice=100)

    config = yaml.safe_load(open("config.yml"))

    train_dataset, atlas_dataset = get_datasets()

    train_queue_dataset = tio.Queue(
        train_dataset,
        config['patches']['queue_length'],
        config['patches']['patches_per_volume'],
        tio.data.UniformSampler(patch_size=config['patches']['patch_size']),
        shuffle_patches=True,
        shuffle_subjects=True,
        num_workers=7
    )

    train_loader = DataLoader(
        train_queue_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=lambda x: x,
        shuffle=True
    )

    sae = SAE(atlas_dataset['tissues'][tio.DATA].permute(0, 2, 3, 1), config, "cuda")

    val_sampler = tio.data.GridSampler(train_dataset[0], config['patches']['patch_size'], patch_overlap=config['patches']['overlap'])
    val_loader = torch.utils.data.DataLoader(val_sampler, batch_size=1)
    sae.train_epochs(train_loader=train_loader, val_loader=val_loader)
