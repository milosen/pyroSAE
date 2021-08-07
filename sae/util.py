import matplotlib.pyplot as plt
import torchio as tio
import torch
from torch.nn.functional import pad, conv3d
import torch.nn.functional as F
from itertools import product
from kornia import one_hot
import numpy as np
from tqdm import tqdm


class Slicer:
    def __init__(self, dimension=0, dim_slice=200):
        self.dimension = torch.tensor(dimension)
        self.dim_slice = torch.tensor(dim_slice)

    def show(self, im: torch.Tensor, ax=None):
        # noinspection PyTypeChecker
        s = torch.index_select(im.cpu(), dim=self.dimension, index=self.dim_slice).squeeze()
        if ax:
            ax.imshow(np.rot90(s.numpy()), cmap='gray')
            ax.axis('off')
        else:
            plt.imshow(np.rot90(s.numpy()), cmap='gray')
            plt.axis('off')
            plt.show()


def get_datasets():
    atlas = tio.datasets.ICBM2009CNonlinearSymmetric(load_4d_tissues=True)
    tissues_tensor = atlas['tissues'][tio.DATA].permute(1, 2, 3, 0)
    atlas['tissues'][tio.DATA] = torch.cat([1 - tissues_tensor.sum(dim=0, keepdim=True), tissues_tensor], dim=0)

    transforms = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(atlas.t1.path),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ])

    ixi_dataset = tio.datasets.ixi.IXITiny(
        'data',
        transform=transforms,
        download=True
    )

    print("Prepare Subjects")
    all_subjects = ixi_dataset.dry_iter()
    for subject in all_subjects:
        # this makes sure that the patch locations are always the same for
        # input images and atlas (there is probably a better way to do it but idk)
        subject.add_image(atlas['tissues'], 'atlas')

    training_dataset = tio.SubjectsDataset(all_subjects[:8], transform=transforms)
    val_dataset = tio.SubjectsDataset(all_subjects[8:10], transform=transforms)

    return training_dataset, val_dataset, atlas


def co_occ(atlas: torch.tensor, device) -> torch.tensor:
    r"""Calculate log co-occurrence matrix for the Markov random field prior.
    """
    d = atlas.shape[0]
    h = atlas.shape[1]
    w = atlas.shape[2]
    n_classes = atlas.shape[3]

    max_idx = torch.argmax(atlas, 3, keepdim=True).to(device)
    t = torch.zeros_like(atlas).cuda()

    potentials = t.new_zeros([n_classes, n_classes]).to(device)

    l1_l2_counts = t.new_zeros([1, n_classes, n_classes, d, h, w]).to(device)

    nh = torch.tensor([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]).to(device).float()

    nh = nh.view(1, 1, 3, 3, 3)

    t.scatter_(3, max_idx, 1)

    t = t.permute(3, 0, 1, 2).unsqueeze(0)
    tp = pad(t, (1, 1, 1, 1, 1, 1))

    for l1, l2 in product(range(n_classes), range(n_classes)):

        l1 = torch.tensor([l1]).cuda()
        l2 = torch.tensor([l2]).cuda()

        l1_bin_map = tp.index_select(dim=1, index=l1)
        l2_bin_map = t.index_select(dim=1, index=l2)

        nh_counts = conv3d(l1_bin_map, nh)
        l2_total = t.index_select(dim=1, index=l2).sum(dim=(-1, -2, -3))

        l1_l2_counts[:, l1.item(), l2.item(), :, :, :] = (l2_bin_map*nh_counts).squeeze(dim=1)

        # divide by 6 to counter the 6 fold counting
        potentials[l1, l2] = torch.where(
            l2_total > 0,
            l1_l2_counts[0, l1, l2, :, :, :].sum(dim=(-1, -2, -3)) / 6. / l2_total,
            t.new_zeros([1])
        )

    potentials[potentials == 0] = 1e-12
    log_rel_cooc = torch.log(potentials)

    return log_rel_cooc


def dice(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Computes Sørensen-Dice Coefficient.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, target.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (0, 2, 3)
    intersection = torch.sum(input * target_one_hot, dims)
    cardinality = torch.sum(input + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)

    return dice_score


def onehot(tensor: torch.Tensor, num_classes=4):
    return F.one_hot(tensor.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)


def stack_batch(batch, label):
    return torch.stack([patch[label][tio.DATA] for patch in batch], dim=0)


def prepare_batch(atlas, batch, device, patch_size, get_labels=False):
    x = stack_batch(batch, 'image').to(device)

    if get_labels:
        y = stack_batch(batch, 'label').to(device)
    else:
        y = torch.empty((0,))

    a = stack_batch(batch, 'atlas').to(device)

    return x, y, a


def to_torch(tensor):
    return tensor.permute(0, 4, 1, 2, 3)


def to_pyro(tensor):
    return tensor.permute(0, 2, 3, 4, 1)


def ensure_checkpoints_dir_exists():
    import os
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
