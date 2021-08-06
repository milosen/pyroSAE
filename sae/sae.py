from itertools import product as cartesian
import torchio as tio
import logging
import pickle

import torch
import pyro
from pyro.infer import Trace_ELBO
import pyro.distributions as dist
from tqdm import tqdm as progressbar
import numpy as np
from unet import UNet3D
from torch.nn.functional import conv3d, pad
import torch.nn.functional as F

from sae.cnn import CNN
from sae.util import *



plt.rcParams["savefig.bbox"] = "tight"


def stack_batch(batch, label):
    return torch.stack([patch[label][tio.DATA] for patch in batch], dim=0)


def prepare_batch(batch, device, use_labels=False):
    x1 = stack_batch(batch, 't1').to(device)
    x2 = stack_batch(batch, 'pd').to(device)
    x = torch.cat([x1, x2], dim=1)
    #x = x1
    a = stack_batch(batch, 'atlas').to(device)

    if use_labels:
        y = stack_batch(batch, 'label').to(device)
    else:
        y = torch.empty((0,))
    return x, y, a


def prepare_slice(im, patch_slice):
    return np.rot90(im[:, :, patch_slice].cpu().numpy())


def plot_test(s, xr, x, target, a, patch_slice=24):

    _, axs = plt.subplots(2, 5)

    axs[0, 0].imshow(prepare_slice(xr[0, 0, :, :, :], patch_slice), cmap='gray')
    axs[1, 0].imshow(prepare_slice(x[0, 0, :, :, :], patch_slice), cmap='gray')

    if x.shape[1] == 2:
        axs[0, 1].imshow(prepare_slice(xr[0, 1, :, :, :], patch_slice), cmap='gray')
        axs[1, 1].imshow(prepare_slice(x[0, 1, :, :, :], patch_slice), cmap='gray')

    y_onehot = onehot(target)
    for i in range(1, 4):
        axs[0, i+1].imshow(prepare_slice(s[0, i, :, :, :], patch_slice), cmap='gray')
        # axs[1, i+1].imshow(prepare_slice(y_onehot[0, i, :, :, :], patch_slice), cmap='gray')
        axs[1, i+1].imshow(prepare_slice(a[0, i, :, :, :], patch_slice), cmap='gray')

    for i, j in cartesian(range(2), range(5)):
        axs[i, j].axis('off')

    axs[0, 0].set_title("Recon T1w")
    axs[0, 1].set_title("Recon PDw")
    axs[0, 2].set_title("Pred CSF")
    axs[0, 3].set_title("Pred GM")
    axs[0, 4].set_title("Pred WM")
    axs[1, 0].set_title("Input T1w")
    axs[1, 1].set_title("Input PDw")
    axs[1, 2].set_title("Atlas CSF")
    axs[1, 3].set_title("Atlas GM")
    axs[1, 4].set_title("Atlas WM")


def to_torch(tensor):
    return tensor.permute(0, 4, 1, 2, 3)


def to_pyro(tensor):
    return tensor.permute(0, 2, 3, 4, 1)


class SAE(torch.nn.Module):
    def __init__(self, prior, config, device="cuda:0"):
        super().__init__()

        self.device = device
        self.config = config
        prior_pt = co_occ(prior)
        self.prior = prior_pt.to(self.device)

        self.save_freq = self.config['eval']['save_freq']
        self.val_freq = self.config['eval']['val_freq']

        self.n_classes = self.config['arch']['n_tissue_channels']
        self.img_channels = self.config['arch']['n_img_channels']
        self.guide_dist_y = dist.RelaxedOneHotCategoricalStraightThrough
        self.temp = self.config['arch']['gumbel_softmax_temp']
        self.encoder = UNet3D(in_channels=self.img_channels, out_classes=self.n_classes, normalization='batch',
                              residual=True, padding=1, upsampling_type='linear')

        self.decoder = CNN(in_channels=self.n_classes, out_channels=1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.config['train']['lr'])
        self.beta = self.config['train']['beta']

        self.loss_fn_elbo = Trace_ELBO().differentiable_loss

        self.summary = dict(
            epoch=[],
            loss_recon=[],
            loss_kl_cat=[],
            loss_kl_mrf=[],
            loss_elbo=[],
            dice=[]
        )

        self.to(self.device)

    def neighboor_q(self, input, neighboor_size):
        '''
        Author: Evan Yu
        Calculate the product of all q(s|x) around voxel i
        Uses convolution to sum the log(q_y) then takes the exp

        input: prob of q
        '''

        k = neighboor_size

        x = pad(input, (1, 1, 1, 1, 1, 1))

        chs = x.shape[1]

        filter = torch.ones(k, k, k).view(1, 1, k, k, k)
        filter[:, :, k // 2, k // 2, k // 2] = 0
        filter = filter.repeat(chs, 1, 1, 1, 1).float().to(self.device)
        filter.requires_grad = False

        out = F.conv3d(x,
                       weight=filter,
                       stride=1,
                       groups=chs)
        return out

    def spatial_consistency(self, input, table, neighboor_size):
        '''
        Author: Evan Yu
        KL divergence between q(s|x) and markov random field

        input: prob of q
        table: lookup table as probability. Rows add up to 1
        '''
        n_batch, chs, dim1, dim2, dim3 = input.shape
        q_y = self.neighboor_q(input, neighboor_size)

        m = table / torch.sum(table, 1, True)  # Normalize to account for the extra eps
        m = m.view(1, chs, chs)

        # Multiplication
        q_i = input.view(n_batch, chs, dim1 * dim2 * dim3)
        q_y = q_y.view(n_batch, chs, dim1 * dim2 * dim3)
        out = torch.bmm(m, q_y)  # shape [n_batch, chs, dim1*dim2*dim3]
        out = torch.sum(q_i * out, 1)
        return -1 * torch.sum(out)

    def temp_like(self, x):
        return x.new_ones([1])*self.temp

    def model(self, x, y, a):
        pyro.module("vae", self)
        with pyro.poutine.scale(scale=self.beta):
            cat = self.guide_dist_y(self.temp_like(x), probs=to_pyro(a))
            y = to_torch(pyro.sample("y", cat))

        with pyro.poutine.scale(scale=1):
            rec_t1 = self.decoder(y)
            dist_t1 = dist.Normal(rec_t1, rec_t1.new_ones(rec_t1.shape))
            pyro.sample("obs_t1", dist_t1, obs=x)

    def guide(self, x, y, a):
        seg = torch.softmax(self.encoder(x), dim=1)
        cat = self.guide_dist_y(temperature=self.temp_like(x), probs=to_pyro(seg))
        pyro.sample("y", cat)

    def reconstruct_img(self, x):
        seg = torch.softmax(self.encoder(x), dim=1)
        cat_dist = self.guide_dist_y(temperature=self.temp_like(x), probs=to_pyro(seg))
        y = cat_dist.sample().permute(0, 4, 1, 2, 3)
        img_t1 = self.decoder(y)
        return seg, img_t1

    def train_epoch(self, dataloader):
        self.train(True)
        pyro.clear_param_store()
        for batch in progressbar(dataloader):
            x, y, a = prepare_batch(batch, self.device)
            loss_elbo = self.loss_fn_elbo(self.model, self.guide, x, y, a)
            loss = loss_elbo + self.spatial_consistency(torch.softmax(self.encoder(x), dim=1), self.mrf_prior, 3)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def evaluate(self, loader, patch_idx=55):
        print("Evaluate...")
        self.eval()
        model_kl_cat = pyro.poutine.block(self.model, hide=["obs_t1", "obs_pd"])
        for e, patches_batch in enumerate(loader):
            if e != patch_idx:
                continue
            x = patches_batch['t1'][tio.DATA].to(self.device)
            tar = patches_batch['label'][tio.DATA].to(self.device)
            a = patches_batch['atlas'][tio.DATA].to(self.device)
            loss_elbo = self.loss_fn_elbo(self.model, self.guide, x, tar, a).item()
            loss_kl_cat = self.loss_fn_elbo(model_kl_cat, self.guide, x, tar, a).item()
            self.summary['loss_kl_cat'].append(loss_kl_cat)
            self.summary['loss_elbo'].append(loss_elbo)
            self.summary['loss_recon'].append(loss_elbo - loss_kl_cat)
            self.summary['loss_kl_mrf'].append(self.spatial_consistency(torch.softmax(self.encoder(x), dim=1), self.mrf_prior, 3))

            s, xr = self.reconstruct_img(x)
            dice_s = s.squeeze().permute(3, 0, 1, 2)
            dice_t = tar.long().squeeze().permute(2, 0, 1)
            self.summary['dice'].append(dice(dice_s, dice_t))

            return s, xr, x, tar, a

    def train_epochs(self, train_loader, val_loader, start_epoch=0):
        for epoch_idx in range(start_epoch, start_epoch + self.config['train']['n_epochs']):
            self.train_epoch(train_loader)
            with torch.no_grad():
                s, xr, x, tar, a = self.evaluate(val_loader)
                self.summary['epoch'].append(epoch_idx)
                if self.save_freq != 0 and epoch_idx % self.save_freq == 0:
                    plot_test(s, xr, x, tar, a)
                    plt.show()
                    torch.save((self.state_dict(), self.optimizer.state_dict(), self.summary), f"checkpoints_test/{self.device[-1]}_vae_epoch_{epoch_idx}.pt")
