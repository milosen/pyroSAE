import pyro
from pyro.infer import Trace_ELBO
import pyro.distributions as dist
from tqdm import tqdm
from unet import UNet3D

from sae.cnn import CNN
from sae.util import *


plt.rcParams["savefig.bbox"] = "tight"


class SAE(torch.nn.Module):
    def __init__(self, atlas, config, device="cuda:0"):
        super().__init__()

        self.device = device
        self.config = config
        self.atlas = atlas
        self.pairwise_potential = co_occ(atlas.permute(1, 2, 3, 0), self.device)

        self.guide_dist_y = dist.RelaxedOneHotCategoricalStraightThrough
        self.encoder = UNet3D(in_channels=self.config['arch']['n_img_channels'],
                              out_classes=self.config['arch']['n_tissue_channels'],
                              normalization='batch', residual=True, padding=1, upsampling_type='linear')

        self.decoder_t1 = CNN(in_channels=self.config['arch']['n_tissue_channels'], out_channels=1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.config['train']['lr'])

        self.loss_fn_elbo = Trace_ELBO().differentiable_loss

        self.summary = dict(
            epoch=[],
            loss_recon=[],
            loss_kl_cat=[],
            loss_kl_mrf=[],
            loss_elbo=[],
            dice=[]
        )

        self.slicer = Slicer(dimension=2, dim_slice=100)

        self.to(self.device)

    def neighboor_q(self, input, neighboor_size):
        """
        Author: Evan Yu

        Calculate the product of all q(s|x) around voxel i
        Uses convolution to sum the log(q_y) then takes the exp
        input: prob of q
        """

        k = neighboor_size

        x = pad(input, (1, 1, 1, 1, 1, 1))

        chs = x.shape[1]

        conv_filter = torch.ones(k, k, k).view(1, 1, k, k, k)
        conv_filter[:, :, k // 2, k // 2, k // 2] = 0
        conv_filter = conv_filter.repeat(chs, 1, 1, 1, 1).float().to(self.device)
        conv_filter.requires_grad = False

        out = F.conv3d(x, weight=conv_filter, stride=1, groups=chs)
        return out

    def spatial_consistency(self, input, table, neighboor_size):
        """
        Author: Evan Yu

        KL divergence between q(s|x) and markov random field
        input: prob of q
        table: lookup table as probability. Rows add up to 1
        """
        n_batch, chs, dim1, dim2, dim3 = input.shape
        q_y = self.neighboor_q(input, neighboor_size)

        m = table / torch.sum(table, 1, True)  # Normalize to account for the extra eps
        m = m.view(1, chs, chs)

        # Multiplication
        q_i = input.view(n_batch, chs, dim1 * dim2 * dim3)
        q_y = q_y.view(n_batch, chs, dim1 * dim2 * dim3)
        out = torch.bmm(m, q_y)
        out = torch.sum(q_i * out, 1)
        return -1 * torch.sum(out)

    def temp_like(self, x):
        # we need a temperature tensor that is on the same device as x because
        # pyro doesn't do it automatically
        return x.new_ones([1])*self.config['arch']['gumbel_softmax_temp']

    def model(self, x, y, a):
        # probabilistic decoder of the VAE in pyro framework
        pyro.module("vae", self)
        with pyro.poutine.scale(scale=self.config['train']['beta']):
            cat = self.guide_dist_y(self.temp_like(x), probs=to_pyro(a))
            y = to_torch(pyro.sample("y", cat))

        with pyro.poutine.scale(scale=1):
            rec_t1 = self.decoder_t1(y)
            dist_t1 = dist.Normal(rec_t1, rec_t1.new_ones(rec_t1.shape))
            pyro.sample("obs_t1", dist_t1, obs=x[:, 0])

    def guide(self, x, y, a):
        # probabilistic encoder of the VAE in pyro framework
        seg = torch.softmax(self.encoder(x), dim=1)
        cat = self.guide_dist_y(temperature=self.temp_like(x), probs=to_pyro(seg))
        pyro.sample("y", cat)

    def reconstruct_img(self, x):
        seg = torch.softmax(self.encoder(x), dim=1)
        cat_dist = self.guide_dist_y(temperature=self.temp_like(x), probs=to_pyro(seg))
        y = to_torch(cat_dist.sample())
        img_t1 = self.decoder_t1(y)
        return seg, img_t1

    def train_epoch(self, dataloader):
        self.train(True)
        pyro.clear_param_store()
        for batch in tqdm(dataloader):
            x, y, a = prepare_batch(self.atlas, batch, self.device, self.config['patches']['patch_size'])
            loss_elbo = self.loss_fn_elbo(self.model, self.guide, x, y, a)
            loss = loss_elbo + self.spatial_consistency(
                torch.softmax(self.encoder(x), dim=1), self.pairwise_potential, 3)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def infer(self, grid_loader, grid_aggr):
        self.eval()
        from copy import deepcopy
        aggr_seg = deepcopy(grid_aggr)
        aggr_xr = deepcopy(aggr_seg)
        aggr_x = deepcopy(aggr_seg)
        aggr_tar = deepcopy(aggr_seg)
        aggr_a = deepcopy(aggr_seg)
        model_kl_cat = pyro.poutine.block(self.model, hide=["obs_t1", "obs_pd"])
        num_patches, loss_kl_cat, loss_elbo, loss_kl_mrf, dsc = len(grid_loader), 0, 0, 0, 0
        for patches_batch in tqdm(grid_loader):

            x = patches_batch['image'][tio.DATA].to(self.device)
            tar = patches_batch['label'][tio.DATA].to(self.device)
            a = patches_batch['atlas'][tio.DATA].to(self.device)

            loss_elbo += self.loss_fn_elbo(self.model, self.guide, x, tar, a).item() / num_patches
            loss_kl_cat += self.loss_fn_elbo(model_kl_cat, self.guide, x, tar, a).item() / num_patches
            loss_kl_mrf += self.spatial_consistency(torch.softmax(self.encoder(x), dim=1), self.pairwise_potential, 3) / num_patches

            s, xr = self.reconstruct_img(x)
            dice_s = s.squeeze().permute(3, 0, 1, 2)
            dice_t = tar.long().squeeze().permute(2, 0, 1)
            dsc += dice(dice_s, dice_t) / num_patches

            loc = patches_batch[tio.LOCATION]
            aggr_seg.add_batch(s, loc)
            aggr_xr.add_batch(xr, loc)
            aggr_x.add_batch(x, loc)
            aggr_tar.add_batch(tar, loc)
            aggr_a.add_batch(a, loc)

        self.summary['loss_kl_cat'].append(loss_kl_cat)
        self.summary['loss_elbo'].append(loss_elbo)
        self.summary['loss_recon'].append(loss_elbo - loss_kl_cat)
        self.summary['loss_kl_mrf'].append(loss_kl_mrf)
        self.summary['dice'].append(dsc)

        return \
            aggr_seg.get_output_tensor().unsqueeze(0), \
            aggr_xr.get_output_tensor().unsqueeze(0), \
            aggr_x.get_output_tensor().unsqueeze(0), \
            aggr_tar.get_output_tensor().unsqueeze(0), \
            aggr_a.get_output_tensor().unsqueeze(0)

    def train_epochs(self, train_loader, val_loader, val_aggr, start_epoch=0):
        ensure_checkpoints_dir_exists()

        for epoch_idx in range(start_epoch, start_epoch + self.config['train']['n_epochs']):
            self.train_epoch(train_loader)
            with torch.no_grad():
                print('Perform inference on Validation Subject...')
                s, xr, x, tar, a = self.infer(val_loader, val_aggr)
                self.summary['epoch'].append(epoch_idx)
                if self.config['eval']['save_freq'] != 0 and epoch_idx % self.config['eval']['save_freq'] == 0:
                    self.plot_test(s, xr, x, tar, a)
                    torch.save((self.state_dict(), self.optimizer.state_dict(), self.summary),
                               f"checkpoints/vae_epoch_{epoch_idx}_{self.device}.pt")
                    torch.save((s, xr, x, tar, a),
                               f"checkpoints/last_tensors.pt")

    def plot_test(self, s, xr, x, target, a):

        _, axs = plt.subplots(2, 5)
        self.slicer.show(xr[0, 0], axs[0, 0])
        self.slicer.show(x[0, 0], axs[1, 0])

        for i in range(1, 4):
            self.slicer.show(s[0, i], axs[0, i + 1])
            self.slicer.show(a[0, i], axs[1, i + 1])

        axs[0, 1].axis('off')
        axs[1, 1].axis('off')
        axs[0, 0].set_title("Recon T1w")
        axs[0, 2].set_title("Pred CSF")
        axs[0, 3].set_title("Pred GM")
        axs[0, 4].set_title("Pred WM")
        axs[1, 0].set_title("Input T1w")
        axs[1, 2].set_title("Atlas CSF")
        axs[1, 3].set_title("Atlas GM")
        axs[1, 4].set_title("Atlas WM")
        plt.show()
