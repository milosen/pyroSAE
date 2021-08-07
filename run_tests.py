import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from sae.util import onehot


def get_summaries(paths):
    summaries = []
    for path in paths:
        _, _, summary = torch.load(path, map_location="cpu")
        summaries.append(summary)

    return summaries


def plot_timecourse(ax, arrays, title, ylabel, xlabel, **kwargs):
    arr = np.stack(arrays)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax.plot(mean, **kwargs)
    try:
        color = kwargs['color']
    except KeyError:
        color = 'blue'
    
    ax.fill_between(
        np.arange(mean.shape[0]),
        mean + std, mean - std,
        color=color, alpha=0.2
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def make_axs_labels(axs, fig):
    for label, ax in axs.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


def plot_timeseries(ch: list):
    summaries = get_summaries(ch)

    loss_kl_cat = []
    loss_kl_mrf = []
    recon_loss = []
    loss_elbo_full = []
    dice_csf = []
    dice_gm = []
    dice_wm = []
    for summary in summaries:
        loss_kl_cat.append(np.array(summary['loss_kl_cat']))
        loss_kl_mrf.append(-np.array(summary['loss_kl_mrf']))
        loss_elbo_wo_mrf = np.array(summary['loss_elbo'])
        recon_loss.append(loss_elbo_wo_mrf - loss_kl_cat[-1])
        loss_elbo_full.append(recon_loss[-1] + loss_kl_cat[-1] - loss_kl_mrf[-1])
        dice = torch.stack(summary['dice'], dim=0).cpu().numpy()
        dice_csf.append(dice[:, 1])
        dice_gm.append(dice[:, 2])
        dice_wm.append(dice[:, 3])

    fig = plt.figure(figsize=[16, 12])
    axs = fig.subplot_mosaic(
        """
        AAAAAAAAABBBBBBBBB
        AAAAAAAAABBBBBBBBB
        AAAAAAAAABBBBBBBBB
        CCCCCCCCCDDDDDDDDD
        CCCCCCCCCDDDDDDDDD
        CCCCCCCCCDDDDDDDDD
        EEEEEEFFFFFFGGGGGG
        EEEEEEFFFFFFGGGGGG
        """
    )

    plot_timecourse(axs['A'], loss_kl_cat, "Loss KL categorical", "KL_cat/nats", "Epoch", color='tab:blue')
    plot_timecourse(axs['B'], loss_kl_mrf, "Loss KL MRF", "KL_MRF/nats", "Epoch", color='tab:blue')
    plot_timecourse(axs['C'], recon_loss, "Reconstruction Loss", "NLL/nats", "Epoch", color='tab:blue')
    plot_timecourse(axs['D'], loss_elbo_full, "ELBO loss", "-ELBO/nats", "Epoch", color='tab:blue')

    plot_timecourse(axs['E'], dice_csf, "Dice Coefficient CSF", "DSC", "Epoch", color='wheat')
    plot_timecourse(axs['F'], dice_gm, "Dice Coefficient GM", "DSC", "Epoch", color='tab:gray')
    plot_timecourse(axs['G'], dice_wm, "Dice Coefficient WM", "DSC", "Epoch", color='black')

    make_axs_labels(axs,  fig)

    fig.tight_layout()
    plt.show()

    return dice_csf, dice_gm, dice_wm


def threshold(seg_onehot, thresholds):
    return ((seg_onehot[:, i+1] > torch.Tensor([thr])).float() for i, thr in enumerate(thresholds))


def plot_classification(path, thr_mean=(0.16, 0.37, 0.54)):
    s, xr, x, tar, a = torch.load(path, map_location="cpu")

    s_csf_pr, s_gm_pr, s_wm_pr = threshold(s, list(thr_mean))

    s_am = onehot(torch.argmax(s, dim=1))
    tar = onehot(tar)

    fig = plt.figure(figsize=[8, 6])
    axs = fig.subplot_mosaic(
        """
        AFGH
        .IJK
        """
    )
    from sae.util import Slicer
    slicer = Slicer(2, 100)

    slicer.show(x[0, 0], axs['A'])
    axs['A'].set_title('T1w')

    slicer.show(s_am[0, 1], axs['F'])
    axs['F'].set_title('Argmax CSF')

    slicer.show(s_am[0, 2], axs['G'])
    axs['G'].set_title('Argmax GM')

    slicer.show(s_am[0, 3], axs['H'])
    axs['H'].set_title('Argmax WM')

    slicer.show(s_csf_pr[0], axs['I'])
    axs['I'].set_title('Mean PR CSF')

    slicer.show(s_gm_pr[0], axs['J'])
    axs['J'].set_title('Mean PR GM')

    slicer.show(s_wm_pr[0], axs['K'])
    axs['K'].set_title('Mean PR WM')

    # make_axs_labels(axs, fig)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # most of the experiments do not make sense because we don't have GM/WM/CSF
    # labels of the IXI dataset
    dice_coeff_vae_2 = plot_timeseries(['checkpoints/vae_epoch_0_cuda.pt'])
    plot_classification('checkpoints/last_tensors.pt')
