import numpy as np
import torch

from scripts.helpers.peripheral_processing import estimate_interauralcues
from scripts.helpers.template_comparison import itd_ild_template_distance


def init_localisation_cue_tracking(localisation_model):
    if localisation_model is None:
        return None

    target_itd, target_ild = _extract_target_itd_ild_pairs(localisation_model)
    return {
        "epoch": [],
        "itd": [],
        "ild": [],
        "distance_combined": [],
        "target_itd": target_itd,
        "target_ild": target_ild,
        "template_azimuths": localisation_model.template["azimuths"].squeeze(-1).detach().cpu().numpy(),
    }


def update_localisation_cue_tracking(
    tracking,
    epoch,
    accum_bin_stim,
    fs,
    gtfb,
    tau_iacc,
    localisation_model,
):
    if tracking is None:
        return

    epoch_itd, epoch_ild, epoch_distance_combined = compute_epoch_itd_ild_pairs(
        accum_bin_stim=accum_bin_stim,
        fs=fs,
        gtfb=gtfb,
        tau_iacc=tau_iacc,
        localisation_model=localisation_model,
    )

    tracking["epoch"].append(int(epoch))
    tracking["itd"].append(epoch_itd)
    tracking["ild"].append(epoch_ild)
    tracking["distance_combined"].append(epoch_distance_combined)


def finalize_localisation_cue_tracking(tracking):
    if tracking is None:
        return None

    return {
        "epoch": np.asarray(tracking["epoch"], dtype=np.int64),
        "itd": np.asarray(tracking["itd"], dtype=np.float64),
        "ild": np.asarray(tracking["ild"], dtype=np.float64),
        "distance_combined": np.asarray(tracking["distance_combined"], dtype=np.float64),
        "target_itd": np.asarray(tracking["target_itd"], dtype=np.float64),
        "target_ild": np.asarray(tracking["target_ild"], dtype=np.float64),
        "template_azimuths": np.asarray(tracking["template_azimuths"], dtype=np.float64),
    }


def compute_epoch_itd_ild_pairs(accum_bin_stim, fs, gtfb, tau_iacc, localisation_model):
    listener_means = []
    for listener_idx in sorted(accum_bin_stim.keys()):
        listener_batches = accum_bin_stim[listener_idx]
        if len(listener_batches) == 0:
            continue
        stacked = torch.cat(listener_batches, dim=0)
        listener_means.append(stacked.mean(dim=0, keepdim=True))

    if len(listener_means) == 0:
        raise ValueError("No binaural stimulus was accumulated for ITD/ILD tracking.")

    bin_stim_epoch = torch.cat(listener_means, dim=0).unsqueeze(0)
    cues = estimate_interauralcues(bin_stim_epoch, fs=fs, gtfb=gtfb, tau_IACC=tau_iacc)
    distance = itd_ild_template_distance(cues, localisation_model.template)

    itd = cues["itd"].squeeze(0).detach().cpu().numpy()
    ild = cues["ild"].squeeze(0).detach().cpu().numpy()
    distance_combined = distance["combined"].squeeze(0).detach().cpu().numpy()
    return itd.astype(np.float64), ild.astype(np.float64), distance_combined.astype(np.float64)


def _extract_target_itd_ild_pairs(localisation_model):
    template_azimuths = localisation_model.template["azimuths"].squeeze(-1)
    template_itd = localisation_model.template["itd"]
    template_ild = localisation_model.template["ild"]
    target_angles = localisation_model.target

    target_itd = []
    target_ild = []
    for angle in target_angles:
        idx = int(torch.argmin(torch.abs(template_azimuths - angle)).item())
        target_itd.append(template_itd[idx].detach().cpu().numpy())
        target_ild.append(template_ild[idx].detach().cpu().numpy())

    return np.asarray(target_itd, dtype=np.float64), np.asarray(target_ild, dtype=np.float64)
