import torch

from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_color


def plot_results(estimation_dict, est_filters, bin_stim, prop_model, loss_dict, target_angle, fs, device, dtype, show=True):
    """
    Plot generic multi-source results (time/frequency filters, localisation arrows,
    and colouration comparison).

    This helper consumes the unified loss_dict and extracts perceptual models from it
    (entries with a non-None `.model`) so callers do not need to keep a separate
    perceptual dictionary in sync.
    """
    perceptual_models = {
        key: loss_cfg.model
        for key, loss_cfg in loss_dict.items()
        if hasattr(loss_cfg, "model") and loss_cfg.model is not None
    }

    time_plot = plot_time(est_filters, show=False)
    freq_plot = plot_freq(est_filters, fs, show=False)

    arrow_plot = None
    loc_model = perceptual_models.get("localisation")
    if "localisation" in estimation_dict and loc_model is not None:
        if loc_model.lstn_pos_conf is not None:
            arrow_plot = plot_arrows(
                prop_model.lstn_pos,
                prop_model.spk_pos,
                torch.tensor(estimation_dict["localisation"], device=device, dtype=dtype),
                torch.tensor(estimation_dict["uncertainty"], device=device, dtype=dtype) if "uncertainty" in estimation_dict else None,
                torch.tensor(estimation_dict["colouration"], device=device, dtype=dtype) if "colouration" in estimation_dict else None,
                loc_target=torch.tensor(target_angle),
                show=False,
            )
        else:
            arrow_plot = plot_arrows(
                prop_model.lstn_pos,
                prop_model.spk_pos,
                torch.tensor(estimation_dict["localisation"], device=device, dtype=dtype),
                torch.tensor(estimation_dict["uncertainty"], device=device, dtype=dtype) if "uncertainty" in estimation_dict else None,
                torch.tensor(estimation_dict["colouration"], device=device, dtype=dtype) if "colouration" in estimation_dict else None,
                show=False,
            )

    color_l_plot = None
    color_r_plot = None
    col_model = perceptual_models.get("colouration")
    if col_model is not None:
        color_l_plot, color_r_plot = plot_color(
            bin_stim,
            col_model.reference_stim,
            prop_model.lstn_pos,
            show=False,
        )

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return {
        "time_plot": time_plot,
        "freq_plot": freq_plot,
        "arrow_plot": arrow_plot,
        "color_l_plot": color_l_plot,
        "color_r_plot": color_r_plot,
    }


def plot_stereo_results(estimation_dict, est_filters, bin_stim, stereo_prop, loss_dict, target_angle, fs, device, dtype, show=True):
    """Backward-compatible alias for older call sites."""
    return plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        prop_model=stereo_prop,
        loss_dict=loss_dict,
        target_angle=target_angle,
        fs=fs,
        device=device,
        dtype=dtype,
        show=show,
    )
