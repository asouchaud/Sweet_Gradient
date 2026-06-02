# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import os 
# import scipy
# import torchaudio

# from scripts.setup.setup import load_gtfb
# from scripts.setup.setup import load_hrirs
# from scripts.setup.setup import load_stim

# from scripts.perceptual_models.desena2020.DEPRACTED_desena2020_estimateinterauralcues import desena2020_estimateinterauralcues
# from scripts.perceptual_models.desena2020.DEPRACTED_desena2020_Hmin import desena2020_Hmin
# from scripts.perceptual_models.desena2020.desena2020 import desena2020

# from scripts.perceptual_models.llado2025.DEPRACTED_llado2025_distance import llado2025_distance
# from scripts.perceptual_models.llado2025.DEPRACTED_llado2025_estimateinterauralcues import llado2025_estimateinterauralcues
# from scripts.perceptual_models.llado2025.DEPRACTED_llado2025_estimatelocalisation import llado2025_estimatelocalisation_torch
# from scripts.perceptual_models.llado2025.llado2025 import llado2025

# from scripts.propagation.stereo_anechoic import stereo_anechoic


## ============================= HMIN TEST ============================== ##

# fs = 48000

# dtype = torch.float32
# device = 'cuda'

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# hrirs = load_hrirs(dtype, device, sofa_path = default_sofa_path, idx_path = sofa_idx_path)

# # # default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# # # sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\CIPIC_hrirs_idx.csv"
# # resampler = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = fs)

# # hrirs = load_hrirs(dtype, 'cpu', sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# # hrirs = resampler(hrirs).to(device=device, dtype=dtype)


# gtfb = load_gtfb(fs, device, dtype)

# template = desena2020_estimateinterauralcues(hrirs,fs, gtfb)
# template['azimuths'] = torch.arange(-90,95,5,dtype=dtype, device=device, requires_grad=True).view(-1,1)
# template['ild'] = template['ild'].squeeze()
# template['itd'] = template['itd'].squeeze()

# H_min = desena2020_Hmin(template)

# print(H_min)


## ============================= IMPULSES AT LOUDSPEAKERS TEST ============================== ##

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
# from scripts.perceptual_models.llado2025.llado2025 import llado2025
# from scripts.plotting.dense_arrows_plot import plot_arrows
# from scripts.plotting.graph_plot import plot_bin_cues

# fs = 48000
# dtype = torch.float32
# device = 'cpu'

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / r"data\impulses\SOFA\SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"

# # Listener at origin, two loudspeakers in front-left / front-right.
# lstn_pos = torch.tensor([[0.0, float(y)] for y in np.linspace(-1, 1, 25)], device=device, dtype=dtype)
# spk_pos = torch.tensor([[1.732, 1.0], [1.732, -1.0]], device=device, dtype=dtype)

# prop = BinauralFreeFieldPropagation(
#     fs=fs,
#     lstn_pos=lstn_pos,
#     spk_pos=spk_pos,
#     sofa_file_path=default_sofa_path,
#     device=device,
#     dtype=dtype,
# )

# # Mono impulse through unit source filters (one filter per loudspeaker).
# # stim = torch.randn(1, 1, int(0.01 * fs), device=device, dtype=dtype)

# stim = torch.zeros(1, 1, int(0.01 * fs), device=device, dtype=dtype)
# stim[:, :, 0] = 1.0

# spk_filters = torch.zeros(1, spk_pos.shape[0], 256, device=device, dtype=dtype)
# spk_filters[:, :, 0] = 1.0

# binaural_impulse = prop(stim, spk_filters)
# print("Binaural impulse shape (batch, listener, ear, time):", binaural_impulse.shape)

# # Set a reference target point at x=1 m, y=0 m for every listener.
# target_pos = torch.zeros((lstn_pos.shape[0], 2), device=device, dtype=dtype)
# target_pos[:, 0] = 1.0

# loc_model = llado2025(
#     loss=torch.nn.MSELoss(),
#     target_pos=target_pos,
#     fs=fs,
#     sofa_path=default_sofa_path,
#     lstn_pos=lstn_pos,
#     spk_pos=spk_pos,
#     device=device,
#     dtype=dtype,
# )

# estimated_localisation = loc_model(binaural_impulse)
# print("Estimated localisation shape (batch, listener):", estimated_localisation.shape)
# print("Estimated localisation (degrees):", estimated_localisation.squeeze(0))

# arrow_plot = plot_arrows(
#     lstn_pos=lstn_pos,
#     spk_pos=spk_pos,
#     loc=estimated_localisation,
#     target_pos=target_pos[0],  # Single 2D marker expected by plot_arrows.
#     show=False,
# )
# arrow_plot.suptitle("Llado2025 estimated localisation", fontsize=10)

# fig_bin_cues = plot_bin_cues(
#     bin_stim=binaural_impulse,
#     fs=fs,
#     gtfb=loc_model.gtfb,
#     show=False,
# )
# fig_bin_cues.suptitle("Interaural cues (ITD vs ILD)", fontsize=10)
# plt.show()



## ============================= VARIANCE OF LOCALISATION MODEL TEST ============================== ##

import torch
from pathlib import Path
from scripts.setup.setup import load_hrirs
from scripts.setup.setup import load_gtfb
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation


fs = 48000
dtype = torch.float64
device = 'cpu'

# Loudspkeaker and listener positions
lstn_pos = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
spk_pos = torch.tensor([[1,1]], device=device, dtype=dtype)

repo_root = Path(__file__).resolve().parent
default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"

prop = BinauralFreeFieldPropagation(fs=fs, 
                                    lstn_pos=lstn_pos, 
                                    spk_pos=spk_pos, 
                                    sofa_file_path=default_sofa_path, 
                                    device=device, 
                                    dtype=dtype
                                    )

loc_model = llado2025(loss=torch.nn.MSELoss(), 
                        target_pos=torch.tensor([0.0]), 
                        fs=fs,
                        sofa_path=default_sofa_path,
                        lstn_pos=lstn_pos,
                        spk_pos=spk_pos,
                        device=device, 
                        dtype=dtype
                        )

n_iter = 30

save = torch.zeros(n_iter, device=device, dtype=dtype)

# stim = torch.randn(n_iter, 1, int(0.01*fs), device=device, dtype=dtype)

# stim = torch.randn(n_iter, 1, int(0.01*fs), device=device, dtype=dtype)
# stim = torch.cat([stim, 2*stim], dim=1)

stim = torch.zeros(n_iter, 1, int(0.01*fs), device=device, dtype=dtype)
stim[:,0,0] = 1

filters = torch.zeros(1, 1, 256, device=device, dtype=dtype)
filters[0,0,0] = 1


for i in range(n_iter):

    bin_stim = prop(stim[i:i+1,:,:], filters)
    perc_loc = loc_model(bin_stim)

    save[i] = perc_loc.squeeze()

print("Localisation values:", save)
print("Mean over random noise stimuli:", torch.mean(save, dim=0))
print("Standard deviation over random noise stimuli:", torch.std(save, dim=0))


## =================================== MATLAB BINAURAL STIMULUS LOCALISATION COMPARISON TEST =================================== ##

# fs = 48000
# dtype = torch.float32
# device = 'cuda'

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\MATLAB_CIPIC_hrirs_idx.csv"

# data = scipy.io.loadmat(r"data\miscellaneous\desenaMATLAB_bin_stim.mat")
# tmp = np.zeros((data['bin_signal'].shape[0], data['bin_signal'][0,0].shape[0], data['bin_signal'][0,0].shape[1]))
# for i in range(data['bin_signal'].shape[0]):
#     tmp[i,:,:] = data['bin_signal'][i,0][:,:]

# bin_stim = torch.tensor(tmp, device=device, dtype=dtype).permute(0,2,1)

# hrirs = load_hrirs(dtype, device, sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# gtfb = load_gtfb(fs, device, dtype)

# # Loudspkeaker and listener positions
# loc_model = llado2025(loss = torch.nn.MSELoss(), target = torch.tensor([0.0], device=device, dtype=dtype), hrirs = hrirs, fs = fs, gtfb = gtfb)

# perc_loc = loc_model(bin_stim)

# print(perc_loc)


## =================================== MATLAB BINAURAL STIMULUS LOCALISATION COMPARISON TEST =================================== ##

# fs = 44100
# dtype = torch.float32
# device = 'cuda'

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\MATLAB_CIPIC_hrirs_idx.csv"

# data = scipy.io.loadmat(r"data\miscellaneous\desenaMATLAB_bin_stim.mat")
# tmp = np.zeros((data['bin_signal'].shape[0], data['bin_signal'][0,0].shape[0], data['bin_signal'][0,0].shape[1]))
# for i in range(data['bin_signal'].shape[0]):
#     tmp[i,:,:] = data['bin_signal'][i,0][:,:]

# bin_stim = torch.tensor(tmp, device=device, dtype=dtype).permute(0,2,1)

# hrirs = load_hrirs(dtype, device, sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# gtfb = load_gtfb(fs, device, dtype)

# # Loudspkeaker and listener positions
# unc_model = desena2020(loss = torch.nn.MSELoss(), target = torch.tensor([0.0], device=device, dtype=dtype), hrirs = hrirs, fs = fs, gtfb = gtfb)

# perc_unc = unc_model(bin_stim)

# print(perc_unc)


## ================================== ICLD LOCALISATION TEST ================================== ##

# fs = 48000
# dtype = torch.float32
# device = 'cuda'

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\CIPIC_hrirs_idx.csv"
# resampler = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = fs)

# hrirs = load_hrirs(dtype, 'cpu', sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# hrirs = resampler(hrirs).to(device=device, dtype=dtype)
# gtfb = load_gtfb(fs, device, dtype)
# stim = load_stim(fs, device, dtype)

# # Loudspkeaker and listener positions
# lstn_pos_conf = [[[-0.732, 0]]] # 9 listener positions
# spk_pos = [[1,1], [1,-1]]

# stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos_conf[0], spk_pos=spk_pos, fs=fs, sofa_file_path=default_sofa_path)

# loc_model = llado2025(loss = torch.nn.MSELoss(), target = torch.tensor([0.0], device=device, dtype=dtype), hrirs = hrirs, fs = fs, gtfb = gtfb)

# # ICLD = torch.tensor(np.loadtxt('./data/miscellaneous/VBAP_panning_curve.csv'), device=device, dtype=dtype)
# ICLD = torch.arange(-18,19,1, device=device, dtype=dtype)

# for i in range(len(ICLD)):
#     print(f"ICLD: {ICLD[i]} dB")
    
#     right_gains = torch.sqrt(1/(1+ 10**(ICLD[i]/10)))
#     left_gains = right_gains*10**(ICLD[i]/20)

#     impulses = torch.zeros(1,2,256, device=device, dtype=dtype)
#     impulses[0,0,0] = left_gains
#     impulses[0,1,0] = right_gains

#     bin_stim = stereo_prop(impulses)

#     perc_loc = loc_model(bin_stim)

#     print(perc_loc.item(), "degrees")

## ==================================== ENZO DEBUGGING SESSION ========================= ## 

# fs = 48000
# dtype = torch.float32
# device = 'cuda'

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\CIPIC_hrirs_idx.csv"

# hrirs = load_hrirs(dtype, device, sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# gtfb = load_gtfb(fs, device, dtype)
# stim = load_stim(fs, device, dtype)
# # # stim_dict = scipy.io.loadmat(r"data\miscellaneous\desenaMATLAB_bin_stim.mat")
# # # stim = torch.tensor(np.real(stim_dict['bin_signal'].T), dtype=dtype, device=device).unsqueeze(1)
# # stim = torch.zeros(1,1,24000, device=device, dtype=dtype)

# # Loudspkeaker and listener positions
# # lstn_pos_conf = [[[0, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0], [-0.232, 0.5], [-0.232, -0.5], [-1.232, 0.5], [-1.232, -0.5]]] # 9 listener positions
# spk_pos = [[1.4,1.4], [1,-1]]
# lstn_pos_conf = [[[0,0]]]

# stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos_conf[0], spk_pos=spk_pos, fs=fs, sofa_file_path=default_sofa_path)

# unc_model = desena2020(loss = torch.nn.MSELoss(), target = torch.tensor([0.0], device=device, dtype=dtype), hrirs = hrirs, fs = fs, gtfb = gtfb)

# impulses = torch.zeros(1,2,256, device=device, dtype=dtype)
# impulses[0,0,0] = 1
# # impulses[0,1,0] = 1

# bin_stim = stereo_prop(impulses)

# perc_unc = unc_model(bin_stim)

# print(perc_unc)



## ================================== ICLD UNCERTAINTY TEST ================================== ##

# fs = 48000
# dtype = torch.float32
# device = 'cuda'

# # repo_root = Path(__file__).resolve().parent
# # default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
# # sofa_idx_path = repo_root / r"data\impulses\SOFA\D1_48k_hrirs_idx.csv"

# repo_root = Path(__file__).resolve().parent
# default_sofa_path = repo_root / r"data\impulses\SOFA\CIPIC_subject_021.sofa"
# sofa_idx_path = repo_root / r"data\impulses\SOFA\CIPIC_hrirs_idx.csv"
# resampler = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = fs)

# hrirs = load_hrirs(dtype, 'cpu', sofa_path = default_sofa_path, idx_path = sofa_idx_path)
# hrirs = resampler(hrirs).to(device=device, dtype=dtype)
# gtfb = load_gtfb(fs, device, dtype)
# stim = load_stim(fs, device, dtype)

# # Loudspkeaker and listener positions
# lstn_pos_conf = [[[-0.732, 0]]] # 9 listener positions
# spk_pos = [[1,1], [1,-1]]

# stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos_conf[0], spk_pos=spk_pos, fs=fs, sofa_file_path=default_sofa_path)

# unc_model = desena2020(loss = torch.nn.MSELoss(), target = torch.tensor([0.0], device=device, dtype=dtype), hrirs = hrirs, fs = fs, gtfb = gtfb)

# # ICLD = torch.tensor(np.loadtxt('./data/miscellaneous/VBAP_panning_curve.csv'), device=device, dtype=dtype)
# ICLD = torch.arange(-18,19,1, device=device, dtype=dtype)
# uncs = np.zeros((len(ICLD)))

# for i in range(len(ICLD)):
#     print(f"ICLD: {ICLD[i]} dB")
    
#     right_gains = torch.sqrt(1/(1+ 10**(ICLD[i]/10)))
#     left_gains = right_gains*10**(ICLD[i]/20)

#     impulses = torch.zeros(1,2,256, device=device, dtype=dtype)
#     impulses[0,0,0] = left_gains
#     impulses[0,1,0] = right_gains

#     bin_stim = stereo_prop(impulses)

#     perc_unc = unc_model(bin_stim)

#     print(perc_unc.item())
#     uncs[i] = perc_unc.item()

# plt.plot(ICLD.detach().cpu().numpy(),uncs)
# plt.xlabel("ICLD (dB)")
# plt.ylabel("Uncertainty")
# plt.ylim(0.58,0.7)
# plt.grid()
# plt.show()


## ================================== CrossTalk Cancellation Filters ================================== ##


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, ifft
# import sofar as sf

# # ---------------------------
# # Parameters
# # ---------------------------
# fs = 48000
# c = 343.0
# distance = 2.0
# speaker_angle = 30
# ear_distance = 0.18
# N = 256

# # ---------------------------
# # Geometry
# # ---------------------------
# def get_ear_positions():
#     return np.array([
#         [-ear_distance/2, 0, 0],
#         [ ear_distance/2, 0, 0],
#     ])

# def get_speaker_positions():
#     theta = np.deg2rad(speaker_angle)
    
#     left_spk = np.array([
#         -distance * np.sin(theta),
#          distance * np.cos(theta),
#          0
#     ])
    
#     right_spk = np.array([
#          distance * np.sin(theta),
#          distance * np.cos(theta),
#          0
#     ])
    
#     return np.array([left_spk, right_spk])


# def sofa_find(source_angle, sofaile):
#     pos = np.mod(np.array(sofaile.SourcePosition[:, :2]),360)
       
#     source_angle = np.array([[source_angle%360, 0]])
#     diff = source_angle[:, None, :] - pos[None, :, :]
#     dists = np.linalg.vector_norm(diff, axis=2)
#     # dists = np.minimum(dists, 360-dists)  # account for circularity
#     return np.argmin(dists, axis=1)
# # ---------------------------
# # Impulse response
# # ---------------------------
# def impulse_response(src, dst):
#     d = np.linalg.norm(src - dst)
#     delay = int(np.round(d / c * fs))
    
#     ir = np.zeros(N)
#     if delay < N:
#         ir[delay] = 1.0 / d
#     return ir

# # ---------------------------
# # Build transfer matrix
# # ---------------------------
# def build_transfer_matrix():
#     ears = get_ear_positions()
#     spks = get_speaker_positions()
    
#     H = np.zeros((2, 2, N), dtype=np.complex64)
#     IRs = np.zeros((2, 2, N))
    
#     for i, ear in enumerate(ears):
#         for j, spk in enumerate(spks):
#             ir = impulse_response(spk, ear)
#             IRs[i, j] = ir
#             H[i, j] = fft(ir)
    
#     return H, IRs

# # ---------------------------
# # Inversion
# # ---------------------------
# def compute_xtc_filters(H, reg=0):
#     # N = H.shape[-1]
#     G = np.zeros_like(H, dtype=np.complex64)
    
#     for k in range(N):
#         Hf = H[:, :, k]
#         G[:, :, k] = np.linalg.inv(Hf + reg * np.eye(2))
    
#     return G

# # ---------------------------
# # Apply XTC
# # ---------------------------
# def apply_xtc(G, left, right):
#     # N = G.shape[-1]
#     L = fft(left, N)
#     R = fft(right, N)
    
#     YL = np.zeros(N, dtype=np.complex64)
#     YR = np.zeros(N, dtype=np.complex64)
    
#     for k in range(N):
#         out = G[:, :, k] @ np.array([L[k], R[k]])
#         YL[k], YR[k] = out
    
#     return np.real(ifft(YL)), np.real(ifft(YR))

# # ---------------------------
# # Plotting
# # ---------------------------
# def plot_irs(IRs):
#     plt.figure(figsize=(10, 6))
    
#     labels = [["H_LL", "H_RL"], ["H_LR", "H_RR"]]
    
#     for i in range(2):
#         for j in range(2):
#             plt.plot(IRs[i, j], label=labels[i][j])
    
#     plt.title("Impulse Responses (Time Domain)")
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid()

# def plot_frequency_response(H):
#     freqs = np.linspace(0, fs/2, N//2)
    
#     plt.figure(figsize=(10, 6))
    
#     labels = [["H_LL", "H_RL"], ["H_LR", "H_RR"]]
    
#     for i in range(2):
#         for j in range(2):
#             mag = 20 * np.log10(np.abs(H[i, j][:N//2]) + 1e-8)
#             plt.semilogx(freqs, mag, label=labels[i][j])
    
#     plt.title("Frequency Responses")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (dB)")
#     plt.legend()
#     plt.grid()

# def plot_xtc_effect(H, G):
#     """Check resulting system: G * H ≈ Identity"""
#     freqs = np.linspace(0, fs/2, N//2)
    
#     diag = []
#     offdiag = []
    
#     for k in range(N):
#         GH = G[:, :, k] @ H[:, :, k]
#         diag.append(np.abs(GH[0,0]) + np.abs(GH[1,1]))
#         offdiag.append(np.abs(GH[0,1]) + np.abs(GH[1,0]))
    
#     diag = np.array(diag[:N//2])
#     offdiag = np.array(offdiag[:N//2])
    
#     plt.figure(figsize=(10, 6))
#     plt.semilogx(freqs, 20*np.log10(diag + 1e-8), label="Desired (Diagonal)")
#     plt.semilogx(freqs, 20*np.log10(offdiag + 1e-8), label="Crosstalk (Off-diagonal)")
    
#     plt.title("Crosstalk Cancellation Performance")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (dB)")
#     plt.legend()
#     plt.grid()

# def smooth_third_octave(freq, mag_db):
#     smoothed = np.zeros_like(mag_db)

#     k = 2**(1/6)  # 1/3 octave half-width

#     for i, fc in enumerate(freq):
#         f_low = fc / k
#         f_high = fc * k

#         idx = np.where((freq >= f_low) & (freq <= f_high))[0]

#         smoothed[i] = np.mean(mag_db[idx])

#     return smoothed


# def causal_conv1d(x, kernel):
#     """
#     x: (T,) input signal
#     kernel: (K,) convolution kernel
#     returns: (T,) causal convolution output
#     """
    
#     K = len(kernel)

#     # pad on the left so output length matches input
#     x_padded = np.pad(x, (K-1, 0), mode='constant')

#     y = np.zeros_like(x, dtype=float)

#     for t in range(len(x)):
#         y[t] = np.sum(x_padded[t:t+K] * kernel[::-1])

#     return y


# t = np.linspace(0, 1, fs)
# left = np.sin(2*np.pi*440*t)
# right = np.sin(2*np.pi*550*t)
# stim = np.random.randn(N)
# impulse = np.zeros(N)
# impulse[0] = 1

# sofaile = sf.read_sofa("data\impulses\SOFA\SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
# left_spk_hrirs = sofaile.Data_IR[760] # 30 deg
# right_spk_hrirs = sofaile.Data_IR[8064] # 330 deg

# angles = [0,10,20,30,40,50,60,70,80,90]
# ctc_filts = np.zeros((len(angles), 2, N))

# for i in range(len(angles)):
#     idx = sofa_find(angles[i], sofaile)
#     print(f"Angle: {angles[i]} deg, SOFA index: {idx[0]}")

#     ref_idx = sofa_find(angles[i], sofaile)
#     ref_hrirs = sofaile.Data_IR[ref_idx[0]] # 0 deg 
#     ref_hrtf = np.fft.fft(ref_hrirs, n=N, axis = -1)

#     IRs = np.concatenate((left_spk_hrirs[:,None,:], right_spk_hrirs[:,None,:]), axis = 1)
#     H = np.fft.fft(IRs, n=N, axis = -1)
#     # H, IRs = build_transfer_matrix()
#     G = compute_xtc_filters(H)

#     L_f = G[0,0] * ref_hrtf[0] + G[0,1] * ref_hrtf[1]
#     R_f = G[1,0] * ref_hrtf[0] + G[1,1] * ref_hrtf[1]

#     L = np.real(np.fft.ifft(L_f, axis = -1)).reshape(1,-1)
#     R = np.real(np.fft.ifft(R_f, axis = -1)).reshape(1,-1)

#     filts = np.concatenate((L,R), axis = 0)

#     ref_ear_L = causal_conv1d(impulse,ref_hrirs[0])
#     ref_ear_R = causal_conv1d(impulse,ref_hrirs[1])

#     spk_L, spk_R = apply_xtc(G, ref_hrirs[0], ref_hrirs[1])

#     ear_L = causal_conv1d(spk_L, left_spk_hrirs[0]) + causal_conv1d(spk_R, right_spk_hrirs[0])
#     ear_R = causal_conv1d(spk_L, left_spk_hrirs[1]) + causal_conv1d(spk_R, right_spk_hrirs[1])

#     tmp_L = causal_conv1d(impulse, filts[0])
#     tmp_R = causal_conv1d(impulse, filts[1])
#     test_L = causal_conv1d(tmp_L, left_spk_hrirs[0]) + causal_conv1d(tmp_R, right_spk_hrirs[0])
#     test_R = causal_conv1d(tmp_L, left_spk_hrirs[1]) + causal_conv1d(tmp_R, right_spk_hrirs[1])

#     CTC_filts = filts


#     ctc_rms = np.sqrt(np.mean(CTC_filts**2))

#     CTC_fft = np.fft.rfft(CTC_filts/ctc_rms, n = N, axis=-1)

#     # Magnitude
#     CTC_f = 20*np.log10(np.abs(CTC_fft))

#     # Phase
#     CTC_phase = np.angle(CTC_fft)

#     CTC_phase_unwrapped = np.unwrap(CTC_phase, axis=-1)

#     freqs = np.fft.rfftfreq(N, d = 1/48000)

#     CTC_f_smooth_left = smooth_third_octave(freqs, CTC_f[0])
#     CTC_f_smooth_right = smooth_third_octave(freqs, CTC_f[1])

#     ctc_filts[i,:,:] = filts


# plt.figure()
# for i in range(len(angles)):
#     plt.subplot(2,1,1)
#     plt.title("Left Loudspeaker Filter", fontsize = 20)
#     mag_left = 20 * np.log10(np.abs(np.fft.rfft(ctc_filts[i,0])) + 1e-8)
#     plt.semilogx(freqs, mag_left, label=f"Angle {angles[i]} deg", linewidth = 2)
#     plt.grid()
#     plt.subplot(2,1,2)
#     plt.title("Right Loudspeaker Filter", fontsize = 20)
#     mag_right = 20 * np.log10(np.abs(np.fft.rfft(ctc_filts[i,1])) + 1e-8)
#     plt.semilogx(freqs, mag_right, label=f"Angle {angles[i]} deg", linewidth = 2)
#     plt.grid()
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.legend(fontsize = 12)
# plt.grid()
# plt.show()


## ================================== CrossTalk Cancellation Filters Ear Signals ================================== ##


# import numpy as np
# import matplotlib.pyplot as plt
# import sofar as sf

# # ---------------------------
# # Parameters
# # ---------------------------
# fs = 48000
# N = 2048              # use larger FFT to reduce circular artefacts
# reg = 0           # small regularisation

# # ---------------------------
# # Helper: find SOFA index
# # ---------------------------
# def sofa_find(source_angle, sofa):
#     pos = np.mod(np.array(sofa.SourcePosition[:, :2]), 360)
#     source_angle = np.array([[source_angle % 360, 0]])
#     diff = source_angle[:, None, :] - pos[None, :, :]
#     dists = np.linalg.norm(diff, axis=2)
#     return np.argmin(dists, axis=1)

# # ---------------------------
# # Load HRIRs
# # ---------------------------
# sofa = sf.read_sofa("data/impulses/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

# deg = 30
# left_idx  = sofa_find(deg, sofa)
# right_idx = sofa_find(360 - deg, sofa)

# # shape: (ear, samples)
# left_hrir  = sofa.Data_IR[left_idx[0]]
# right_hrir = sofa.Data_IR[right_idx[0]]

# # Build H: (ear, speaker, time)
# IRs = np.stack([
#     [left_hrir[0],  right_hrir[0]],   # left ear
#     [left_hrir[1],  right_hrir[1]]    # right ear
# ])

# # ---------------------------
# # Optional: align direct path
# # ---------------------------
# def align_ir(ir):
#     idx = np.argmax(np.abs(ir))
#     return np.roll(ir, -idx)

# for i in range(2):
#     for j in range(2):
#         IRs[i, j] = align_ir(IRs[i, j])

# # ---------------------------
# # FFT → transfer matrix
# # ---------------------------
# H = np.fft.fft(IRs, n=N, axis=-1)  # (2,2,N)

# # ---------------------------
# # Compute inverse (regularised)
# # ---------------------------
# G = np.zeros_like(H, dtype=np.complex64)

# for k in range(N):
#     Hf = H[:, :, k]

#     # Conventional inverse
#     G[:,:,k] = np.linalg.inv(Hf + reg * np.eye(2))

#     # Tikhonov inverse
#     # G[:, :, k] = np.linalg.inv(
#     #     Hf.conj().T @ Hf + reg * np.eye(2)
#     # ) @ Hf.conj().T

# # ---------------------------
# # Desired ear signal
# # ---------------------------
# dL = np.zeros(N)
# dR = np.zeros(N)
# dL[0] = 1.0   # impulse at left ear

# idx = sofa_find(0, sofa)
# dL = np.pad(sofa.Data_IR[idx[0], 0], (0, N - len(sofa.Data_IR[idx[0], 0])))  # left ear 
# dR = np.pad(sofa.Data_IR[idx[0], 1], (0, N - len(sofa.Data_IR[idx[0], 1])))  # right ear

# # FFT
# D = np.stack([np.fft.fft(dL, n=N), np.fft.fft(dR, n=N)])  # (2, N)

# # ---------------------------
# # Compute speaker signals
# # ---------------------------
# S = np.zeros((2, N), dtype=np.complex64)

# for k in range(N):
#     S[:, k] = G[:, :, k] @ D[:, k]

# spk_L = np.real(np.fft.ifft(S[0]))
# spk_R = np.real(np.fft.ifft(S[1]))

# # ---------------------------
# # Propagate to ears
# # ---------------------------


# def causal_conv1d(x, kernel):
#     """
#     x: (T,) input signal
#     kernel: (K,) convolution kernel
#     returns: (T,) causal convolution output
#     """
    
#     K = len(kernel)

#     # pad on the left so output length matches input
#     x_padded = np.pad(x, (K-1, 0), mode='constant')

#     y = np.zeros_like(x, dtype=float)

#     for t in range(len(x)):
#         y[t] = np.sum(x_padded[t:t+K] * kernel[::-1])

#     return y


# ear_L = causal_conv1d(spk_L, IRs[0,0]) + \
#         causal_conv1d(spk_R, IRs[0,1])

# ear_R = causal_conv1d(spk_L, IRs[1,0]) + \
#         causal_conv1d(spk_R, IRs[1,1])

# # ---------------------------
# # Plot results
# # ---------------------------
# plt.figure(figsize=(10,5))
# plt.title("Ear Signals (Time Domain)")
# plt.subplot(2,1,1)
# plt.plot(ear_L, label="Left ear (should be impulse)")
# plt.plot(dL, label="Desired left ear (impulse)", linestyle='dashed')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(ear_R, label="Right ear")
# plt.plot(dR, label="Desired right ear", linestyle='dashed')  
# plt.legend()
# plt.grid()

# # ---------------------------
# # Frequency-domain check: H·G
# # ---------------------------
# diag = []
# offdiag = []

# for k in range(N):
#     GH = H[:, :, k] @ G[:, :, k]
#     diag.append(np.abs(GH[0,0]) + np.abs(GH[1,1]))
#     offdiag.append(np.abs(GH[0,1]) + np.abs(GH[1,0]))

# freqs = np.fft.rfftfreq(N, 1/fs)

# plt.figure(figsize=(10,5))
# plt.title("Crosstalk Cancellation (Frequency Domain)")
# plt.semilogx(freqs, 20*np.log10(np.array(diag[:N//2 +1]) + 1e-10), label="Diagonal")
# plt.semilogx(freqs, 20*np.log10(np.array(offdiag[:N//2 +1]) + 1e-10), label="Off-diagonal")
# plt.legend()
# plt.grid()


# # ---------------------------
# # Print residual error
# # ---------------------------
# print("Right ear residual (RMS):", np.sqrt(np.mean(ear_R**2)))


# L = G[0,0]*D[0] + G[0,1]*D[1]
# R = G[1,0]*D[0] + G[1,1]*D[1]

# L_time = np.real(np.fft.ifft(L))
# R_time = np.real(np.fft.ifft(R))

# ear_2_L = causal_conv1d(L_time, IRs[0,0]) + causal_conv1d(R_time, IRs[0,1])
# ear_2_R = causal_conv1d(L_time, IRs[1,0]) + causal_conv1d(R_time, IRs[1,1])

# plt.figure(figsize=(10,5))
# plt.title("Ear Signals (Time Domain)")
# plt.subplot(2,1,1)
# plt.plot(ear_2_L, label="Time implementation - Left ear")
# plt.plot(dL, label="Target - Left ear", linestyle='dashed')
# plt.grid()
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(ear_2_R, label="Time implementation - Right ear")
# plt.plot(dR, label="Target - Right ear", linestyle='dashed')
# plt.legend()
# plt.grid()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, ifft
# import sofar as sf

# # ---------------------------
# # Parameters
# # ---------------------------
# fs = 48000
# c = 343.0
# distance = 2.0
# speaker_angle = 30
# ear_distance = 0.18
# N = 256

# # ---------------------------
# # Geometry
# # ---------------------------
# def get_ear_positions():
#     return np.array([
#         [-ear_distance/2, 0, 0],
#         [ ear_distance/2, 0, 0],
#     ])

# def get_speaker_positions():
#     theta = np.deg2rad(speaker_angle)
    
#     left_spk = np.array([
#         -distance * np.sin(theta),
#          distance * np.cos(theta),
#          0
#     ])
    
#     right_spk = np.array([
#          distance * np.sin(theta),
#          distance * np.cos(theta),
#          0
#     ])
    
#     return np.array([left_spk, right_spk])


# def sofa_find(source_angle, sofaile):
#     pos = np.mod(np.array(sofaile.SourcePosition[:, :2]),360)
       
#     source_angle = np.array([[source_angle%360, 0]])
#     diff = source_angle[:, None, :] - pos[None, :, :]
#     dists = np.linalg.vector_norm(diff, axis=2)
#     # dists = np.minimum(dists, 360-dists)  # account for circularity
#     return np.argmin(dists, axis=1)
# # ---------------------------
# # Impulse response
# # ---------------------------
# def impulse_response(src, dst):
#     d = np.linalg.norm(src - dst)
#     delay = int(np.round(d / c * fs))
    
#     ir = np.zeros(N)
#     if delay < N:
#         ir[delay] = 1.0 / d
#     return ir

# # ---------------------------
# # Build transfer matrix
# # ---------------------------
# def build_transfer_matrix():
#     ears = get_ear_positions()
#     spks = get_speaker_positions()
    
#     H = np.zeros((2, 2, N), dtype=np.complex64)
#     IRs = np.zeros((2, 2, N))
    
#     for i, ear in enumerate(ears):
#         for j, spk in enumerate(spks):
#             ir = impulse_response(spk, ear)
#             IRs[i, j] = ir
#             H[i, j] = fft(ir)
    
#     return H, IRs

# # ---------------------------
# # Inversion
# # ---------------------------
# def compute_xtc_filters(H, reg=0):
#     # N = H.shape[-1]
#     G = np.zeros_like(H, dtype=np.complex64)
    
#     for k in range(N):
#         Hf = H[:, :, k]
#         G[:, :, k] = np.linalg.inv(Hf + reg * np.eye(2))
    
#     return G

# # ---------------------------
# # Apply XTC
# # ---------------------------
# def apply_xtc(G, left, right):
#     # N = G.shape[-1]
#     L = fft(left, N)
#     R = fft(right, N)
    
#     YL = np.zeros(N, dtype=np.complex64)
#     YR = np.zeros(N, dtype=np.complex64)
    
#     for k in range(N):
#         out = G[:, :, k] @ np.array([L[k], R[k]])
#         YL[k], YR[k] = out
    
#     return np.real(ifft(YL)), np.real(ifft(YR))

# # ---------------------------
# # Plotting
# # ---------------------------
# def plot_irs(IRs):
#     plt.figure(figsize=(10, 6))
    
#     labels = [["H_LL", "H_RL"], ["H_LR", "H_RR"]]
    
#     for i in range(2):
#         for j in range(2):
#             plt.plot(IRs[i, j], label=labels[i][j])
    
#     plt.title("Impulse Responses (Time Domain)")
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid()

# def plot_frequency_response(H):
#     freqs = np.linspace(0, fs/2, N//2)
    
#     plt.figure(figsize=(10, 6))
    
#     labels = [["H_LL", "H_RL"], ["H_LR", "H_RR"]]
    
#     for i in range(2):
#         for j in range(2):
#             mag = 20 * np.log10(np.abs(H[i, j][:N//2]) + 1e-8)
#             plt.semilogx(freqs, mag, label=labels[i][j])
    
#     plt.title("Frequency Responses")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (dB)")
#     plt.legend()
#     plt.grid()

# def plot_xtc_effect(H, G):
#     """Check resulting system: G * H ≈ Identity"""
#     freqs = np.linspace(0, fs/2, N//2)
    
#     diag = []
#     offdiag = []
    
#     for k in range(N):
#         GH = G[:, :, k] @ H[:, :, k]
#         diag.append(np.abs(GH[0,0]) + np.abs(GH[1,1]))
#         offdiag.append(np.abs(GH[0,1]) + np.abs(GH[1,0]))
    
#     diag = np.array(diag[:N//2])
#     offdiag = np.array(offdiag[:N//2])
    
#     plt.figure(figsize=(10, 6))
#     plt.semilogx(freqs, 20*np.log10(diag + 1e-8), label="Desired (Diagonal)")
#     plt.semilogx(freqs, 20*np.log10(offdiag + 1e-8), label="Crosstalk (Off-diagonal)")
    
#     plt.title("Crosstalk Cancellation Performance")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (dB)")
#     plt.legend()
#     plt.grid()

# def smooth_third_octave(freq, mag_db):
#     smoothed = np.zeros_like(mag_db)

#     k = 2**(1/6)  # 1/3 octave half-width

#     for i, fc in enumerate(freq):
#         f_low = fc / k
#         f_high = fc * k

#         idx = np.where((freq >= f_low) & (freq <= f_high))[0]

#         smoothed[i] = np.mean(mag_db[idx])

#     return smoothed


# def causal_conv1d(x, kernel):
#     """
#     x: (T,) input signal
#     kernel: (K,) convolution kernel
#     returns: (T,) causal convolution output
#     """
    
#     K = len(kernel)

#     # pad on the left so output length matches input
#     x_padded = np.pad(x, (K-1, 0), mode='constant')

#     y = np.zeros_like(x, dtype=float)

#     for t in range(len(x)):
#         y[t] = np.sum(x_padded[t:t+K] * kernel[::-1])

#     return y


# t = np.linspace(0, 1, fs)
# left = np.sin(2*np.pi*440*t)
# right = np.sin(2*np.pi*550*t)
# stim = np.random.randn(N)
# impulse = np.zeros(N)
# impulse[0] = 1

# sofaile = sf.read_sofa("data\impulses\SOFA\SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
# deg = 30
# left_idx = sofa_find(deg, sofaile)
# right_idx = sofa_find(360-deg, sofaile)
# left_spk_hrirs = sofaile.Data_IR[left_idx[0]] # 30 deg
# right_spk_hrirs = sofaile.Data_IR[right_idx[0]] # 330 deg

# IRs = np.concatenate((left_spk_hrirs[:,None,:], right_spk_hrirs[:,None,:]), axis = 1)
# H = np.fft.fft(IRs, n=N, axis = -1)
# # H, IRs = build_transfer_matrix()
# G = compute_xtc_filters(H)

# angles = [0,10,20,30,45,50,60,70,80]
# ctc_filts = np.zeros((len(angles), 2, N))

# for i in range(len(angles)):
#     idx = sofa_find(angles[i], sofaile)
#     print(f"Angle: {angles[i]} deg, SOFA index: {idx[0]}")

#     # ref_idx = sofa_find(angles[i], sofaile)
#     # ref_hrirs = sofaile.Data_IR[ref_idx[0]] # 0 deg 
#     ref_hrirs = np.zeros((2,N))
#     ref_hrirs[0,0] = 1
#     ref_hrtf = np.fft.fft(ref_hrirs, n=N, axis = -1)

#     L_f = G[0,0] * ref_hrtf[0] + G[0,1] * ref_hrtf[1]
#     R_f = G[1,0] * ref_hrtf[0] + G[1,1] * ref_hrtf[1]

#     L = np.real(np.fft.ifft(L_f, axis = -1)).reshape(1,-1)
#     R = np.real(np.fft.ifft(R_f, axis = -1)).reshape(1,-1)

#     filts = np.concatenate((L,R), axis = 0)

#     ref_ear_L = np.convolve(impulse,ref_hrirs[0], mode = 'same')
#     ref_ear_R = np.convolve(impulse,ref_hrirs[1], mode = 'same')

#     spk_L, spk_R = apply_xtc(G, ref_hrirs[0], ref_hrirs[1])

#     ear_L = np.convolve(spk_L, left_spk_hrirs[0], mode = 'same') + np.convolve(spk_R, right_spk_hrirs[0], mode = 'same')
#     ear_R = np.convolve(spk_L, left_spk_hrirs[1], mode = 'same') + np.convolve(spk_R, right_spk_hrirs[1], mode = 'same')

#     tmp_L = np.convolve(impulse, filts[0], mode = 'same')
#     tmp_R = np.convolve(impulse, filts[1], mode = 'same')
#     test_L = np.convolve(tmp_L, left_spk_hrirs[0], mode = 'same') + np.convolve(tmp_R, right_spk_hrirs[0], mode = 'same')
#     test_R = np.convolve(tmp_L, left_spk_hrirs[1], mode = 'same') + np.convolve(tmp_R, right_spk_hrirs[1], mode = 'same')

#     CTC_filts = filts


#     ctc_rms = np.sqrt(np.mean(CTC_filts**2))

#     CTC_fft = np.fft.rfft(CTC_filts/ctc_rms, n = N, axis=-1)

#     # Magnitude
#     CTC_f = 20*np.log10(np.abs(CTC_fft))

#     # Phase
#     CTC_phase = np.angle(CTC_fft)

#     CTC_phase_unwrapped = np.unwrap(CTC_phase, axis=-1)

#     freqs = np.fft.rfftfreq(N, d = 1/48000)

#     CTC_f_smooth_left = smooth_third_octave(freqs, CTC_f[0])
#     CTC_f_smooth_right = smooth_third_octave(freqs, CTC_f[1])

#     ctc_filts[i,0,:] = test_L
#     ctc_filts[i,1,:] = test_R


# plt.figure()
# for i in range(len(angles)):
#     plt.subplot(3,3,1+i)
#     plt.grid()
#     plt.title(f"Angle {angles[i]} deg", fontsize = 16)
#     ctc_left = 20*np.log10(np.abs(np.fft.rfft(ctc_filts[i,0])) + 1e-8)
#     plt.semilogx(freqs, smooth_third_octave(freqs, ctc_left), label="CTC: left ear", linewidth = 3)
#     ctc_right = 20*np.log10(np.abs(np.fft.rfft(ctc_filts[i,1])) + 1e-8)
#     plt.semilogx(freqs, smooth_third_octave(freqs, ctc_right), label="CTC: right ear", linewidth = 2)
#     # plt.semilogx(freqs, ctc, label="CTC", linewidth = 2)

#     # ref_idx = sofa_find(angles[i], sofaile)
#     # ref_hrirs = sofaile.Data_IR[ref_idx[0]] # 0 deg 
#     ref = 20*np.log10(np.abs(np.fft.rfft(ref_hrirs[0])) + 1e-8)
#     plt.semilogx(freqs, smooth_third_octave(freqs, ref), label="Reference: left ear", linewidth = 3, color = 'blue', linestyle = '--')
#     ref = 20*np.log10(np.abs(np.fft.rfft(ref_hrirs[1])) + 1e-8)
#     plt.semilogx(freqs, smooth_third_octave(freqs, ref), label="Reference: right ear", linewidth = 3, color = 'orange', linestyle = '--')
#     # plt.semilogx(freqs, ref, label="Reference", linewidth = 3, color = 'black', linestyle = '--')
#     plt.legend(fontsize = 12)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.legend(fontsize = 12)
# plt.grid()

# # plt.figure()
# # for i in range(len(angles)):
# #     plt.subplot(3,3,1+i)
# #     plt.grid()
# #     plt.title(f"Angle {angles[i]} deg", fontsize = 16)
# #     plt.plot(ctc_filts[i,0], label="CTC: left ear", linewidth = 3)
# #     plt.plot(ctc_filts[i,1], label="CTC: right ear", linewidth = 2)
# #     # plt.semilogx(freqs, ctc, label="CTC", linewidth = 2)

# #     # ref_idx = sofa_find(angles[i], sofaile)
# #     # ref_hrirs = sofaile.Data_IR[ref_idx[0]] # 0 deg  
# #     plt.plot(ref_hrirs[0], label="Reference: left ear", linewidth = 3, color = 'blue', linestyle = '--')
    
# #     plt.plot(ref_hrirs[1], label="Reference: right ear", linewidth = 3, color = 'orange', linestyle = '--')
# #     # plt.semilogx(freqs, ref, label="Reference", linewidth = 3, color = 'black', linestyle = '--')
# #     plt.legend(fontsize = 12)

# plot_xtc_effect(H, G)

# plt.show()


## -------------------------------------------- BATCH LEARNING EXPERIMENT -------------------------------------------- ##

# # Enable MPS fallback for PyTorch on macOS
# import os
# import json
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# # Import interface libraries
# from pathlib import Path
# import time
# from types import SimpleNamespace

# # Import computational libraries
# import numpy as np
# import torch
# import torch.nn.functional as F

# # Import Plotting libraries
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.cm as cm
# import matplotlib.colors as colors


# # Import project modules
# from scripts.optimisation.custom_modules import MultiSourceFIR
# from scripts.plotting.results_plot import plot_results
# from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
# from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
# from scripts.perceptual_models.llado2025.llado2025 import llado2025
# from scripts.perceptual_models.desena2020.desena2020 import desena2020
# from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
# from scripts.optimisation.regularisation import second_moment_regularisation
# from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
# from scripts.saving.save_results import saver


# def run_experiment(loss_dict_builder, run_label, target_angle):
#     repo_root = Path(__file__).resolve().parent
#     d = 60
#     env = SimpleNamespace( # Include here all variables that you want stored into config.json
#         epochs=300,
#         num_lstn_pos=4,
#         listener_batch_size=2,  # 5 mirrored pairs plus the centre listener
#         symmetric_listener_sampling=False,
#         n_samples=128,
#         target_pos=[d, float(np.tan(np.deg2rad(target_angle)) * d)],
#         selected_stim="impulse",
#         sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
#         sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
#         lr=0.001,
#         seed=46,
#         log=1,
#         fs=48000,
#     )
#     env.dtype = torch.float64
#     env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
#     env.device = torch.device(env.device_str)

# ## Initial setup
#     print(f"Using device: {env.device}")

#     # Loudspkeaker and listener positions
#     n_per_segment = env.num_lstn_pos // 4
#     segments = [
#         np.linspace(-0.5, -0.325, n_per_segment),
#         np.linspace(-0.225, -0.05, n_per_segment),
#         np.linspace(0.05, 0.225, n_per_segment),
#         np.linspace(0.325, 0.5, n_per_segment),
#     ]
#     y_positions = np.concatenate(segments)
#     env.lstn_pos = [[0.0, float(y)] for y in y_positions]
#     # env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.3, 0.3, env.num_lstn_pos)] 
#     # env.lstn_pos = [[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]] # Centre listener plus two mirrored pairs at 27.5 cm from the centre
#     base_angle_rad = np.deg2rad(30.0)
#     env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
#     env.spk_pos = [
#         [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
#         [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
#     ]
#     env.n_sources = len(env.spk_pos)

#     # Reset the seed so each run starts from the same point.
#     set_seed(seed=env.seed)

# ## Load data
#     gtfb = load_gtfb(fs=env.fs, device=env.device, dtype=env.dtype)  # Load Gamma Tone Filter Bank
#     stim = load_stim(fs=env.fs, device=env.device, dtype=env.dtype, selected_stim=env.selected_stim)  # Load stimulus
#     hrirs = load_hrirs(dtype=env.dtype, device=env.device, sofa_path=env.sofa_path, idx_path=env.sofa_idx_path) # Load HRIRs using specific location from SOFA file

# ## Initialize model
#     # Initialize sound propagation module
#     stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path, apply_distance_attenuation=True) # Load stereo anechoic propagation module

#     # Initiate perceptual models
#     loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # Llado et al. 2025 model
#     col_model = spectral_matching(loss = F.l1_loss, target = 0, stim=stim, hrirs=hrirs, target_pos=env.target_pos, fs=env.fs).to(device=env.device, dtype=env.dtype) # Spectral Matching model
#     unc_model = desena2020(loss = F.l1_loss, target = 0, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
#     regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
#     loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)

#     # Initialise neural network
#     nn_model = MultiSourceFIR(
#         n_samples=env.n_samples,
#         n_sources=env.n_sources,
#     ).to(device=env.device, dtype=env.dtype)
#     # Initialise FIR filter weights
#     with torch.no_grad():
#         for i, branch in enumerate(nn_model.branches):
#             branch.weight.fill_(0.0)
#         nn_model.branches[0].weight[0, 0] = 1.0  # Initialise left source FIR to an impulse

#     nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

# ## Initliase saving folder and logger
#     saver_obj = saver(loss_dict, env.lstn_pos)
#     logger = saver_obj if env.log else None

# ## Training loop
#     # Train the neural network
#     trainer = Train(n_epoch=env.epochs,
#                     nn=nn_model,
#                     prop=stereo_prop,
#                     loss_dict=loss_dict,
#                     optimiser=nn_optimiser,
#                     loggers = [logger, None], 
#                     listener_batch_size=env.listener_batch_size,
#                     symmetric_listener_sampling=env.symmetric_listener_sampling) # Initialize training class
#     time_train_start = time.time()
#     trainer.train()
#     env.time_train_seconds = time.time() - time_train_start
#     print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

# ## Print and Plot results
#     # Evaluate final neural network FIRs
#     eval_perceptual_dict = {
#             "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=env.device, dtype=env.dtype),
#             "uncertainty": ListenerBasedLoss(model=unc_model, weight=1, device=env.device, dtype=env.dtype),
#             "colouration": ListenerBasedLoss(model=col_model, weight=1, device=env.device, dtype=env.dtype)
#         }
#     eval_lstn_pos = [[0.0 , 0.0], [0.0, 0.275], [0.0, -0.275]]
#     stereo_prop_eval = BinauralFreeFieldPropagation(stim=stim, lstn_pos=eval_lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path, apply_distance_attenuation=True)
#     trainer.prop = stereo_prop_eval
#     est_filters, estimation_dict, bin_stim = trainer.evaluate(loss_dict=eval_perceptual_dict)
#     # Normalise filters to prevent clipping 
#     est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

#     # Print estimated perceptual metrics
#     for key, value in estimation_dict.items():
#         print(f"Estimated {key}: {value}")

#     saver_obj.save_results(
#         estimation_dict=estimation_dict,
#         est_filters=est_filters,
#         fs=env.fs,
#         lstn_pos=eval_lstn_pos,
#         spk_pos=env.spk_pos,
#         trainer=trainer,
#         env=env,
#     )
#     plots = plot_results(
#         estimation_dict=estimation_dict,
#         est_filters=est_filters,
#         bin_stim=bin_stim,
#         prop_model=stereo_prop_eval,
#         loss_dict=loss_dict,
#         target_pos=env.target_pos,
#         stim=stim,
#         fs=env.fs,
#         device=env.device,
#         dtype=env.dtype,
#         show=False,
#     )
#     saver_obj.save_plots(plots)

#     # Save per-epoch source FIR evolution as videos (one per source)
#     # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
#     # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)

#     plt.close('all')

#     return(saver_obj.run_dir, eval_lstn_pos, env.spk_pos)


# def DAFx2026_fig2_plots(run_folders, lstn_pos, spk_pos):

#     arrow_width = 3
#     arrow_head_width = 0.12
#     arrow_head_len = 0.14
#     arrow_len = 1.2
#     left_ir_width = 3
#     right_ir_width = 2 
#     fontsize = 14

#     # Create a colormap with log scaling
#     max_col = 18.5
#     norm = colors.Normalize(vmin=0, vmax=max_col)  # avoid zero
#     cmap = colors.LinearSegmentedColormap.from_list(
#         "bright_green_red",
#         [(0,1,0), (0.8,0.4,0), (0.8,0,0)]  # green → yellow → red
#     )

#     fig, axes = plt.subplots(2, 1)  # 3 columns × 7 rows
#     axes = axes.flatten()  # easier indexing

#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
    
#     for i, ax in enumerate(axes):
#         folder_name = run_folders[i//2]
#         json_file_path = os.path.join(folder_name, "results.json")
#         json_data = json.load(open(json_file_path))

#         #### IRs PLOT ####
#         if i % 2 == 0:
#             filters_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
#             filters = np.load(filters_path)
#             filters = filters * 0.9 / np.max(np.abs(filters))

#             ax.plot(filters[0], linewidth = left_ir_width)
#             ax.plot(filters[1], linewidth = right_ir_width)
#             ax.grid()
#             ax.set_ylim(-0.5,1)
#             ax.set_ylabel("Amplitude", fontsize = fontsize)

#             if i > 3:
#                 ax.set_xlabel("Coefficients", fontsize = fontsize)


        
#         #### ARROWS PLOT #####
#         if i % 2 == 1:
#             loc = json_data['localisation']
#             unc = json_data['uncertainty']
#             col = json_data['colouration']

#             ax.scatter(spk_pos[:,0], spk_pos[:,1], c='black', label='Speaker', marker = 'x')
#             ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

#             # === DIRECTIVITY AROUND EACH SPEAKER ===
#             angles = np.linspace(0, 2*np.pi, 200)

#             for sx, sy in spk_pos:
#                 # Example cardioid pattern
#                 r = 0.15 * (1 + np.cos(angles))  # scale controls size

#                 x_dir = sx + r * np.cos(angles)
#                 y_dir = sy + r * np.sin(angles)

#                 ax.plot(x_dir, y_dir, color='black', linewidth=1, alpha=0.6, zorder=2)
#                 # Optional fill:
#                 # ax.fill(x_dir, y_dir, color='gray', alpha=0.15, zorder=1)


#             for x, y, ang, leng, col_val in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc, col):  

#                 angles = [0,10,20]
#                 x_lin = np.linspace(-1,3,200)
#                 y_lin = x_lin * np.tan(np.deg2rad(angles[i//2])) - y

#                 ax.plot(x_lin,y_lin, color = "black", linestyle = "dashed", lw = 1, zorder = 2)
#                 dx = (1-leng)*arrow_len * np.cos(np.deg2rad(ang))
#                 dy = (1-leng)*arrow_len * np.sin(np.deg2rad(ang))

#                 color = cmap(norm(max(col_val, 1e-3)))  # avoid zero for log scale

#                 ax.arrow(float(x), float(y), float(dx), float(dy),
#                         head_width=arrow_head_width, head_length=arrow_head_len, fc=color, ec=color, linewidth = arrow_width, zorder = 3)  

#             ax.set_ylim(-1.25,1.25)
#             ax.set_xlim(-0.5,2.0)
#             ax.grid(zorder = 1)
#             ax.set_ylabel('Y position [m]', fontsize = fontsize)
#             if i > 3:
#                 ax.set_xlabel('X position [m]', fontsize = fontsize)

#         ax.tick_params(axis='both', labelsize=fontsize)  

#     plt.tight_layout(rect=[0, 0.05, 1, 1])

#     cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
#     cbar.set_label("LSD", fontsize=fontsize)
#     cbar.ax.tick_params(labelsize=fontsize)
#     plt.savefig("DAFx2026_fig2.pdf", format = "pdf")
#     plt.show()

    



# if __name__ == "__main__":
#     run_dir_00, lstn_pos, spk_pos = run_experiment(
#         loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
#             # "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=device, dtype=dtype),
#             "regularisation_l": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1.0),
#             "regularisation_r": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1.0)
#         },
#         run_label="localisation_only_0",
#         target_angle=30.0
#     )


#     folders = [run_dir_00]

#     DAFx2026_fig2_plots(folders, np.array(lstn_pos), np.array(spk_pos))


## ----------------------------------------------------- DIFFERENT FILTER TESTS -------------------------------------------- ##

    
# # Enable MPS fallback for PyTorch on macOS
# import os
# import json
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# # Import interface libraries
# from pathlib import Path
# import time
# from types import SimpleNamespace

# # Import computational libraries
# import numpy as np
# import torch
# import torch.nn.functional as F

# # Import Plotting libraries
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.cm as cm
# import matplotlib.colors as colors


# # Import project modules
# from scripts.optimisation.custom_modules import MultiSourceFIR
# from scripts.plotting.results_plot import plot_results
# from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
# from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
# from scripts.perceptual_models.llado2025.llado2025 import llado2025
# from scripts.perceptual_models.desena2020.desena2020 import desena2020
# from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
# from scripts.optimisation.regularisation import second_moment_regularisation
# from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
# from scripts.saving.save_results import saver


# def run_experiment(loss_dict_builder, run_label, target_angle):
#     repo_root = Path(__file__).resolve().parent
#     d = 60
#     env = SimpleNamespace( # Include here all variables that you want stored into config.json
#         epochs=300,
#         num_lstn_pos=1,
#         listener_batch_size=None,  # 5 mirrored pairs plus the centre listener
#         symmetric_listener_sampling=False,
#         n_samples=128,
#         target_pos=[d, float(np.tan(np.deg2rad(target_angle)) * d)],
#         selected_stim="impulse",
#         sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
#         sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
#         lr=0.001,
#         seed=46,
#         log=1,
#         fs=48000,
#     )
#     env.dtype = torch.float64
#     env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
#     env.device = torch.device(env.device_str)

# ## Initial setup
#     print(f"Using device: {env.device}")

#     # Loudspkeaker and listener positions
#     env.lstn_pos = [[0.0,0.0]]
#     base_angle_rad = np.deg2rad(30.0)
#     env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
#     env.spk_pos = [
#         [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
#         [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
#     ]
#     env.n_sources = len(env.spk_pos)

#     # Reset the seed so each run starts from the same point.
#     set_seed(seed=env.seed)

# ## Load data
#     gtfb = load_gtfb(fs=env.fs, device=env.device, dtype=env.dtype)  # Load Gamma Tone Filter Bank
#     stim = load_stim(fs=env.fs, device=env.device, dtype=env.dtype, selected_stim=env.selected_stim)  # Load stimulus
#     hrirs = load_hrirs(dtype=env.dtype, device=env.device, sofa_path=env.sofa_path, idx_path=env.sofa_idx_path) # Load HRIRs using specific location from SOFA file

# ## Initialize model
#     # Initialize sound propagation module
#     stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path, apply_distance_attenuation=True) # Load stereo anechoic propagation module

#     # Initiate perceptual models
#     loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # Llado et al. 2025 model
#     col_model = spectral_matching(loss = F.l1_loss, target = 0, stim=stim, hrirs=hrirs, target_pos=env.target_pos, fs=env.fs).to(device=env.device, dtype=env.dtype) # Spectral Matching model
#     unc_model = desena2020(loss = F.l1_loss, target = 0, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
#     regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
#     loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)

#     # Initialise filters 
#     torch.zeros((1, env.n_sources, env.n_samples), device=env.device, dtype=env.dtype)


## ---------------------------------------- DEBUGGING DELAY TEST SCRIPT ------------------------------------------- ##

# import pytest
# import torch 
# import numpy as np
# import sofar as sf

# from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

# def sofa_find(angle, sofaile):
#     # Find index of closest angle in sofaile.SourcePosition
#     sofa_angles = sofaile.SourcePosition[:,:2]%360 # N_y x (az, elv)
#     angles = np.concat((angle[:, None], np.zeros_like(angle[:, None])), axis=1)%360 # N_x x (az, elv)
#     diff = sofa_angles[:,None,:] - angles[None,:,:] # N_y x N_x x (az, elv)
#     dist = np.linalg.norm(diff, axis=-1) # N_y x N_x
#     idx = np.argmin(dist, axis=0) # N_x
#     return idx
    

# def test_propagation_delay():
#     # This test checks that the delay applied to the HRIRs in free_field_propagation is correct.

#     # Define parameters for the test
#     fs = 48000  # Sampling frequency
#     speed_of_sound = 343.0  # Speed of sound in m/s
#     lstn_pos = [[1.0,0.0], [-1.0,0.0]]
#     spk_pos = [[2.0,0.0],[-3.0,0.0]]

#     # Calculate expected delays based on distances
#     expected_delay_spk_1_lstn_1 = np.linalg.norm(np.array(spk_pos[0]) - np.array(lstn_pos[0])) / speed_of_sound # 1 m / 343 m/s  = ~0.00292 s  
#     expected_delay_spk_1_lstn_2 = np.linalg.norm(np.array(spk_pos[0]) - np.array(lstn_pos[1])) / speed_of_sound # 3 m / 343 m/s  = ~0.00875 s
#     expected_delay_spk_2_lstn_1 = np.linalg.norm(np.array(spk_pos[1]) - np.array(lstn_pos[0])) / speed_of_sound # 4 m / 343 m/s  = ~0.01166 s  
#     expected_delay_spk_2_lstn_2 = np.linalg.norm(np.array(spk_pos[1]) - np.array(lstn_pos[1])) / speed_of_sound # 2 m / 343 m/s  = ~0.00583 s   

#     expected_sample_delay_spk_1_lstn_1 = int(round(expected_delay_spk_1_lstn_1*fs)) # 0.00292 s * 48000 Hz = ~140 samples
#     expected_sample_delay_spk_1_lstn_2 = int(round(expected_delay_spk_1_lstn_2*fs)) # 0.00875 s * 48000 Hz = ~420 samples
#     expected_sample_delay_spk_2_lstn_1 = int(round(expected_delay_spk_2_lstn_1*fs)) # 0.01166 s * 48000 Hz = ~560 samples
#     expected_sample_delay_spk_2_lstn_2 = int(round(expected_delay_spk_2_lstn_2*fs)) # 0.00583 s * 48000 Hz = ~280 samples

#     # Calculate applied delays
#     applied_delay_spk_1_lstn_1 = int(round((expected_delay_spk_1_lstn_1 - np.min([expected_delay_spk_1_lstn_1, expected_delay_spk_2_lstn_1]))*fs)) # Should be 0 samples, as spk 1 is closer to lstn 1
#     applied_delay_spk_1_lstn_2 = int(round((expected_delay_spk_1_lstn_2 - np.min([expected_delay_spk_1_lstn_2, expected_delay_spk_2_lstn_2]))*fs)) # Should be ~280 samples
#     applied_delay_spk_2_lstn_1 = int(round((expected_delay_spk_2_lstn_1 - np.min([expected_delay_spk_2_lstn_1, expected_delay_spk_1_lstn_1]))*fs)) # Should be ~140 samples
#     applied_delay_spk_2_lstn_2 = int(round((expected_delay_spk_2_lstn_2 - np.min([expected_delay_spk_2_lstn_2, expected_delay_spk_1_lstn_2]))*fs)) # Should be 0 samples, as spk 2 is closer to lstn 2

#     # Create a dummy stimulus (1 second of white noise)
#     stim = torch.randn(1, 1, fs)  # (batch, n_sources, T)

#     # Extract HRIRs 
#     sofaile = sf.read_sofa("data/impulses/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
#     front_spk_idx = sofa_find(np.array([0]), sofaile)
#     back_spk_idx = sofa_find(np.array([180]), sofaile)
#     hrirs_spk_1 = sofaile.Data_IR[front_spk_idx,:,:] # HRIRs for spk 1 (closer to lstn 1)
#     hrirs_spk_2 = sofaile.Data_IR[back_spk_idx,:,:] # HRIRs for spk 2 (closer to lstn 2)

#     expected_hrirs_spk_1_lstn_1 = np.pad(hrirs_spk_1, (0, applied_delay_spk_1_lstn_1), mode='constant') # Pad with zeros at the beginning to simulate delay
#     expected_hrirs_spk_2_lstn_2 = np.pad(hrirs_spk_2, (0, applied_delay_spk_2_lstn_2), mode='constant') # Pad with zeros at the beginning to simulate delay
#     expected_hrirs_spk_1_lstn_2 = np.pad(hrirs_spk_1, (0, applied_delay_spk_1_lstn_2), mode='constant') # Pad with zeros at the beginning to simulate delay
#     expected_hrirs_spk_2_lstn_1 = np.pad(hrirs_spk_2, (0, applied_delay_spk_2_lstn_1), mode='constant') # Pad with zeros at the beginning to simulate delay

#     # Create the propagation module with distance attenuation enabled
#     propagation = BinauralFreeFieldPropagation(
#         stim=stim,
#         lstn_pos=lstn_pos,
#         spk_pos=spk_pos,
#         sofa_file_path="data/impulses/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa",
#         fs=fs,
#         apply_distance_attenuation=True
#     )


#     np.testing.assert_array_equal(propagation.sources[0].hrirs[0], expected_hrirs_spk_1_lstn_1) # Spk 1 to lstn 1 should have 0 delay
#     np.testing.assert_array_equal(propagation.sources[0].hrirs[1], expected_hrirs_spk_1_lstn_2) # Spk 1 to lstn 2 should have ~280 samples of delay
#     np.testing.assert_array_equal(propagation.sources[1].hrirs[0], expected_hrirs_spk_2_lstn_1) # Spk 2 to lstn 1 should have ~140 samples of delay
#     np.testing.assert_array_equal(propagation.sources[1].hrirs[1], expected_hrirs_spk_2_lstn_2) # Spk 2 to lstn 2 should have 0 delay


# test_propagation_delay()


    

## -------------------------------------- DEBUGGING LISTENER BATCH LOADER TEST SCRIPT ------------------------------------------- ##


# import pytest
# import torch

# from scripts.datasets.listener_dataset import get_listener_loader



# lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 100)])
# listener_batch_size = 4
# symmetric_listener_sampling = False

# listener_data_loader, listener_dataset = get_listener_loader(
#     lstn_pos, listener_batch_size, symmetric_listener_sampling
# )

# sampled_lstn_pos = torch.empty((0, 2))  

# for listener_idx, listener_pos in listener_data_loader:
#     sampled_lstn_pos = torch.cat((sampled_lstn_pos, listener_pos), dim=0)

# # sort returns (values, indices)
# sorted_vals, _ = sampled_lstn_pos.sort(dim=0)

# # comparison must use torch.equal or allclose
# assert torch.allclose(sorted_vals, lstn_pos)


# lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 101)])
# listener_batch_size = 10
# symmetric_listener_sampling = True

# listener_data_loader, _ = get_listener_loader(
#     lstn_pos, listener_batch_size, symmetric_listener_sampling
# )

# for listener_idx, listener_pos in listener_data_loader:
#     sorted_listener_pos, _ = listener_pos.sort(dim=0)

#     for i in range(len(sorted_listener_pos)):
#         assert torch.allclose(sorted_listener_pos[i], -1*sorted_listener_pos[-i-1])


# lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 101)])
# listener_batch_size = 10
# symmetric_listener_sampling = False

# listener_data_loader, _ = get_listener_loader(
#     lstn_pos, listener_batch_size, symmetric_listener_sampling
# )

# for listener_idx, listener_pos in listener_data_loader:
    
#     assert torch.allclose(lstn_pos[listener_idx], listener_pos)


## -------------------------------------- BEAMFORMING TEST SCRIPT ------------------------------------------- ##

# import numpy as np 
# import sounddevice as sd
# from pathlib import Path
# from scipy.io import wavfile
# from scipy import signal

# fs = 48000
# signal_duration_sec = 30.0  # Make the signal longer in time (e.g. 5 seconds)
# sig_len = int(fs * signal_duration_sec)
# sin = 0.01 * np.sin(2 * np.pi * 660 * np.arange(sig_len) / fs)
# print(sig_len)

# # Generate pink noise using Voss-McCartney algorithm
# import numpy as np

# def pink_noise(n: int, random_state=None) -> np.ndarray:
#     """
#     Generate pink noise (1/f noise) of length n.

#     Parameters
#     ----------
#     n : int
#         Number of samples to generate.
#     random_state : int or np.random.Generator, optional
#         Seed or generator for reproducibility.

#     Returns
#     -------
#     np.ndarray
#         Pink noise signal of length n, zero-mean and unit variance.
#     """
#     rng = np.random.default_rng(random_state)

#     # Work in the frequency domain.
#     # We need n//2 + 1 positive-frequency bins (rfft output length).
#     num_bins = n // 2 + 1

#     # Draw random complex amplitudes (white noise spectrum).
#     phases = rng.uniform(0, 2 * np.pi, num_bins)
#     magnitudes = rng.standard_normal(num_bins)
#     spectrum = magnitudes * np.exp(1j * phases)

#     # Shape the spectrum by 1/sqrt(f) so that power ∝ 1/f.
#     # Bin 0 (DC) is set to zero to keep the signal zero-mean.
#     frequencies = np.arange(num_bins, dtype=float)
#     frequencies[0] = 1.0          # avoid division by zero; will be zeroed anyway
#     spectrum /= np.sqrt(frequencies)
#     spectrum[0] = 0.0             # zero DC component

#     # Transform back to time domain and keep only the real part.
#     signal = np.fft.irfft(spectrum, n=n)

#     # Normalise to zero mean and unit variance.
#     signal -= signal.mean()
#     signal /= signal.std()

#     return signal


# stimulus = pink_noise(sig_len)  # zero pad to 5x duration
# # Apply a 500 Hz low-pass filter.
# sos = signal.butter(N=6, Wn=2000.0, btype="low", fs=fs, output="sos")
# # stimulus = signal.sosfiltfilt(sos, stimulus)
# stimulus /= (np.max(np.abs(stimulus)) / 0.1)


# audio = np.zeros((stimulus.shape[0], 4))

# audio[:, 0] = stimulus
# audio[:, 2] = -stimulus

# audio[:,1] = stimulus
# audio[:,3] = -stimulus

# required_channels = audio.shape[1]
# devices = sd.query_devices()

# # Pick the default output device first if it can play all channels,
# # otherwise fall back to the first device with enough output channels.
# default_output = sd.default.device[1]
# if default_output is not None and devices[default_output]["max_output_channels"] >= required_channels:
#     output_device = default_output
# else:
#     output_device = next(
#         (
#             idx
#             for idx, dev in enumerate(devices)
#             if dev["max_output_channels"] >= required_channels
#         ),
#         None,
#     )

# if output_device is None:
#     raise RuntimeError(
#         f"No output device with at least {required_channels} channels was found."
#     )

# sd.default.device = (None, output_device)
# print(f"Using output device {output_device}: {devices[output_device]['name']}")

# sd.play(audio.astype(np.float32), fs, device=(None, output_device))
# sd.wait()


## -------------------------------------- RANDOM LISTENING TEST SCRIPT ------------------------------------------- ##

# import numpy as np 
# import sounddevice as sd
# import scipy.io.wavfile as wav


# fs = 48000
# duration_s = 30
# burst_s = 0.1
# silence_s = 0.4
# sig_len = int(duration_s * fs)

# burst_len = int(burst_s * fs)
# silence_len = int(silence_s * fs)
# cycle_len = burst_len + silence_len
# num_cycles = int(np.ceil(sig_len / cycle_len))

# mono_signal = np.zeros(sig_len, dtype=np.float32)
# for cycle_idx in range(num_cycles):
#     start = cycle_idx * cycle_len
#     end = min(start + burst_len, sig_len)
#     mono_signal[start:end] = 0.01 * np.random.randn(end - start).astype(np.float32)

# audio = np.zeros((sig_len,4), dtype=np.float32)

# audio[:,0] = mono_signal
# audio[:,1] = mono_signal


# required_channels = audio.shape[1]
# devices = sd.query_devices()

# # Pick the default output device first if it can play all channels,
# # otherwise fall back to the first device with enough output channels.
# default_output = sd.default.device[1]
# if default_output is not None and devices[default_output]["max_output_channels"] >= required_channels:
#     output_device = default_output
# else:
#     output_device = next(
#         (
#             idx
#             for idx, dev in enumerate(devices)
#             if dev["max_output_channels"] >= required_channels
#         ),
#         None,
#     )

# if output_device is None:
#     raise RuntimeError(
#         f"No output device with at least {required_channels} channels was found."
#     )

# sd.default.device = (None, output_device)
# print(f"Using output device {output_device}: {devices[output_device]['name']}")

# sd.play(audio.astype(np.float32), fs, device=(None, output_device))
# sd.wait()









## ============================= ITD-ONLY OPTIMISATION TEST (FREQUENCY-DOMAIN DELAY) ============================== ##

# import torch
# from scripts.setup.setup import load_gtfb
# from scripts.helpers.peripheral_processing import estimate_interauralcues

# # Frequency-domain ITD optimisation:
# # - Create binaural target with known ITD.
# # - Apply differentiable phase delay to one ear.
# # - Optimise delay parameter from ITD-only loss.

# fs = 48000
# dtype = torch.float64
# device = "cpu"
# tau_IACC = 0.01
# torch.manual_seed(0)

# gtfb = load_gtfb(fs, device, dtype)

# # Mono stimulus (batch=1, listener=1, time=T).
# T = int(0.08 * fs)
# stim = torch.randn(1, 1, T, device=device, dtype=dtype)

# def apply_fractional_delay_freq(x, delay_samples, fs_hz, n_fft=None):
#     """
#     Differentiable fractional delay in frequency domain.
#     x shape: (..., T), delay_samples: scalar tensor or shape (1,)
#     """
#     if n_fft is None:
#         n_fft = x.shape[-1]

#     X = torch.fft.rfft(x, n=n_fft, dim=-1)
#     freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs_hz).to(device=x.device, dtype=x.dtype)

#     delay_seconds = delay_samples / fs_hz
#     phase = -2.0 * torch.pi * freqs * delay_seconds
#     H = torch.exp(1j * phase).reshape((1,) * (x.ndim - 1) + (-1,))

#     y = torch.fft.irfft(X * H, n=n_fft, dim=-1)
#     return y[..., : x.shape[-1]]

# # Target ITD in samples (right ear delayed).
# target_delay_samples = torch.tensor([-2.4], device=device, dtype=dtype)

# target_left = stim
# target_right = apply_fractional_delay_freq(stim, target_delay_samples, fs)

# target_bin = torch.zeros(1, 1, 2, T, device=device, dtype=dtype)
# target_bin[:, :, 0, :] = target_left
# target_bin[:, :, 1, :] = target_right

# target_cues = estimate_interauralcues(target_bin, fs, gtfb, tau_IACC=tau_IACC)
# target_itd = target_cues["itd"].detach()

# # Trainable delay applied to right ear in frequency domain.
# delay_samples = torch.tensor([0.0], device=device, dtype=dtype, requires_grad=True)
# optimiser = torch.optim.Adam([delay_samples], lr=0.5)

# n_iter = 200
# for i in range(n_iter):
#     optimiser.zero_grad()

#     pred_left = stim
#     pred_right = apply_fractional_delay_freq(stim, delay_samples, fs)

#     pred_bin = torch.zeros(1, 1, 2, T, device=device, dtype=dtype)
#     pred_bin[:, :, 0, :] = pred_left
#     pred_bin[:, :, 1, :] = pred_right

#     pred_cues = estimate_interauralcues(pred_bin, fs, gtfb, tau_IACC=tau_IACC)
#     loss_itd = torch.mean((pred_cues["itd"] - target_itd) ** 2)
#     loss_itd.backward()
#     optimiser.step()

#     if i % 25 == 0 or i == n_iter - 1:
#         print(
#             f"Iter {i:03d} | Loss(ITD): {loss_itd.item():.6e} | "
#             f"Delay estimate: {delay_samples.item():.4f} samples ({delay_samples.item()/fs*1e6:.2f} us)"
#         )

# target_delay_us = target_delay_samples.item() / fs * 1e6
# est_delay_us = delay_samples.item() / fs * 1e6
# print(f"Target delay: {target_delay_samples.item():.4f} samples ({target_delay_us:.2f} us)")
# print(f"Estimated delay: {delay_samples.item():.4f} samples ({est_delay_us:.2f} us)")

# # ============================= ITD-ONLY OPTIMISATION TEST (estimate_interauralcues) ==============================
# # FIR-filter ITD optimisation:
# # - Create binaural target with known ITD.
# # - Create binaural signals by convolving mono stimulus with left/right filters.
# # - Optimise a pair of trainable filters from ITD-only loss.

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from pathlib import Path
# from scripts.setup.setup import load_gtfb
# from scripts.helpers.peripheral_processing import estimate_interauralcues
# from scripts.helpers.dsp import causal_conv1d
# from scripts.optimisation.regularisation import second_moment_regularisation, rms_normalisation

# fs = 48000
# dtype = torch.float64
# device = "cpu"
# tau_IACC = 0.01
# torch.manual_seed(0)

# gtfb = load_gtfb(fs, device, dtype)

# T = int(0.08 * fs)
# stim = torch.randn(1, 1, T, device=device, dtype=dtype)

# def convolve_mono_with_pair_filters(x, pair_filters):
#     return causal_conv1d(x, pair_filters)

# Lf = 128
# target_filters = torch.zeros(2, 1, Lf, device=device, dtype=dtype)
# target_filters[0, 0, 55] = 1.0
# target_filters[1, 0, 124] = 1.0
# target_bin_2ch = convolve_mono_with_pair_filters(stim, target_filters)
# target_bin = target_bin_2ch.unsqueeze(1)
# target_cues = estimate_interauralcues(target_bin, fs, gtfb, tau_IACC=tau_IACC)
# target_itd = target_cues["itd"].detach()

# train_filters = torch.zeros(2, 1, Lf, device=device, dtype=dtype)
# center_tap = Lf // 2
# train_filters[0, 0, center_tap] = 1.0
# train_filters[1, 0, center_tap] = 1.0
# train_filters = torch.nn.Parameter(train_filters)
# optimiser = torch.optim.Adam([train_filters], lr=1e-3)
# lambda_m2 = 1 / 4
# reg_m2 = second_moment_regularisation(n_samples=Lf, device=device, dtype=dtype)

# n_iter = 500
# save_every = 5
# filter_history = []
# epoch_history = []
# for i in range(n_iter):
#     optimiser.zero_grad()
#     train_filters_norm = rms_normalisation(train_filters)
#     if i % save_every == 0 or i == n_iter - 1:
#         filter_history.append(train_filters_norm.detach().cpu().numpy().copy())
#         epoch_history.append(i)
#     pred_bin_2ch = convolve_mono_with_pair_filters(stim, train_filters_norm)
#     pred_bin = pred_bin_2ch.unsqueeze(1)
#     pred_cues = estimate_interauralcues(pred_bin, fs, gtfb, tau_IACC=tau_IACC)
#     loss_itd = torch.mean((pred_cues["itd"] - target_itd) ** 2)
#     loss_m2 = torch.mean(reg_m2.compute_loss(train_filters_norm))
#     # grad_itd = torch.autograd.grad(loss_itd, train_filters, retain_graph=True, create_graph=False)[0]
#     # grad_m2 = torch.autograd.grad(loss_m2, train_filters, retain_graph=True, create_graph=False)[0]
#     # grad_norm_itd = torch.norm(grad_itd, p=2)
#     # grad_norm_m2 = torch.norm(grad_m2, p=2)
#     # loss_itd_norm = loss_itd / (grad_norm_itd.detach() + 1e-12)
#     # loss_m2_norm = loss_m2 / (grad_norm_m2.detach() + 1e-12)
#     loss = loss_itd + loss_m2
#     # loss = loss_it
#     loss.backward()
#     optimiser.step()
#     if i % 25 == 0 or i == n_iter - 1:
#         pred_itd_mean_us = pred_cues["itd"].mean().item() * 1e6
#         target_itd_mean_us = target_itd.mean().item() * 1e6
#         itd_err_us = pred_itd_mean_us - target_itd_mean_us
#         # grad_cos = torch.sum(grad_itd * grad_m2) / (grad_norm_itd * grad_norm_m2 + 1e-12)
#         # grad_cos_clamped = torch.clamp(grad_cos, -1.0 + 1e-7, 1.0 - 1e-7)
#         # grad_angle_deg = torch.rad2deg(torch.acos(grad_cos_clamped))
#         print(
#             f"Iter {i:03d} | Loss(ITD): {loss_itd.item():.6e} | "
#             # f"Loss(M2): {loss_m2.item():.6e} | "
#             # f"LossN(ITD): {loss_itd_norm.item():.6e} | "
#             # f"LossN(M2): {loss_m2_norm.item():.6e} | "
#             # f"Weighted LossN(M2): {(lambda_m2 * loss_m2_norm).item():.6e} | "
#             # f"||grad ITD||: {grad_norm_itd.item():.3e} | "
#             # f"||grad M2||: {grad_norm_m2.item():.3e} | "
#             # f"cos(grad_ITD, grad_M2): {grad_cos.item():.4f} | "
#             # f"angle(grad): {grad_angle_deg.item():.2f} deg | "
#             f"Pred mean ITD: {pred_itd_mean_us:.2f} us | "
#             f"Target mean ITD: {target_itd_mean_us:.2f} us | "
#             f"Error: {itd_err_us:.2f} us"
#         )

# gif_path = Path("filter_taps_evolution.gif")
# gif_path.parent.mkdir(parents=True, exist_ok=True)
# fig_anim, ax_anim = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# line_tgt_l, = ax_anim[0].plot(target_filters[0, 0].cpu().numpy(), color="k", linewidth=1.5, label="Target Left")
# line_est_l, = ax_anim[0].plot(filter_history[0][0, 0], "--", linewidth=2, label="Estimated Left")
# line_tgt_r, = ax_anim[1].plot(target_filters[1, 0].cpu().numpy(), color="k", linewidth=1.5, label="Target Right")
# line_est_r, = ax_anim[1].plot(filter_history[0][1, 0], "--", linewidth=2, label="Estimated Right")
# ax_anim[0].set_ylabel("Amplitude")
# ax_anim[1].set_ylabel("Amplitude")
# ax_anim[1].set_xlabel("Tap Index")
# ax_anim[0].grid(True, alpha=0.3)
# ax_anim[1].grid(True, alpha=0.3)
# ax_anim[0].legend(loc="upper right")
# ax_anim[1].legend(loc="upper right")
# title = fig_anim.suptitle(f"Filter Tap Evolution - Epoch {epoch_history[0]}")
# writer = animation.PillowWriter(fps=8)
# with writer.saving(fig_anim, str(gif_path), dpi=120):
#     for frame_idx, h in enumerate(filter_history):
#         line_est_l.set_ydata(h[0, 0])
#         line_est_r.set_ydata(h[1, 0])
#         title.set_text(f"Filter Tap Evolution - Epoch {epoch_history[frame_idx]}")
#         writer.grab_frame()
# plt.close(fig_anim)
# print(f"Saved filter evolution GIF: {gif_path}")

# est_filters = rms_normalisation(train_filters).detach()
# fig_filters, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# axes[0].plot(target_filters[0, 0].cpu().numpy(), label="Target Left", linewidth=2)
# axes[0].plot(est_filters[0, 0].cpu().numpy(), "--", label="Estimated Left", linewidth=2)
# axes[0].set_ylabel("Amplitude")
# axes[0].set_title("Left Ear Filter")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# axes[1].plot(target_filters[1, 0].cpu().numpy(), label="Target Right", linewidth=2)
# axes[1].plot(est_filters[1, 0].cpu().numpy(), "--", label="Estimated Right", linewidth=2)
# axes[1].set_xlabel("Tap Index")
# axes[1].set_ylabel("Amplitude")
# axes[1].set_title("Right Ear Filter")
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
# fig_filters.tight_layout()

# final_bin_2ch = convolve_mono_with_pair_filters(stim, est_filters)
# final_bin = final_bin_2ch.unsqueeze(1)
# final_cues = estimate_interauralcues(final_bin, fs, gtfb, tau_IACC=tau_IACC)
# final_itd_mean_us = final_cues["itd"].mean().item() * 1e6
# target_itd_mean_us = target_itd.mean().item() * 1e6
# print(f"Final mean ITD: {final_itd_mean_us:.2f} us")
# print(f"Target mean ITD: {target_itd_mean_us:.2f} us")
# print(f"Final ITD error: {final_itd_mean_us - target_itd_mean_us:.2f} us")

# left_sig = final_bin_2ch[0, 0].cpu().numpy()
# right_sig = final_bin_2ch[0, 1].cpu().numpy()
# xcorr = np.correlate(left_sig, right_sig, mode="full")
# lags = np.arange(-len(left_sig) + 1, len(left_sig))
# lags_ms = 1000.0 * lags / fs
# peak_idx = np.argmax(np.abs(xcorr))
# peak_lag_samples = lags[peak_idx]
# peak_lag_ms = lags_ms[peak_idx]
# plt.figure(figsize=(8, 3.5))
# plt.plot(lags_ms, xcorr, linewidth=1.5)
# plt.axvline(peak_lag_ms, color="r", linestyle="--", label=f"Peak lag: {peak_lag_samples} samples")
# plt.xlabel("Lag (ms)")
# plt.ylabel("Cross-correlation")
# plt.title("Final Binaural Cross-Correlation (Left vs Right)")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()



# --------------------------------------------------------------------------------
# Active experiment: stereo loudspeakers + free-field propagation; HRIR-at-angle ITD/ILD target.
# --------------------------------------------------------------------------------

## ============================= STEREO PROP + HRIR TARGET (ITD / ILD, NO REG / NO NORM) ============================== ##
# Target: mono convolved with HRIR at ref_azimuth (virtual source).
# Train: MultiSourceFIR (per loudspeaker) + BinauralFreeFieldPropagation like main_stereo_rendering.py.
# Loss: mean((ITD-ITD_ref)^2) + mean((ILD-ILD_ref)^2) with no weighting, no filter normalisation, no regularisation.
# """
# FIR filter optimisation to match binaural ITD/ILD targets.

# Key features:
#   - Gradient norm balancing: ITD loss is rescaled each step so its
#     gradient norm at the filter parameters matches the ILD gradient norm,
#     preventing ILD from dominating training.
#   - Second-moment regularisation: encourages each FIR filter to remain
#     compact (sinc-like) without penalising its overall energy or pulling
#     the impulse toward a fixed tap.
#   - Detailed per-step logging of gradient norms and cosine alignments
#     so the balance between ITD and ILD can be monitored.
# """

# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import sofar as sf
# import torch
# from matplotlib import animation

# from scripts.helpers.dsp import causal_conv1d
# from scripts.helpers.miscelaneous import sofa_find
# from scripts.helpers.peripheral_processing import estimate_interauralcues
# from scripts.optimisation.custom_modules import MultiSourceFIR
# from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
# from scripts.setup.setup import load_gtfb

# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# FS = 48_000
# DTYPE = torch.float64
# DEVICE = "cpu"

# N_FILTER_TAPS = 32
# N_ITER = 500
# LR = 5e-2

# TAU_IACC = 0.01
# REF_AZIMUTH_DEG = 30.0
# SPEAKER_DISTANCE = 2.0
# SPEAKER_AZIMUTH_DEG = 30.0

# # second-moment regularisation weight (lambda)
# # keep small so it breaks filter degeneracy without biasing the delay
# LAMBDA_SM = 1e-4

# LOG_EVERY = 50
# SAVE_EVERY = 10

# REPO_ROOT = Path(__file__).resolve().parent
# SOFA_PATH = (
#     REPO_ROOT
#     / "data"
#     / "impulses"
#     / "SOFA"
#     / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
# )

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def build_geometry(azimuth_deg: float, distance: float, dtype, device):
#     """Return listener position and symmetric speaker positions."""
#     lstn_pos = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
#     az = np.deg2rad(azimuth_deg)
#     spk_pos = torch.tensor(
#         [
#             [distance * np.cos(az),  distance * np.sin(az)],
#             [distance * np.cos(az), -distance * np.sin(az)],
#         ],
#         device=device,
#         dtype=dtype,
#     )
#     return lstn_pos, spk_pos


# def second_moment_loss(filters: torch.Tensor) -> torch.Tensor:
#     """
#     Compactness regularisation via the second moment of filter energy.

#     Penalises spread of energy around the filter's own centre of mass,
#     encouraging a sinc-like impulse anywhere in the tap range without
#     pulling it toward a fixed tap or penalising total energy.

#     Args:
#         filters: shape (n_sources, n_taps)

#     Returns:
#         Scalar loss.
#     """
#     total = 0.0
#     n_taps = filters.shape[-1]
#     indices = torch.arange(n_taps, dtype=filters.dtype, device=filters.device)
#     for h in filters:
#         energy = h.pow(2)
#         total_energy = energy.sum() + 1e-12
#         mu = (indices * energy).sum() / total_energy
#         sigma2 = ((indices - mu).pow(2) * energy).sum() / total_energy
#         total = total + sigma2
#     return total / filters.shape[0]


# def grad_norm_for_loss(
#     loss: torch.Tensor,
#     params: list,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute the L2 norm of the gradient of `loss` w.r.t. `params`
#     without accumulating into .grad.

#     Returns:
#         norm:   scalar tensor
#         vector: flat concatenated gradient vector
#     """
#     grads = torch.autograd.grad(
#         loss, params, retain_graph=True, create_graph=False, allow_unused=False
#     )
#     flat = torch.cat([g.reshape(-1) for g in grads])
#     return torch.norm(flat), flat


# def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> float:
#     """Cosine similarity between two flat vectors, clamped to [-1, 1]."""
#     cos = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + eps)
#     return float(torch.clamp(cos, -1.0, 1.0).detach().cpu())


# # ---------------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------------

# torch.manual_seed(0)

# lstn_pos, spk_pos = build_geometry(SPEAKER_AZIMUTH_DEG, SPEAKER_DISTANCE, DTYPE, DEVICE)

# stereo_prop = BinauralFreeFieldPropagation(
#     fs=FS,
#     lstn_pos=lstn_pos,
#     spk_pos=spk_pos,
#     sofa_file_path=str(SOFA_PATH),
#     apply_distance_attenuation=False,
#     device=DEVICE,
#     dtype=DTYPE,
# )

# gtfb = load_gtfb(FS, DEVICE, DTYPE)

# # Build HRIR-convolved binaural target
# sofa_file = sf.read_sofa(str(SOFA_PATH))
# ref_idx = int(sofa_find(np.array([REF_AZIMUTH_DEG]), sofa_file)[0])
# hrir_weights = (
#     torch.tensor(sofa_file.Data_IR[ref_idx], device=DEVICE, dtype=DTYPE)
#     .unsqueeze(1)
# )

# T = int(0.08 * FS)
# stim = torch.randn(1, 1, T, device=DEVICE, dtype=DTYPE)

# target_bin = causal_conv1d(stim, hrir_weights).unsqueeze(1)
# with torch.no_grad():
#     target_cues = estimate_interauralcues(target_bin, FS, gtfb, tau_IACC=TAU_IACC)
# target_itd = target_cues["itd"].detach()
# target_ild = target_cues["ild"].detach()

# # Initialise filters as central impulses
# nn_model = MultiSourceFIR(n_samples=N_FILTER_TAPS, n_sources=2).to(
#     device=DEVICE, dtype=DTYPE
# )
# centre_tap = N_FILTER_TAPS // 2
# with torch.no_grad():
#     for branch in nn_model.branches:
#         branch.weight.fill_(0.0)
#         branch.weight[centre_tap] = 1.0

# unit_input = torch.ones(1, 1, device=DEVICE, dtype=DTYPE)
# listener_idx = torch.tensor([0], device=DEVICE, dtype=torch.long)
# optimiser = torch.optim.Adam(nn_model.parameters(), lr=LR)
# params_train = [p for p in nn_model.parameters() if p.requires_grad]

# # ---------------------------------------------------------------------------
# # Training loop
# # ---------------------------------------------------------------------------

# history = dict(
#     rmse_itd=[],
#     rmse_ild=[],
#     mae_itd=[],
#     mae_ild=[],
#     grad_norm_itd=[],
#     grad_norm_ild=[],
#     grad_norm_total=[],
#     cos_itd_total=[],
#     cos_ild_total=[],
#     scale_itd=[],
# )
# filter_snapshots, snapshot_steps = [], []

# for step in range(N_ITER):
#     optimiser.zero_grad()

#     spk_filters = nn_model(unit_input)
#     bin_pred = stereo_prop(stim, spk_filters, listener_idx)
#     pred_cues = estimate_interauralcues(bin_pred, FS, gtfb, tau_IACC=TAU_IACC)

#     loss_itd = torch.mean((pred_cues["itd"] - target_itd) ** 2)
#     loss_ild = torch.mean((pred_cues["ild"] - target_ild) ** 2)
#     # loss_sm  = second_moment_loss(spk_filters.squeeze(0))

#     # ------------------------------------------------------------------
#     # Gradient norm balancing
#     # Compute per-loss gradient norms at filter parameters, then scale
#     # the ITD loss so its gradient contribution matches ILD, preventing
#     # ILD from dominating filter updates throughout training.
#     # ------------------------------------------------------------------
#     n_itd, v_itd = grad_norm_for_loss(loss_itd, params_train)
#     n_ild, v_ild = grad_norm_for_loss(loss_ild, params_train)

#     # avoid division by zero at the very first step
#     scale_itd = (n_ild / (n_itd + 1e-30)).detach()

#     loss = scale_itd * loss_itd + loss_ild
#     # loss = loss_itd + loss_ild
#     # loss = loss_itd
#     loss.backward()

#     # total gradient vector for cosine diagnostics
#     v_tot = torch.cat([p.grad.reshape(-1) for p in params_train if p.grad is not None])
#     n_tot = torch.norm(v_tot)

#     # ------------------------------------------------------------------
#     # Logging
#     # ------------------------------------------------------------------
#     history["rmse_itd"].append(float(torch.sqrt(loss_itd).detach()))
#     history["rmse_ild"].append(float(torch.sqrt(loss_ild).detach()))
#     history["mae_itd"].append(float(torch.mean(torch.abs(pred_cues["itd"] - target_itd)).detach()))
#     history["mae_ild"].append(float(torch.mean(torch.abs(pred_cues["ild"] - target_ild)).detach()))
#     history["grad_norm_itd"].append(float(n_itd.detach()))
#     history["grad_norm_ild"].append(float(n_ild.detach()))
#     history["grad_norm_total"].append(float(n_tot.detach()))
#     history["cos_itd_total"].append(cosine_similarity(v_itd, v_tot))
#     history["cos_ild_total"].append(cosine_similarity(v_ild, v_tot))
#     history["scale_itd"].append(float(scale_itd))

#     if step % SAVE_EVERY == 0 or step == N_ITER - 1:
#         filter_snapshots.append(spk_filters.detach().cpu().numpy().copy())
#         snapshot_steps.append(step)

#     optimiser.step()

#     if step % LOG_EVERY == 0 or step == N_ITER - 1:
#         print(
#             f"step {step:4d} | "
#             f"RMSE ITD {history['rmse_itd'][-1]*1e6:7.2f} µs  "
#             f"RMSE ILD {history['rmse_ild'][-1]:.3f} dB | "
#             f"‖g_itd‖ {history['grad_norm_itd'][-1]:.3e}  "
#             f"‖g_ild‖ {history['grad_norm_ild'][-1]:.3e}  "
#             f"scale_itd {history['scale_itd'][-1]:.3e} | "
#             f"cos(g_itd,g) {history['cos_itd_total'][-1]:+.3f}  "
#             f"cos(g_ild,g) {history['cos_ild_total'][-1]:+.3f}"
#         )

# # ---------------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------------

# steps = np.arange(N_ITER)

# # — RMSE curves —
# fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
# axes[0].plot(steps, np.array(history["rmse_itd"]) * 1e6, color="C0", linewidth=1.5)
# axes[0].set_ylabel("RMSE ITD (µs)")
# axes[0].set_title("Cue RMSE vs optimisation step")
# axes[0].grid(True, alpha=0.3)
# axes[1].plot(steps, history["rmse_ild"], color="C1", linewidth=1.5)
# axes[1].set_ylabel("RMSE ILD (dB)")
# axes[1].set_xlabel("Step")
# axes[1].grid(True, alpha=0.3)
# fig.tight_layout()

# # — Gradient norms + ITD scale factor —
# fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
# axes[0].plot(steps, history["grad_norm_itd"],   color="C0", linewidth=1.5, label="‖∂L_itd/∂θ‖ (raw)")
# axes[0].plot(steps, history["grad_norm_ild"],   color="C1", linewidth=1.5, label="‖∂L_ild/∂θ‖")
# axes[0].plot(steps, history["grad_norm_total"], color="k",  linewidth=1.0, linestyle=":", label="‖∂L/∂θ‖")
# axes[0].set_yscale("log")
# axes[0].set_ylabel("L2 norm")
# axes[0].set_title("Filter-space gradient norms")
# axes[0].legend(loc="best", fontsize=9)
# axes[0].grid(True, alpha=0.3)
# axes[1].plot(steps, history["scale_itd"], color="C2", linewidth=1.5)
# axes[1].axhline(1.0, color="0.6", linewidth=0.8, linestyle="--")
# axes[1].set_ylabel("ITD scale factor")
# axes[1].set_xlabel("Step")
# axes[1].set_title("Adaptive ITD gradient scale (‖g_ild‖ / ‖g_itd‖)")
# axes[1].grid(True, alpha=0.3)
# fig.tight_layout()

# # — Cosine alignment —
# fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
# for ax, key, label, color in zip(
#     axes,
#     ["cos_itd_total", "cos_ild_total"],
#     ["cos(g_itd, g)", "cos(g_ild, g)"],
#     ["C0", "C1"],
# ):
#     ax.plot(steps, history[key], color=color, linewidth=1.5)
#     ax.axhline(0.0, color="0.5", linewidth=0.8)
#     ax.set_ylim(-1.05, 1.05)
#     ax.set_ylabel(label)
#     ax.grid(True, alpha=0.3)
# axes[0].set_title("Alignment of partial gradients with full update direction")
# axes[1].set_xlabel("Step")
# fig.tight_layout()

# # — Per-band final errors —
# with torch.no_grad():
#     final_bin = stereo_prop(stim, nn_model(unit_input), listener_idx)
# final_cues = estimate_interauralcues(final_bin, FS, gtfb, tau_IACC=TAU_IACC)
# n_bands = int(target_itd.shape[-1])
# bands = np.arange(n_bands)
# itd_err = (final_cues["itd"] - target_itd).detach().cpu().numpy().reshape(-1)
# ild_err = (final_cues["ild"] - target_ild).detach().cpu().numpy().reshape(-1)

# fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
# axes[0].bar(bands, itd_err * 1e6, color="C0")
# axes[0].set_ylabel("ITD error (µs)")
# axes[0].set_title(f"Final per-band errors  (target HRIR @ {REF_AZIMUTH_DEG:g}°)")
# axes[0].grid(True, alpha=0.3)
# axes[1].bar(bands, ild_err, color="C1")
# axes[1].set_ylabel("ILD error (dB)")
# axes[1].set_xlabel("Critical band index")
# axes[1].grid(True, alpha=0.3)
# fig.tight_layout()

# # — FIR evolution GIF —
# gif_path = Path("stereo_fir_evolution.gif")
# fig_anim, ax_anim = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# line0, = ax_anim[0].plot(filter_snapshots[0][0, 0], linewidth=1.8, label="Source 0 FIR")
# line1, = ax_anim[1].plot(filter_snapshots[0][0, 1], linewidth=1.8, label="Source 1 FIR")
# for ax in ax_anim:
#     ax.set_ylabel("Amplitude")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="upper right", fontsize=8)
# ax_anim[1].set_xlabel("Tap index")
# title_anim = fig_anim.suptitle(f"FIR filters — step {snapshot_steps[0]}")
# writer = animation.PillowWriter(fps=8)
# with writer.saving(fig_anim, str(gif_path), dpi=120):
#     for frame_idx, hf in enumerate(filter_snapshots):
#         line0.set_ydata(hf[0, 0])
#         line1.set_ydata(hf[0, 1])
#         title_anim.set_text(f"FIR filters — step {snapshot_steps[frame_idx]}")
#         writer.grab_frame()
# plt.close(fig_anim)
# print(f"Saved FIR evolution GIF → {gif_path}")

# plt.show()


## ============================= PAIR DIRECTIVITY VS LOUDSPEAKER SPACING ============================== ##
# Directivity computation matches plot_pair_directivity_polar in scripts/plotting/results_plot.py.
# Geometry: driver A at +y/2, driver B at -y/2 (center-to-center spacing along y).
# Observation circle: radius _RADIUS_M centred on the pair midpoint; polar 0 rad = +x.

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# _repo_root = Path(__file__).resolve().parent
# _FILTER_CSV_A = _repo_root / "data" / "miscellaneous" / "estimated_rodenas_driver_A_mid.csv"
# _FILTER_CSV_B = _repo_root / "data" / "miscellaneous" / "estimated_rodenas_driver_B_mid.csv"

# _DIR_FS = 48_000
# _DIR_C = 343.0
# _SPEAKER_X_M = 0.0  # common x for both drivers; does not affect pattern (circle follows midpoint)
# # Rodenas spacing search (centre-to-centre, y-axis): 1–20 mm in 1 mm steps.
# _SPACING_SEARCH_START_MM = 10
# _SPACING_SEARCH_STOP_MM = 500
# _SPACING_SEARCH_STEP_MM = 10
# # Fewer curves on polar plots (search still uses every 1 mm step)
# _N_SPACINGS_PLOT = 12
# _RADIUS_M = 10.0
# _FREQS_EVAL_HZ = (250, 500, 1000, 2000, 4000)
# _N_ANGLES = 720
# _PLOT_FREQ_HZ = 1000.0
# # Peak-normalised level axis (dB); same on all directivity figures for cross-run comparison.
# _DIR_DB_YMIN = -25.0
# _DIR_DB_YMAX = 0.5
# _GROUP_INDICES = np.array([0, 1], dtype=int)

# _h_a = np.loadtxt(_FILTER_CSV_A, dtype=float).ravel()
# _h_b = np.loadtxt(_FILTER_CSV_B, dtype=float).ravel()
# if _h_a.shape != _h_b.shape:
#     raise ValueError(f"Filter CSV tap counts differ: A={_h_a.shape}, B={_h_b.shape}")
# _dir_est_filters = np.stack([_h_b, _h_a], axis=0)  # ch0 at +y/2, ch1 at -y/2

# _spacing_search_m = (
#     np.arange(
#         _SPACING_SEARCH_START_MM,
#         _SPACING_SEARCH_STOP_MM + _SPACING_SEARCH_STEP_MM,
#         _SPACING_SEARCH_STEP_MM,
#     )
#     / 1000.0
# )
# _spacing_plot_m = np.linspace(
#     _spacing_search_m[0], _spacing_search_m[-1], _N_SPACINGS_PLOT
# )
# _dir_results = {}
# _eps_dir = 1e-12

# # Spacing-invariant FFT setup (same filters for every spacing)
# _dir_filters = np.atleast_2d(np.asarray(_dir_est_filters, dtype=float))
# _dir_h_group = _dir_filters[_GROUP_INDICES]
# _dir_n_taps = _dir_filters.shape[1]
# _dir_theta = np.linspace(0.0, 2.0 * np.pi, int(_N_ANGLES), endpoint=False)
# _dir_freqs_hz = [float(f) for f in _FREQS_EVAL_HZ]
# _dir_n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, _dir_n_taps)))))
# _dir_freqs = np.fft.rfftfreq(_dir_n_fft, d=1.0 / _DIR_FS)
# _dir_omega = 2.0 * np.pi * _dir_freqs
# _dir_H_group = np.fft.rfft(_dir_h_group, n=_dir_n_fft, axis=1)
# _dir_freq_bin_idx = [int(np.argmin(np.abs(_dir_freqs - f_hz))) for f_hz in _dir_freqs_hz]

# for _spacing_m in _spacing_plot_m:
#     _spk_pos = np.array(
#         [
#             [_SPEAKER_X_M, 0.5 * _spacing_m],
#             [_SPEAKER_X_M, -0.5 * _spacing_m],
#         ],
#         dtype=float,
#     )
#     _group_mid = np.mean(_spk_pos[_GROUP_INDICES], axis=0)
#     _p_group = _spk_pos[_GROUP_INDICES]
#     _mag2 = np.zeros((len(_dir_freqs_hz), len(_dir_theta)), dtype=float)
#     for k, ang in enumerate(_dir_theta):
#         _obs = _group_mid + np.array(
#             [_RADIUS_M * np.cos(ang), _RADIUS_M * np.sin(ang)], dtype=float
#         )
#         _tau_group = np.linalg.norm(_obs - _p_group, axis=1) / _DIR_C
#         _phase_group = np.exp(-1j * np.outer(_tau_group, _dir_omega))
#         _H_sum = np.sum(_dir_H_group * _phase_group, axis=0)
#         for fi, bi in enumerate(_dir_freq_bin_idx):
#             _mag2[fi, k] = np.abs(_H_sum[bi]) ** 2
#     _log_mean_db = 10.0 * np.log10(np.mean(_mag2, axis=0) + _eps_dir)
#     _dir_results[float(_spacing_m)] = dict(
#         theta=_dir_theta,
#         freqs_hz=_dir_freqs_hz,
#         mag2_by_freq=_mag2,
#         log_mean_mag2_db=_log_mean_db,
#         spk_pos=_spk_pos,
#     )

# _freq_idx = int(np.argmin(np.abs(np.array(_FREQS_EVAL_HZ) - _PLOT_FREQ_HZ)))


# def _db_at_nearest_angle_deg(sample_angles_deg, levels_db, target_angle_deg):
#     sample_angles_deg = np.asarray(sample_angles_deg, dtype=float).ravel()
#     levels_db = np.asarray(levels_db, dtype=float).ravel()
#     target_wrapped = float(np.mod(target_angle_deg, 360.0))
#     samples_wrapped = np.mod(sample_angles_deg, 360.0)
#     separation_deg = np.abs((samples_wrapped - target_wrapped + 180.0) % 360.0 - 180.0)
#     return float(levels_db[int(np.argmin(separation_deg))])


# def _sample_dir_db_at_angles_deg(theta_rad, dir_db, query_angles_deg):
#     sample_angles_deg = np.rad2deg(theta_rad) % 360.0
#     return np.array(
#         [_db_at_nearest_angle_deg(sample_angles_deg, dir_db, q) for q in query_angles_deg],
#         dtype=float,
#     )


# def _peak_aligned_rmse_db(est_db, ref_db):
#     est_n = est_db - np.max(est_db)
#     ref_n = ref_db - np.max(ref_db)
#     return float(np.sqrt(np.mean((est_n - ref_n) ** 2)))


# def _peak_normalize_db(levels_db):
#     levels_db = np.asarray(levels_db, dtype=float)
#     return levels_db - np.max(levels_db)


# def _compute_spacing_result(spacing_m, est_filters=None):
#     """Directivity for one spacing; est_filters shape (2, n_taps), ch0 at +y/2."""
#     filters = np.atleast_2d(
#         _dir_est_filters if est_filters is None else np.asarray(est_filters, dtype=float)
#     )
#     h_group = filters[_GROUP_INDICES]
#     h_group_fft = np.fft.rfft(h_group, n=_dir_n_fft, axis=1)
#     spk_pos = np.array(
#         [
#             [_SPEAKER_X_M, 0.5 * spacing_m],
#             [_SPEAKER_X_M, -0.5 * spacing_m],
#         ],
#         dtype=float,
#     )
#     group_mid = np.mean(spk_pos[_GROUP_INDICES], axis=0)
#     p_group = spk_pos[_GROUP_INDICES]
#     mag2 = np.zeros((len(_dir_freqs_hz), len(_dir_theta)), dtype=float)
#     for k, ang in enumerate(_dir_theta):
#         obs = group_mid + np.array(
#             [_RADIUS_M * np.cos(ang), _RADIUS_M * np.sin(ang)], dtype=float
#         )
#         tau_group = np.linalg.norm(obs - p_group, axis=1) / _DIR_C
#         phase_group = np.exp(-1j * np.outer(tau_group, _dir_omega))
#         h_sum = np.sum(h_group_fft * phase_group, axis=0)
#         for fi, bi in enumerate(_dir_freq_bin_idx):
#             mag2[fi, k] = np.abs(h_sum[bi]) ** 2
#     return dict(
#         theta=_dir_theta,
#         freqs_hz=_dir_freqs_hz,
#         mag2_by_freq=mag2,
#         log_mean_mag2_db=10.0 * np.log10(np.mean(mag2, axis=0) + _eps_dir),
#         spk_pos=spk_pos,
#     )


# # Sanity check: swapping filters A↔B mirrors |P| about the x-axis (θ → −θ).
# _FLIP_CHECK_SPACING_M = float(_spacing_search_m[len(_spacing_search_m) // 2])
# _FLIP_CHECK_FILTERS_SWAP = np.flip(_dir_est_filters, axis=0)
# _flip_res_orig = _compute_spacing_result(_FLIP_CHECK_SPACING_M, est_filters=_dir_est_filters)
# _flip_res_swap = _compute_spacing_result(_FLIP_CHECK_SPACING_M, est_filters=_FLIP_CHECK_FILTERS_SWAP)
# _n_theta = _flip_res_orig["mag2_by_freq"].shape[1]
# _flip_mirror_idx = (_n_theta - np.arange(_n_theta)) % _n_theta
# _FLIP_MAG2_RTOL = 1e-9
# _FLIP_MAG2_ATOL = 1e-18
# print("\n--- Filter swap mirror check (|P_swap(theta)| vs |P_orig(-theta)|) ---")
# print(
#     f"Test spacing: {1000 * _FLIP_CHECK_SPACING_M:.0f} mm; "
#     f"orig = [A,B] on [+y,-y], swap = [B,A]"
# )
# _flip_all_ok = True
# for _fi, _f_hz in enumerate(_dir_freqs_hz):
#     _mag_orig = _flip_res_orig["mag2_by_freq"][_fi]
#     _mag_swap = _flip_res_swap["mag2_by_freq"][_fi]
#     _mag_orig_mirrored = _mag_orig[_flip_mirror_idx]
#     _abs_err = np.max(np.abs(_mag_swap - _mag_orig_mirrored))
#     _scale = float(np.max(_mag_orig)) + _eps_dir
#     _rel_err = _abs_err / _scale
#     _ok = np.allclose(_mag_swap, _mag_orig_mirrored, rtol=_FLIP_MAG2_RTOL, atol=_FLIP_MAG2_ATOL)
#     _flip_all_ok &= _ok
#     print(
#         f"  {_f_hz:5.0f} Hz: max |dmag2| = {_abs_err:.3e} "
#         f"(rel {_rel_err:.3e})  {'PASS' if _ok else 'FAIL'}"
#     )
# if not _flip_all_ok:
#     raise RuntimeError(
#         "Filter swap did not mirror directivity as expected; "
#         "check speaker positions vs filter channel order."
#     )
# print("Filter swap mirror check: all evaluated frequencies PASS.")

# # Compare 1 kHz patterns to Rodenas Fig. 12 on 0–60° (CSV: angle_deg, level_db).
# _RODENAS_CSV = _repo_root / "data" / "miscellaneous" / "rodenas2003_fig12_1000Hz.csv"
# _RODENAS_ANGLE_MIN_DEG = 0.0
# _RODENAS_ANGLE_MAX_DEG = 60.0
# _rodenas_pts = np.loadtxt(_RODENAS_CSV, delimiter=",", skiprows=1)
# _rodenas_mask = (
#     (_rodenas_pts[:, 0] >= _RODENAS_ANGLE_MIN_DEG)
#     & (_rodenas_pts[:, 0] <= _RODENAS_ANGLE_MAX_DEG)
# )
# _rodenas_angles_deg = _rodenas_pts[_rodenas_mask, 0]
# _rodenas_db = _rodenas_pts[_rodenas_mask, 1]
# _arc_plot_deg = np.linspace(_RODENAS_ANGLE_MIN_DEG, _RODENAS_ANGLE_MAX_DEG, 241)


# def _spacing_directivity_result(spacing_m):
#     key = float(spacing_m)
#     if key not in _dir_results:
#         _dir_results[key] = _compute_spacing_result(spacing_m)
#     return _dir_results[key]


# _match_scores = {}
# for _spacing_m in _spacing_search_m:
#     _res = _spacing_directivity_result(_spacing_m)
#     _dir_db_1k = 10.0 * np.log10(_res["mag2_by_freq"][_freq_idx] + _eps_dir)
#     _est_db = _sample_dir_db_at_angles_deg(_res["theta"], _dir_db_1k, _rodenas_angles_deg)
#     _match_scores[float(_spacing_m)] = {
#         "rmse_peak_aligned_db": _peak_aligned_rmse_db(_est_db, _rodenas_db),
#         "rmse_raw_db": float(np.sqrt(np.mean((_est_db - _rodenas_db) ** 2))),
#         "est_db_at_ref_angles": _est_db,
#         "est_db_arc": _sample_dir_db_at_angles_deg(_res["theta"], _dir_db_1k, _arc_plot_deg),
#     }

# _best_spacing_m = min(_match_scores, key=lambda s: _match_scores[s]["rmse_peak_aligned_db"])
# _best_rmse_peak_aligned_db = _match_scores[_best_spacing_m]["rmse_peak_aligned_db"]

# print("\n--- Rodenas 2003 Fig. 12 @ 1 kHz comparison (0–60 deg only) ---")
# print("Reference CSV levels are dB; azimuth column is 0–60 deg (no polar remapping).")
# print(
#     f"Spacing search: {_SPACING_SEARCH_START_MM}–{_SPACING_SEARCH_STOP_MM} mm "
#     f"in {_SPACING_SEARCH_STEP_MM} mm steps ({len(_spacing_search_m)} spacings)"
# )
# print(f"{'spacing_mm':>10}  {'RMSE (peak-aligned)':>20}  {'RMSE (raw dB)':>14}")
# for _spacing_m in sorted(_match_scores):
#     if _spacing_m not in {float(s) for s in _spacing_search_m}:
#         continue
#     _scores = _match_scores[_spacing_m]
#     print(
#         f"{int(round(1000 * _spacing_m)):>10}  "
#         f"{_scores['rmse_peak_aligned_db']:>20.3f}  "
#         f"{_scores['rmse_raw_db']:>14.3f}"
#     )
# print(
#     f"\nBest match: {1000 * _best_spacing_m:.0f} mm "
#     f"(peak-aligned RMSE = {_best_rmse_peak_aligned_db:.3f} dB)"
# )
# _spacing_directivity_result(_best_spacing_m)

# _rodenas_db_peak_norm = _peak_normalize_db(_rodenas_db)
# _plot_keys = sorted({float(s) for s in _spacing_plot_m} | {float(_best_spacing_m)})

# # Peak-normalised levels on 0–60° (fixed dB axis below for cross-run comparison).
# for _spacing_m in _plot_keys:
#     _key = float(_spacing_m)
#     if "est_db_arc" not in _match_scores.get(_key, {}):
#         _res = _spacing_directivity_result(_spacing_m)
#         _dir_db_1k = 10.0 * np.log10(_res["mag2_by_freq"][_freq_idx] + _eps_dir)
#         _arc = _peak_normalize_db(_sample_dir_db_at_angles_deg(_res["theta"], _dir_db_1k, _arc_plot_deg))
#     else:
#         _arc = _peak_normalize_db(_match_scores[_key]["est_db_arc"])
#     _match_scores.setdefault(_key, {})["est_db_arc_plot"] = _arc
# _y_floor = _DIR_DB_YMIN
# _y_ceiling = _DIR_DB_YMAX

# # Cartesian: compare peak-normalised directivity on 0–60°.
# _fig_arc, _ax_arc = plt.subplots(figsize=(9, 5))
# for _spacing_m in _plot_keys:
#     _key = float(_spacing_m)
#     _arc_norm = _match_scores[_key]["est_db_arc_plot"]
#     _ax_arc.plot(
#         _arc_plot_deg,
#         _arc_norm,
#         linewidth=3.0 if np.isclose(_spacing_m, _best_spacing_m) else 1.2,
#         label=f"{int(round(1000 * _spacing_m))} mm",
#         zorder=4 if np.isclose(_spacing_m, _best_spacing_m) else 2,
#     )
# _ax_arc.scatter(
#     _rodenas_angles_deg,
#     _rodenas_db_peak_norm,
#     color="k",
#     s=40,
#     marker="o",
#     label="Rodenas 2003 (1 kHz)",
#     zorder=5,
# )
# _ax_arc.set_xlim(_RODENAS_ANGLE_MIN_DEG, _RODENAS_ANGLE_MAX_DEG)
# _ax_arc.set_ylim(_y_floor, _y_ceiling)
# _ax_arc.set_xlabel("Azimuth (deg)")
# _ax_arc.set_ylabel("Level (dB, peak normalised to 0 dB)")
# _ax_arc.set_title(
#     f"Directivity @ {int(round(_PLOT_FREQ_HZ))} Hz on 0–60° "
#     f"(best spacing {1000 * _best_spacing_m:.0f} mm, RMSE={_best_rmse_peak_aligned_db:.2f} dB)"
# )
# _ax_arc.grid(True, alpha=0.3)
# _ax_arc.legend(loc="best", fontsize=8)

# # Polar zoom 0–60° (same peak-normalised curves).
# _fig_polar_arc, _ax_polar_arc = plt.subplots(figsize=(8.5, 6), subplot_kw={"projection": "polar"})
# for _spacing_m in _plot_keys:
#     _key = float(_spacing_m)
#     _ax_polar_arc.plot(
#         np.deg2rad(_arc_plot_deg),
#         _match_scores[_key]["est_db_arc_plot"],
#         linewidth=3.0 if np.isclose(_spacing_m, _best_spacing_m) else 1.2,
#         label=f"{int(round(1000 * _spacing_m))} mm",
#     )
# _ax_polar_arc.scatter(
#     np.deg2rad(_rodenas_angles_deg),
#     _rodenas_db_peak_norm,
#     color="k",
#     s=32,
#     marker="o",
#     label="Rodenas 2003 (1 kHz)",
#     zorder=5,
# )
# _ax_polar_arc.set_thetamin(_RODENAS_ANGLE_MIN_DEG)
# _ax_polar_arc.set_thetamax(_RODENAS_ANGLE_MAX_DEG)
# _ax_polar_arc.set_ylim(_y_floor, _y_ceiling)
# _ax_polar_arc.set_title(
#     f"Polar view, 0–60° @ {int(round(_PLOT_FREQ_HZ))} Hz (peak normalised)"
# )
# _fig_polar_arc.subplots_adjust(right=0.72)
# _ax_polar_arc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
# _ax_polar_arc.grid(True)

# # Full-circle polar (optional context) at 1 kHz.
# _fig_1k, _ax_1k = plt.subplots(figsize=(8.5, 6), subplot_kw={"projection": "polar"})
# for _spacing_m in _plot_keys:
#     _res = _dir_results[float(_spacing_m)]
#     _dir_db = _peak_normalize_db(10.0 * np.log10(_res["mag2_by_freq"][_freq_idx] + _eps_dir))
#     _ax_1k.plot(
#         _res["theta"],
#         _dir_db,
#         linewidth=3.0 if np.isclose(_spacing_m, _best_spacing_m) else 1.2,
#         label=f"{int(round(1000 * _spacing_m))} mm",
#     )
# _ax_1k.scatter(
#     np.deg2rad(_rodenas_angles_deg),
#     _rodenas_db_peak_norm,
#     color="k",
#     s=28,
#     marker="o",
#     label="Rodenas 0–60°",
#     zorder=5,
# )
# _ax_1k.set_ylim(_y_floor, _y_ceiling)
# _ax_1k.set_title(f"Full circle @ {int(round(_PLOT_FREQ_HZ))} Hz (peak normalised)")
# _fig_1k.subplots_adjust(right=0.72)
# _ax_1k.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
# _ax_1k.grid(True)

# print(
#     f"\nPair directivity: filters shape {_dir_est_filters.shape}, "
#     f"{len(_plot_keys)} curves on polar plots (best spacing always included)"
# )
# print(f"Plot level axis (peak-normalised dB): [{_DIR_DB_YMIN:.1f}, {_DIR_DB_YMAX:.1f}]")
# plt.show()


# ## ==================================== GRID SEARCH ICLD/ICTD EFFECT ON ILD/ITD ====================== ##

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sofar as sf
import torch
from matplotlib import animation
import matplotlib.pyplot as plt

from scripts.helpers.peripheral_processing import estimate_interauralcues
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.setup.setup import load_gtfb

device = 'cpu'
dtype = torch.float64
fs = 48000

lstn_pos = torch.tensor([[0.0, 0.0]], device = device, dtype = dtype)
base_angle_rad = np.deg2rad(30.0)
speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
spk_pos = [
    [speaker_distance * np.cos(base_angle_rad), speaker_distance * np.sin(base_angle_rad)],
    [speaker_distance * np.cos(base_angle_rad), -speaker_distance * np.sin(base_angle_rad)],
]
spk_pos = torch.tensor(spk_pos, device = device, dtype = dtype)
n_sources = len(spk_pos)

repo_root = Path(__file__).resolve().parent
sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
gtfb = load_gtfb(fs, device = device, dtype = dtype)

stereo_prop = BinauralFreeFieldPropagation(fs=fs, lstn_pos=lstn_pos, spk_pos=spk_pos, sofa_file_path=sofa_path, device=device, dtype=dtype) 

ICLD = torch.arange(-20,21,1, device = device, dtype = dtype)
ICTD = torch.arange(-1,1.1, 0.1, device = device, dtype = dtype)*0.001

stim = torch.randn(480, device = device, dtype = dtype)
stim = torch.cat((torch.zeros(48, device= device, dtype = dtype), 
                stim,
                torch.zeros(48, device= device, dtype = dtype))).view(1,1,-1)

base_filter = torch.zeros(97, device = device, dtype = dtype).view(1,1,-1)
base_filter[0,0,48] = 1.0

def apply_delay_freq(x, delay_seconds, fs_hz, n_fft=None):
    """
    Differentiable delay in frequency domain.
    x shape: (..., T), delay_samples: scalar tensor or shape (1,)
    """
    if n_fft is None:
        n_fft = x.shape[-1]

    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs_hz).to(device=x.device, dtype=x.dtype)

    phase = -2.0 * torch.pi * freqs * delay_seconds
    H = torch.exp(1j * phase).reshape((1,) * (x.ndim - 1) + (-1,))

    y = torch.fft.irfft(X * H, n=n_fft, dim=-1)
    return y[..., : x.shape[-1]]

saved_ild = torch.zeros(len(ICLD), len(ICTD), len(gtfb))
saved_itd = torch.zeros(len(ICLD), len(ICTD), len(gtfb))

for i in range(len(ICLD)):
    for j in range(len(ICTD)):

        right_gain = torch.sqrt(1/(1+ 10**(ICLD[i]/10)))
        left_gain = right_gain*10**(ICLD[i]/20)

        left_filter = left_gain * apply_delay_freq(base_filter, -ICTD[j], fs)
        right_filter = right_gain * base_filter

        filters = torch.cat((left_filter, right_filter), dim = 1)

        bin_stim = stereo_prop(stim, filters, torch.tensor([0], device = device, dtype = torch.long))

        interauralcues = estimate_interauralcues(bin_stim, fs, gtfb)

        saved_ild[i,j] = interauralcues['ild'][0,0]
        saved_itd[i,j] = interauralcues['itd'][0,0]

saved_ild = -saved_ild.detach().numpy()  # left − right: positive = louder left ear
saved_itd = saved_itd.detach().numpy() * 1e3  # convert to ms; positive = left leads

icld_vals = np.arange(-20, 21, 1)
ictd_vals_ms = np.arange(-1, 1.1, 0.1)
extent = [ictd_vals_ms[0], ictd_vals_ms[-1], icld_vals[0], icld_vals[-1]]

n_bands = saved_ild.shape[2]
n_cols = 4
n_rows = int(np.ceil(n_bands / n_cols))


def plot_cue_grid(data, title, cbar_label, cmap="viridis"):
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.4, n_rows * 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    vmin, vmax = np.min(data), np.max(data)
    last_im = None

    for band_idx, ax in enumerate(axes):
        if band_idx >= n_bands:
            ax.set_visible(False)
            continue

        last_im = ax.imshow(
            data[:, :, band_idx],
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(f"Band {band_idx + 1}", fontsize=9)

    for ax in axes[:n_bands]:
        ax.label_outer()

    fig.supxlabel("Applied ICTD [ms] (+ earlier left spk.)")
    fig.supylabel("Applied ICLD [dB] (+ louder left spk.)")
    fig.colorbar(last_im, ax=axes[:n_bands], shrink=0.85, pad=0.02, label=cbar_label)
    fig.suptitle(title, fontsize=12)
    return fig


plot_cue_grid(saved_ild, "Estimated ILD across ICLD/ICTD grid", "Estimated ILD [dB] (left − right)")
plot_cue_grid(saved_itd, "Estimated ITD across ICLD/ICTD grid", "Estimated ITD [ms] (left leads +)", cmap="magma")

plt.show()


## ================= RANDOM PHASE LISTENING ================= ##

# import numpy as np
# from pathlib import Path
# from scipy.io import wavfile
# from scipy import signal as scipy_signal
# import sounddevice as sd
# import pygame

# _FS = 48_000
# _REPO_ROOT = Path(__file__).resolve().parent
# _STIMULUS_PATHS = {
#     "male": _REPO_ROOT / "data" / "stimulus" / "EBU_MaleSpeech.wav",
#     "female": _REPO_ROOT / "data" / "stimulus" / "EBU_FemaleSpeech.wav",
#     "pop": _REPO_ROOT / "data" / "stimulus" / "EBU_PopMusic.wav",
# }


# _PHASE_RNG = np.random.default_rng(0)
# _SCRAMBLE_AMOUNT = 1.0
# _SCRAMBLE_STEP = 0.05
# _phase_rand_unit = None


# def _rebuild_scrambled(new_random_phase=False):
#     """Blend original and random phase; 0 = original, 1 = fully random (|X| kept)."""
#     global _phase_random, _difference, _phase_rand_unit

#     spectrum = np.fft.rfft(_original)
#     n_bins = spectrum.shape[0]
#     if (
#         new_random_phase
#         or _phase_rand_unit is None
#         or _phase_rand_unit.shape[0] != n_bins
#     ):
#         _phase_rand_unit = np.exp(
#             1j * _PHASE_RNG.uniform(-np.pi, np.pi, size=n_bins)
#         )

#     scramble = float(np.clip(_SCRAMBLE_AMOUNT, 0.0, 1.0))
#     if scramble == 0.0:
#         _phase_random = _original.copy()
#     else:
#         mag = np.abs(spectrum)
#         z_orig = np.exp(1j * np.angle(spectrum))
#         z_mix = (1.0 - scramble) * z_orig + scramble * _phase_rand_unit
#         _phase_random = np.fft.irfft(mag * np.exp(1j * np.angle(z_mix)), n=len(_original))[
#             : len(_original)
#         ]

#     _difference = _original - _phase_random


# def _select_stimulus(name):
#     global _selected, _original
#     _selected = name
#     _original = _stimuli[_selected]
#     _rebuild_scrambled(new_random_phase=True)


# def _set_scramble_amount(amount):
#     global _SCRAMBLE_AMOUNT
#     _SCRAMBLE_AMOUNT = float(np.clip(amount, 0.0, 1.0))
#     _rebuild_scrambled(new_random_phase=False)


# def _nudge_scramble(delta):
#     _set_scramble_amount(_SCRAMBLE_AMOUNT + delta)


# def _load_mono_speech(wav_path, fs=_FS, segment_start_s=0.5, segment_duration_s=4.0):
#     src_fs, audio = wavfile.read(wav_path)
#     audio = np.asarray(audio)
#     if audio.ndim > 1:
#         audio = np.mean(audio, axis=1)
#     if np.issubdtype(audio.dtype, np.integer):
#         audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max)
#     else:
#         audio = audio.astype(np.float64)

#     if src_fs != fs:
#         target_len = int(round(len(audio) * fs / src_fs))
#         audio = scipy_signal.resample(audio, target_len)

#     start = int(round(segment_start_s * fs))
#     stop = start + int(round(segment_duration_s * fs))
#     audio = audio[start:stop]
#     peak = np.max(np.abs(audio))
#     if peak > 0:
#         audio = audio / (peak / 0.9)
#     return audio.astype(np.float64)


# def _normalize_for_playback(x, target_peak=0.9):
#     x = np.asarray(x, dtype=np.float64).flatten()
#     peak = np.max(np.abs(x))
#     if peak > 0:
#         x = x * (target_peak / peak)
#     return x.astype(np.float32)


# def _play_mono(audio, fs=_FS):
#     sd.stop()
#     sd.play(_normalize_for_playback(audio), samplerate=fs, blocking=False)


# _stimuli = {name: _load_mono_speech(path) for name, path in _STIMULUS_PATHS.items()}
# _select_stimulus("male")

# print("Random-phase listening test (magnitude preserved, phase blend 0-100%).")
# print("Paths:", {k: str(v) for k, v in _STIMULUS_PATHS.items()})
# print(
#     "Controls: 1=original  2=scrambled  3=difference  "
#     "M/F/P=stimulus  [/]=scramble -/+  R=new random phase  ESC=quit"
# )

# pygame.init()
# _screen = pygame.display.set_mode((760, 260))
# pygame.display.set_caption("Random phase listening")
# _font = pygame.font.SysFont(None, 26)
# _running = True

# while _running:
#     _screen.fill((25, 25, 25))
#     for _i, _line in enumerate(
#         [
#             f"Stimulus: {_selected}   Phase scramble: {100 * _SCRAMBLE_AMOUNT:.0f}%",
#             "1: original   2: scrambled (at % above)   3: difference",
#             "M / F / P: male / female / pop",
#             "[ / ] or - / +: less / more scramble   R: new random phase   ESC: quit",
#         ]
#     ):
#         _screen.blit(_font.render(_line, True, (220, 220, 220)), (20, 24 + 36 * _i))
#     pygame.display.flip()

#     for _event in pygame.event.get():
#         if _event.type == pygame.QUIT:
#             _running = False
#         elif _event.type == pygame.KEYDOWN:
#             if _event.key == pygame.K_ESCAPE:
#                 _running = False
#             elif _event.key == pygame.K_1:
#                 print("Playing: original")
#                 _play_mono(_original)
#             elif _event.key == pygame.K_2:
#                 print(f"Playing: scrambled ({100 * _SCRAMBLE_AMOUNT:.0f}% random phase)")
#                 _play_mono(_phase_random)
#             elif _event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
#                 _nudge_scramble(-_SCRAMBLE_STEP)
#                 print(f"Phase scramble: {100 * _SCRAMBLE_AMOUNT:.0f}%")
#             elif _event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
#                 _nudge_scramble(_SCRAMBLE_STEP)
#                 print(f"Phase scramble: {100 * _SCRAMBLE_AMOUNT:.0f}%")
#             elif _event.key == pygame.K_3:
#                 print("Playing: original minus phase-random (difference)")
#                 _play_mono(_difference)
#             elif _event.key == pygame.K_m:
#                 _select_stimulus("male")
#                 print("Selected: male speech")
#             elif _event.key == pygame.K_f:
#                 _select_stimulus("female")
#                 print("Selected: female speech")
#             elif _event.key == pygame.K_p:
#                 _select_stimulus("pop")
#                 print("Selected: pop music")
#             elif _event.key == pygame.K_r:
#                 _rebuild_scrambled(new_random_phase=True)
#                 print(
#                     f"Re-randomised phase at {100 * _SCRAMBLE_AMOUNT:.0f}% scramble; press 2"
#                 )

# sd.stop()
# pygame.quit()