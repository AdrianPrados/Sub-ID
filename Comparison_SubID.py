import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
import pandas as pd
import torch
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

import time
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial.transform import Rotation, Slerp

# Import of our new unified code
import SubID as subid

# Import Scattershot
from Methods.submovements.python import movement_decompose_2d
# Import SSSUMO
from Methods.sssumoMethod.notebooks import inferencia
# Import Gowda
from Methods.GowdaMethod import gowda_algorithm  

OUTPUT_DIR = 'Plots/Plots_ALL_COMPARISONS'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

np.set_printoptions(precision=4, suppress=True)

# -----------------------------------------------------------------------------
# 1. DATA LOADING & REGULARIZATION FUNCTIONS
# -----------------------------------------------------------------------------

def regularize_trajectory(t, xs, ys, zs=None, size='2D', target_fps=None):
    print(f"   [JITTER CORRECTION] Regularizing Data (Resampling)...")
    duration = t[-1] - t[0]
    
    # 1. Determinar FPS objetivo
    if target_fps is None:
        n_samples_orig = len(t)
        target_fps = (n_samples_orig - 1) / duration
    
    # 2. Vector de tiempo constante
    n_new = int(np.round(duration * target_fps)) + 1
    t_reg = np.linspace(0, duration, n_new)
    
    if size == '3D':
        cs_x = CubicSpline(t, xs)
        cs_y = CubicSpline(t, ys)
        cs_z = CubicSpline(t, zs)
        xs_reg = cs_x(t_reg)
        ys_reg = cs_y(t_reg)
        zs_reg = cs_z(t_reg)
        
        dt_reg = t_reg[1] - t_reg[0]
        vx_reg = np.gradient(xs_reg, dt_reg)
        vy_reg = np.gradient(ys_reg, dt_reg)
        vz_reg = np.gradient(zs_reg, dt_reg)
        vt_reg = np.sqrt(vx_reg**2 + vy_reg**2 + vz_reg**2)
        
        print(f"   -> Resampled: {len(t)} -> {len(t_reg)} samples (@{target_fps:.2f}Hz)")
        return t_reg, xs_reg, ys_reg, zs_reg, vx_reg, vy_reg, vz_reg, vt_reg
    else:
        # 3. Interpolación de POSICIONES
        cs_x = CubicSpline(t, xs)
        cs_y = CubicSpline(t, ys)
        xs_reg = cs_x(t_reg)
        ys_reg = cs_y(t_reg)
        zs_reg = np.zeros_like(xs_reg)
        
        dt_reg = t_reg[1] - t_reg[0]
        vx_reg = np.gradient(xs_reg, dt_reg)
        vy_reg = np.gradient(ys_reg, dt_reg)
        vz_reg = np.zeros_like(vx_reg)
        vt_reg = np.sqrt(vx_reg**2 + vy_reg**2)
    
        print(f"   -> Resampled: {len(t)} -> {len(t_reg)} samples (@{target_fps:.2f}Hz)")
        return t_reg, xs_reg, ys_reg, vx_reg, vy_reg, vt_reg

def import_data(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 4
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    
    states_raw = zarr_data["data/flat_spatula"][:] if "data/flat_spatula" in zarr_data else zarr_data["data/state"][:]
    times_raw = np.asarray(zarr_data["data/time"][:], dtype=np.float64).ravel()
    
    dt = 1.0 / float(target_hz)
    state_dim = states_raw.shape[1]

    target_times_list, states_list, episode_ends_list = [], [], []
    n_list = 0

    for trial in range(episode_ends_raw.shape[0]):
        dex_start = 0 if trial == 0 else episode_ends_raw[trial - 1]
        dex_end = episode_ends_raw[trial] 
        dex_states = np.arange(dex_start, dex_end)

        t_episode = times_raw[dex_states].copy()
        t_zero = t_episode[0]
        t_episode = t_episode - t_zero
        t_episode = np.maximum.accumulate(t_episode)
        t_episode = t_episode + np.linspace(0, 1e-3, len(t_episode))
        t_episode = t_episode + t_zero

        t_start_ep, t_end_ep = t_episode[0], t_episode[-1]
        target_times = np.arange(t_start_ep + dt, t_end_ep - dt, dt)
        states = np.zeros((len(target_times), state_dim))

        dim_pos = min(3, state_dim)
        for i in range(dim_pos):
            f_interp = interp1d(t_episode, states_raw[dex_states, i], kind="cubic", assume_sorted=True)
            states[:, i] = f_interp(target_times)

        b, a = butter(butter_order, lowpass_cutoff / (target_hz / 2), btype="low", analog=False)
        for i in range(dim_pos):
            states[:, i] = filtfilt(b, a, states[:, i], axis=0)

        target_times_list.append(target_times)
        states_list.append(states)
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_processed_all = np.concatenate(states_list, axis=0)
    episode_ends_processed = np.array(episode_ends_list)

    return (states_raw, times_raw, episode_ends_raw, states_processed_all, target_times_all, episode_ends_processed)
    
def import_data_filtered(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 4.0 
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    states_raw = zarr_data["data/flat_spatula"][:] if "data/flat_spatula" in zarr_data else zarr_data["data/state"][:]
    times_raw = np.asarray(zarr_data["data/time"][:], dtype=np.float64).ravel()
    dt = 1.0 / float(target_hz)

    target_times_list, states_list, episode_ends_list = [], [], []
    n_list = 0

    for trial in range(episode_ends_raw.shape[0]):
        dex_start = 0 if trial == 0 else episode_ends_raw[trial - 1]
        dex_end = episode_ends_raw[trial] - 1
        dex_states = np.arange(dex_start, dex_end)

        target_times = np.arange(times_raw[dex_start] + dt, times_raw[dex_end] - dt, dt)
        states = np.zeros((len(target_times), 7))

        for i in range(3):
            f_interp = interp1d(times_raw[dex_states], states_raw[dex_states, i], kind="cubic")
            states[:, i] = f_interp(target_times)

        if states_raw.shape[1] >= 7:
            rot_in = Rotation.from_quat(states_raw[dex_states, 3:7])
            slerp = Slerp(times_raw[dex_states], rot_in)
            states[:, 3:7] = slerp(target_times).as_quat()

        b, a = butter(butter_order, lowpass_cutoff / (target_hz / 2), btype="low", analog=False)
        for i in range(3):
            states[:, i] = filtfilt(b, a, states[:, i], axis=0)

        target_times_list.append(target_times)
        states_list.append(states)
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_processed_all = np.concatenate(states_list, axis=0)
    episode_ends_processed = np.array(episode_ends_list)

    return (states_raw, times_raw, episode_ends_raw, states_processed_all, target_times_all, episode_ends_processed)

def import_data_CSV(csv_path_str: str):
    csv_path = Path(csv_path_str)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    target_hz, lowpass_cutoff, butter_order, pad_time = 100.0, 4.0, 4, 0.5 
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz, Pad Time={pad_time}s")

    episode_starts = df[df['time'].diff() < 0].index.tolist()
    episode_ends_raw = np.array(episode_starts + [len(df)], dtype=np.int64)
    
    states_raw = np.zeros((len(df), 7), dtype=np.float64)
    states_raw[:, 0], states_raw[:, 1], states_raw[:, 2] = df['x'].values, df['y'].values, df['z'].values
    states_raw[:, 6] = 1.0 
    
    times_raw = df['time'].values
    dt = 1.0 / float(target_hz)

    target_times_list, states_list, episode_ends_list, n_list = [], [], [], 0

    for trial in range(episode_ends_raw.shape[0]):
        dex_start = 0 if trial == 0 else episode_ends_raw[trial - 1]
        dex_end = episode_ends_raw[trial] - 1
        dex_states = np.arange(dex_start, dex_end)

        if len(dex_states) < 2: continue
        target_times = np.arange(times_raw[dex_start] - pad_time, times_raw[dex_end] + pad_time, dt)
        if len(target_times) == 0: continue
            
        states = np.zeros((len(target_times), 7))
        for i in range(3):
            f_interp = interp1d(times_raw[dex_states], states_raw[dex_states, i], kind="cubic", assume_sorted=True,
                                bounds_error=False, fill_value=(states_raw[dex_states[0], i], states_raw[dex_states[-1], i]))
            states[:, i] = f_interp(target_times)

        b, a = butter(butter_order, lowpass_cutoff / (target_hz / 2), btype="low", analog=False)
        pad_len = min(len(target_times) - 1, 3 * max(len(a), len(b)))
        if pad_len > 0:
            states[:, 0] = filtfilt(b, a, states[:, 0], axis=0, padlen=pad_len)
            states[:, 1] = filtfilt(b, a, states[:, 1], axis=0, padlen=pad_len)
            states[:, 2] = filtfilt(b, a, states[:, 2], axis=0, padlen=pad_len)

        target_times_list.append(target_times)
        states_list.append(states)
        n_list += len(target_times)
        episode_ends_list.append(n_list)

    return (states_raw, times_raw, episode_ends_raw, np.concatenate(states_list, axis=0), 
            np.concatenate(target_times_list, axis=0), np.array(episode_ends_list, dtype=np.int64))


# -----------------------------------------------------------------------------
# 1. SSSUMO WRAPPER
# -----------------------------------------------------------------------------
def run_sssumo_inference(t, vt_ref):
    print("\n>>> RUNNING SSSUMO INFERENCE <<<")
    ROOT_SSSUMO = '/sssumo/'
    CFG_PATH = '/sssumoMethod/configs/config-0425-tune_ModGauss_wo_writing.yaml'
    WGT_PATH = '/sssumoMethod/checkpoints/config-0425-tune_ModGauss_wo_writing_9.pth'
    
    cfg = inferencia.Config(CFG_PATH, root_dir=ROOT_SSSUMO)
    dev = cfg.device if torch.cuda.is_available() else 'cpu'
    model = inferencia.TDNNDetector(batchnorm=cfg.batchnorm, dilations=cfg.dilations, channels=cfg.channels, 
                        kernel_sizes=cfg.kernel_sizes, num_layers=cfg.num_layers).to(dev, cfg.dtype)
    model.load_state_dict(torch.load(WGT_PATH, map_location=dev))
    model.eval()
    
    vt_norm = (vt_ref - np.mean(vt_ref)) / (np.std(vt_ref) + 1e-6)
    x_tensor = torch.tensor(vt_norm, dtype=cfg.dtype).unsqueeze(0).unsqueeze(0).to(dev)
    
    with torch.no_grad():
        y_pred = model(x_tensor)
        
    mask = y_pred[0, 0].cpu().numpy()
    dur = y_pred[0, 2].cpu().numpy()
    
    onsets = np.where((mask[:-1] < 0.5) & (mask[1:] >= 0.5))[0]
    
    starts_s, ends_s = [], []
    dt = t[1] - t[0]
    for i in onsets:
        d_val = dur[i]
        if d_val > 5:
            starts_s.append(t[i])
            ends_s.append(t[i] + d_val * dt)
            
    return np.array(starts_s), np.array(ends_s)
        

# -----------------------------------------------------------------------------
# 2. PLOTTING FUNCTION (Comparison 4-Way)
# -----------------------------------------------------------------------------
def plot_comparison_3way(t, refs, results, title_suffix="", size='2D', method='minjerk'):
    is_3d = (size == '3D')
    print(f"Plotting in {size} mode...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Comparison ({size}): {title_suffix}", fontsize=16)

    # A. Trajectory (Top Left)
    if is_3d:
        ax_traj = fig.add_subplot(3, 3, 1, projection='3d')
        ax_traj.plot(refs['x'], refs['y'], refs['z'], 'k-', lw=3, alpha=0.3, label='GT')
        if 'Our' in results:
            ax_traj.plot(results['Our'][0], results['Our'][1], results['Our'][2], 'b--', label='Ours')
        if 'Gowda' in results:
            ax_traj.plot(results['Gowda'][0], results['Gowda'][1], results['Gowda'][2], color='purple', linestyle=':', label='Gowda')    
        ax_traj.set_xlabel('X'); ax_traj.set_ylabel('Y'); ax_traj.set_zlabel('Z')
        ax_traj.set_title('3D Trajectory')
    else:
        ax_traj = fig.add_subplot(3, 3, 1)
        ax_traj.plot(refs['x'], refs['y'], 'k-', lw=3, alpha=0.3, label='GT')
        if 'Our' in results: 
            ax_traj.plot(results['Our'][0], results['Our'][1], 'b--', label='Ours')
        if 'Jason' in results: 
            ax_traj.plot(results['Jason'][0], results['Jason'][1], 'r:', lw=2, label='Jason')
        if 'Gowda' in results:
            ax_traj.plot(results['Gowda'][0], results['Gowda'][1], color='purple', linestyle=':', lw=2, label='Gowda')
        ax_traj.set_title('2D Trajectory')
    ax_traj.legend()

    # B. Velocidad Tangencial (Top Center)
    ax_vel = fig.add_subplot(3, 3, 2)
    ax_vel.plot(t, refs['vt'], 'k-', lw=2, alpha=0.3, label='GT')
    
    if 'Our' in results:
        idx_vt = 3 if is_3d else 2
        ax_vel.plot(t, results['Our'][idx_vt], 'b--', label='Ours')
    if 'Jason' in results and not is_3d:
        ax_vel.plot(t, results['Jason'][2], 'r:', label='Jason')
    if 'Gowda' in results:
        idx_vt = 3 if is_3d else 2
        ax_vel.plot(t, results['Gowda'][idx_vt], color='purple', linestyle=':', label='Gowda')
    if 'SSSUMO' in results:
        ax_vel.plot(results['SSSUMO'][0][0], results['SSSUMO'][0][1], 'g-.', alpha=0.8, lw=1.5, label='SSSUMO')
        
    ax_vel.set_title('Tangential Velocity')
    ax_vel.legend()
    
    ax_empty1 = fig.add_subplot(3, 3, 3)
    ax_empty1.axis('off')

    # C. Ours Primitives (Middle Left)
    ax_ours = fig.add_subplot(3, 3, 4)
    if 'Our' in results:
        data_our = results['Our']
        if is_3d:
            ts, te = data_our[4], data_our[5]
            sx, sy, sz = data_our[6], data_our[7], data_our[8]
            mus = data_our[9]
            vt_fit = data_our[3]
            Phi = subid.build_phi_matrix(t, ts, te, method=method, mus=mus)
        else:
            ts, te = data_our[3], data_our[4]
            sx, sy = data_our[5], data_our[6]
            mus = data_our[7]
            sz = np.zeros_like(sx) 
            vt_fit = data_our[2]
            Phi = subid.build_phi_matrix(t, ts, te, method=method, mus=mus)

        ax_ours.plot(t, refs['vt'], 'k', alpha=0.3, label='GT')
        ax_ours.plot(t, vt_fit, 'b--', label='Ours Fit')
        
        added_label = False
        for k in range(len(ts)):
            if is_3d: real_amp = np.sqrt(sx[k]**2 + sy[k]**2 + sz[k]**2)
            else: real_amp = np.sqrt(sx[k]**2 + sy[k]**2)
            curve = Phi[:, k] * real_amp
            lbl = 'Primitive' if not added_label else None
            ax_ours.plot(t, curve, 'b', alpha=0.6, lw=1, label=lbl)
            ax_ours.fill_between(t, 0, curve, color='blue', alpha=0.1)
            added_label = True
            
        ax_ours.legend()
        ax_ours.set_title(f"Primitives (Ours) - K={len(ts)}")
    else:
        ax_ours.axis('off')

    # D. Jason Primitives (Middle Center)
    ax_jason = fig.add_subplot(3, 3, 5)
    if 'Jason' in results and not is_3d:
        _, _, _, ts, te, sx, sy = results['Jason']
        Phi = subid.build_phi_matrix(t, ts, te, method='minjerk')
        ax_jason.plot(t, refs['vt'], 'k', alpha=0.3, label='GT')
        
        added_label = False
        ax_jason.plot(t, results['Jason'][2], 'r:', label='Jason')
        for k in range(len(ts)):
            real_amp = np.sqrt(sx[k]**2 + sy[k]**2)
            curve = Phi[:,k] * real_amp
            lbl = 'Primitive' if not added_label else None
            ax_jason.plot(t, curve, 'r', alpha=0.6, lw=1, label=lbl)
            ax_jason.fill_between(t, 0, curve, color='red', alpha=0.1)
            added_label = True
        ax_jason.legend()
        ax_jason.set_title(f"Primitives (Jason) - K={len(ts)}")
    else:
        if is_3d: ax_jason.text(0.5, 0.5, "Jason Not Available in 3D", ha='center')
        ax_jason.axis('off')
        
    # GOWDA Primitives (Middle Right) 
    ax_gowda = fig.add_subplot(3, 3, 6)
    if 'Gowda' in results:
        data_gow = results['Gowda']
        if is_3d:
            ts, te = data_gow[4], data_gow[5]
            sx, sy, sz = data_gow[6], data_gow[7], data_gow[8]
            vt_fit = data_gow[3]
            Phi = subid.build_phi_matrix(t, ts, te, method='minjerk')
        else:
            ts, te = data_gow[3], data_gow[4]
            sx, sy = data_gow[5], data_gow[6]
            sz = np.zeros_like(sx) 
            vt_fit = data_gow[2]
            Phi = subid.build_phi_matrix(t, ts, te, method='minjerk')

        ax_gowda.plot(t, refs['vt'], 'k', alpha=0.3, label='GT')
        ax_gowda.plot(t, vt_fit, color='purple', linestyle=':', label='Gowda Fit')
        
        added_label = False
        for k in range(len(ts)):
            if is_3d: real_amp = np.sqrt(sx[k]**2 + sy[k]**2 + sz[k]**2)
            else: real_amp = np.sqrt(sx[k]**2 + sy[k]**2)
            curve = Phi[:, k] * real_amp
            lbl = 'Primitive' if not added_label else None
            ax_gowda.plot(t, curve, color='purple', alpha=0.6, lw=1, label=lbl)
            ax_gowda.fill_between(t, 0, curve, color='purple', alpha=0.1)
            added_label = True
            
        ax_gowda.legend()
        ax_gowda.set_title(f"Primitives (Gowda) - K={len(ts)}")
    else:
        ax_gowda.axis('off')

    # E. SSSUMO Primitives (Bottom Left)
    ax_sssumo = fig.add_subplot(3, 3, 7)
    if 'SSSUMO' in results:
        (t_s, vt_s), primitives_list = results['SSSUMO']
        ax_sssumo.plot(t_s, vt_s, 'g-.', label='Recon Sum', alpha=0.5)
        ax_sssumo.plot(t, refs['vt'], 'k', alpha=0.3, label='GT')
        
        added_label = False
        for t_curve, v_curve in primitives_list:
            lbl = 'Primitive' if not added_label else None
            ax_sssumo.plot(t_curve, v_curve, 'g', alpha=0.6, lw=1, label=lbl)
            ax_sssumo.fill_between(t_curve, 0, v_curve, color='green', alpha=0.1)
            added_label = True
            
        ax_sssumo.legend()
        ax_sssumo.set_title(f'SSSUMO Primitives (K={len(primitives_list)})')
    else:
        ax_sssumo.axis('off')

    ax_empty2 = fig.add_subplot(3, 3, 8)
    ax_empty2.axis('off')
    ax_empty3 = fig.add_subplot(3, 3, 9)
    ax_empty3.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Comparison_{title_suffix}.svg")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    
def calc_metrics(y_true, y_pred,size="2D"):
    if size=="2D":
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
    elif size=="3D":
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2, rmse

def save_decomposition_data(filename, t, refs, fit_data, params, exec_time, size="2D"):
    path = os.path.join(OUTPUT_DIR, filename)
    r2_vt, rmse_vt = calc_metrics(refs['vt'], fit_data['vt'],size=size)
    num_k = len(params['starts'])
    
    if size=="2D":
        np.savez(
            path,
            t=t,
            vx_ref=refs['vx'], vy_ref=refs['vy'], vt_ref=refs['vt'],
            px_ref=refs['x'], py_ref=refs['y'],
            vx_fit=fit_data['vx'], vy_fit=fit_data['vy'], vt_fit=fit_data['vt'],
            px_fit=fit_data['x'], py_fit=fit_data['y'],
            subm_starts=params['starts'],
            subm_ends=params['ends'],
            subm_scales_x=params['sx'],
            subm_scales_y=params['sy'],
            subm_mu=params.get('mu', np.zeros_like(params['sx'])),
            metrics=np.array([r2_vt, rmse_vt, num_k, exec_time])
        )
    elif size=="3D":
        np.savez(
            path,
            t=t,
            vx_ref=refs['vx'], vy_ref=refs['vy'],vz_ref=refs['vz'], vt_ref=refs['vt'],
            px_ref=refs['x'], py_ref=refs['y'],pz_ref=refs['z'],
            vx_fit=fit_data['vx'], vy_fit=fit_data['vy'],vz_fit=fit_data['vz'], vt_fit=fit_data['vt'],
            px_fit=fit_data['x'], py_fit=fit_data['y'],pz_fit=fit_data['z'],
            subm_starts=params['starts'],
            subm_ends=params['ends'],
            subm_scales_x=params['sx'],
            subm_scales_y=params['sy'],
            subm_scales_z=params['sz'],
            subm_mu=params.get('mu', np.zeros_like(params['sx'])),
            metrics=np.array([r2_vt, rmse_vt, num_k, exec_time])
        )
    print(f"   -> Saved: {path}")

# -----------------------------------------------------------------------------
# 3. MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    # --- GLOBAL CONFIGURATION (SUBID PARAMETERS) ---
    METHOD = 'minjerk' # or 'lgnb'
    ALPHA_STRATEGY = 'dynamic' # 'dynamic' or manual float (e.g., 0.05 or 0.0)
    
    if ALPHA_STRATEGY == 'dynamic':
        subid.USE_DYNAMIC_ALPHA = True
        subid.GLOBAL_ALPHA = 0.0
    else:
        subid.USE_DYNAMIC_ALPHA = False
        subid.GLOBAL_ALPHA = float(ALPHA_STRATEGY)

    # --- SELECT DATA ---
    DATA_SOURCE = 'PUSHTReal3d' # 'SYNTHETIC', 'PUSHT', 'PUSHTReal2d', 'SUBJECT', 'LETTERS', 'PUSHTReal3d', 'SPATULA', 'MOVING3D'
    
    PUSHT_SIMULATED = "data/pusht_real/real_pusht_20230105/replay_buffer.zarr"
    PUSHT_REAL_2D = "data/EPFL_data/adrian_adrian2D_2026-03-13-13-50/data.zarr"
    SUBJECT_STROKE = "data/subject_stroke/subject08day1post"
    PUSHT_REAL_3D = "data/EPFL_data/adrian_adrian3D_2026-03-13-16-04/data.zarr"
    MOVING_DATA = "data/moving_object/object_moving_tangential_velocity_data.csv"
    LETTERS_PATH = "data/Handwriting/character_D_minjerk.zarr" 
    SPATULA_PATH = "data/adrian_data/pushing_2026-02-20-16-16/spatula_pose_raw.zarr"
    
    print(f"--- 1. LOADING {DATA_SOURCE} ---")
    
    t = None
    xs, ys, zs = None, None, None
    vx, vy, vz, vt = None, None, None, None
    P = None
    
    if DATA_SOURCE == 'SYNTHETIC':
        print("Generating Synthetic MinJerk Data (Cascaded)...")
        N_SAMPLES = 300
        DURATION = 1.0
        K_TARGET = 5
        
        ts_gt, te_gt, sx_gt, sy_gt, p0_gt = subid.get_synthetic_params_cascaded(K_TARGET)
        
        t = np.linspace(0, DURATION, N_SAMPLES)
        Phi_gt = subid.build_phi_matrix(t, ts_gt, te_gt, method='minjerk') 
        
        vx = Phi_gt.dot(sx_gt)
        vy = Phi_gt.dot(sy_gt)
        vt = np.sqrt(vx**2 + vy**2)
        
        xs = cumtrapz(vx, t, initial=0.0) + p0_gt[0]
        ys = cumtrapz(vy, t, initial=0.0) + p0_gt[1]
        P = np.column_stack([xs, ys])
        
        print(f"Generated {K_TARGET} submovements over {DURATION}s.")
        
    elif DATA_SOURCE == 'PUSHT':
        if not os.path.exists(PUSHT_SIMULATED): exit(f"Zarr not found at {PUSHT_SIMULATED}")
        z = zarr.open(PUSHT_SIMULATED, mode='r')
        ends = z['meta/episode_ends'][:]
        ep_idx = 0 
        s = 0 if ep_idx==0 else ends[ep_idx-1]; e = ends[ep_idx]
        raw = z['data/robot_eef_pose'][s:e].astype(float)
        ts_raw = z['data/timestamp'][s:e]
        t = ts_raw - ts_raw[0]
        xs = savgol_filter(raw[:,0], 31, 3); ys = savgol_filter(raw[:,1], 31, 3)
        xs = xs*100; ys = ys*100
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
        vt = np.sqrt(vx**2 + vy**2)
        P = np.column_stack([xs, ys])
        
    elif DATA_SOURCE == 'PUSHTReal2d':
        (states_raw, times_raw, episode_ends_raw, 
        states_processed_all, target_times_all, episode_ends_processed) = import_data(PUSHT_REAL_2D)
        
        EPISODE_IDX = 0
        start_idx = 0 if EPISODE_IDX == 0 else episode_ends_processed[EPISODE_IDX-1]
        end_idx = episode_ends_processed[EPISODE_IDX]
        
        episode_data = states_processed_all[start_idx:end_idx, :].astype(float)
        t_raw = target_times_all[start_idx:end_idx]
        t = t_raw - t_raw[0]
        
        mask = t <= 8.0
        t = t[mask]
        episode_data = episode_data[mask, :]
        
        N = t.size
        xs = episode_data[:, 0]
        ys = episode_data[:, 1]
        xs = xs*100; ys = ys*100
            
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
        vt = np.sqrt(vx**2 + vy**2)
        
        P = np.column_stack([xs, ys])
        print(f"Loaded Episode {EPISODE_IDX}. Samples: {N}.")
        
    elif DATA_SOURCE == 'SUBJECT':
        position_filtered, velocity_list, time_list = movement_decompose_2d.load_data(SUBJECT_STROKE)
        if not position_filtered: exit("No data loaded.")
        EPISODE_IDX = 1
        P_raw = position_filtered[EPISODE_IDX]
        V = velocity_list[EPISODE_IDX]
        t_raw = time_list[EPISODE_IDX]
        t = t_raw - t_raw[0]
        xs, ys = P_raw[:, 0], P_raw[:, 1]
        vx, vy = V[:, 0], V[:, 1]
        vt = np.sqrt(vx**2 + vy**2)
        P = np.column_stack([xs, ys])
        print("Valores de vx y vy (primeros 5):", vx[:5], vy[:5])
        
    elif DATA_SOURCE == 'LETTERS':
        print(f"Loading Zarr: {LETTERS_PATH}")
        if not os.path.exists(LETTERS_PATH): exit(f"Zarr not found at {LETTERS_PATH}")
        z = zarr.open(str(LETTERS_PATH), mode='r')
        
        if "data/state" not in z:
            raise RuntimeError("Zarr no contiene 'data/state' en la ruta esperada.")
        states_all = z["data/state"][:]
        
        if "meta/episode_ends" in z:
            episode_ends = z["meta/episode_ends"][:]
        elif "data/episode_ends" in z:
            episode_ends = z["data/episode_ends"][:]
        else:
            raise RuntimeError("Zarr no contiene 'episode_ends' en 'meta' ni 'data'.")

        times_all = None
        if "data/time" in z:
            times_all = np.asarray(z["data/time"][:]).ravel()

        EPISODE_IDX = 3 
        start = 0 if EPISODE_IDX == 0 else int(episode_ends[EPISODE_IDX-1])
        end = int(episode_ends[EPISODE_IDX])
        ep_states = states_all[start:end]

        if ep_states.shape[1] >= 2:
            P_raw = ep_states[:, :2].astype(np.float64)
        else:
            raise RuntimeError("State shape inesperada: no hay 2 columnas físicas al inicio.")

        target_hz = 100.0
        if times_all is None:
            N_len = P_raw.shape[0]
            t = np.arange(N_len) * (1.0/target_hz)
        else:
            t = times_all[start:end]
            t = t - t[0]
            
        t = np.maximum.accumulate(t)
        if t.size < 2:
            raise RuntimeError("Episodio demasiado corto para procesar.")

        dt = t[1] - t[0]

        nyq = 0.5 * target_hz
        normal_cutoff = 5.0 / nyq  
        b, a = butter(4, normal_cutoff, btype='low', analog=False) 
        pad_len = min(len(t)-1, 3 * max(len(a), len(b)))
        
        P = np.zeros_like(P_raw)
        if pad_len >= 1:
            P[:,0] = filtfilt(b, a, P_raw[:,0], padlen=pad_len)
            P[:,1] = filtfilt(b, a, P_raw[:,1], padlen=pad_len)
        else:
            P = P_raw.copy()
        P = P*100
        xs, ys = P[:, 0], P[:, 1]

        V_ref_2d = np.zeros_like(P)
        for i in range(2):
            V_ref_2d[:, i] = np.gradient(P[:, i], t)
            
        vt = np.linalg.norm(V_ref_2d, axis=1)
        vx = V_ref_2d[:, 0]
        vy = V_ref_2d[:, 1]
        print("Valores de vx y vy (primeros 5):", vx[:5], vy[:5])
        
    elif DATA_SOURCE == 'PUSHTReal3d' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data_filtered(PUSHT_REAL_3D)
        
        EPISODE_IDX = 1
        start_raw = 0 if EPISODE_IDX == 0 else episode_ends_raw[EPISODE_IDX-1]
        end_raw = episode_ends_raw[EPISODE_IDX]
        t_raw_ep = times_raw[start_raw:end_raw]
        p_raw_ep = states_raw[start_raw:end_raw, :3] 

        start_proc = 0 if EPISODE_IDX == 0 else episode_ends_proc[EPISODE_IDX-1]
        end_proc = episode_ends_proc[EPISODE_IDX]
        t_proc_ep = times_proc[start_proc:end_proc]
        p_proc_ep = states_proc[start_proc:end_proc, :3] 
        
        t = t_proc_ep - t_proc_ep[0]
        mask = t <= 7.0
        t = t[mask]
        p_proc_ep = p_proc_ep[mask, :]
        
        print("Total time (s):", t[-1])
        P = p_proc_ep
        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        
        V_ref_3d = np.zeros_like(P)
        for i in range(3):
            V_ref_3d[:, i] = np.gradient(P[:, i], t)
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]
        
    elif DATA_SOURCE == 'SPATULA' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data(SPATULA_PATH)
        
        EPISODE_IDX = 1
        start_raw = 0 if EPISODE_IDX == 0 else episode_ends_raw[EPISODE_IDX-1]
        end_raw = episode_ends_raw[EPISODE_IDX]
        t_raw_ep = times_raw[start_raw:end_raw]
        p_raw_ep = states_raw[start_raw:end_raw, :3] 

        start_proc = 0 if EPISODE_IDX == 0 else episode_ends_proc[EPISODE_IDX-1]
        end_proc = episode_ends_proc[EPISODE_IDX]
        t_proc_ep = times_proc[start_proc:end_proc]
        p_proc_ep = states_proc[start_proc:end_proc, :3] 
        
        t = t_proc_ep - t_proc_ep[0]
        mask = t <= 7.0
        t = t[mask]
        p_proc_ep = p_proc_ep[mask, :]
        
        print("Total time (s):", t[-1])
        P = p_proc_ep
        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        
        V_ref_3d = np.zeros_like(P)
        for i in range(3):
            V_ref_3d[:, i] = np.gradient(P[:, i], t)
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]

    elif DATA_SOURCE == 'MOVING3D' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data_CSV(MOVING_DATA)
        
        EPISODE_IDX = 3
        start_raw = 0 if EPISODE_IDX == 0 else episode_ends_raw[EPISODE_IDX-1]
        end_raw = episode_ends_raw[EPISODE_IDX]
        t_raw_ep = times_raw[start_raw:end_raw]
        p_raw_ep = states_raw[start_raw:end_raw, :3] 

        start_proc = 0 if EPISODE_IDX == 0 else episode_ends_proc[EPISODE_IDX-1]
        end_proc = episode_ends_proc[EPISODE_IDX]
        t_proc_ep = times_proc[start_proc:end_proc]
        p_proc_ep = states_proc[start_proc:end_proc, :3] 
        
        t = t_proc_ep - t_proc_ep[0] 
        print("Total time (s):", t[-1])
        P = p_proc_ep
        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        
        V_ref_3d = np.zeros_like(P)
        for i in range(3):
            V_ref_3d[:, i] = np.gradient(P[:, i], t)
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]

    # --- >>> CORRECCIÓN DE JITTER APLICADA AQUÍ <<< ---
    if DATA_SOURCE in ['PUSHTReal3d', 'SPATULA', 'MOVING3D'] :
        t_S, xs_S, ys_S, zs_S, vx_S, vy_S, vz_S, vt_S = regularize_trajectory(t, xs, ys, zs,size='3D')
    else:
        t_S, xs_S, ys_S, vx_S, vy_S, vt_S = regularize_trajectory(t, xs, ys)

    RESULTS = {}
    is_3d = (zs is not None)
    
    if is_3d:
        REFS = {'vx': vx, 'vy': vy, 'vz': vz, 'vt': vt, 'x': xs, 'y': ys, 'z': zs}
        V_ref = np.column_stack([vx, vy, vz])
    else:
        REFS = {'vx': vx, 'vy': vy, 'vt': vt, 'x': xs, 'y': ys}
        V_ref = np.column_stack([vx, vy])
    
    D_dim = P.shape[1]
    
    # -------------------------------------------------------------------------
    # RUN 1: OURs
    # -------------------------------------------------------------------------
    print("\n> Running YOUR Algorithm (SubID Unified)...")
    t0 = time.time()
    
    frefs = [V_ref[:, d] for d in range(D_dim)]
    ts_init, te_init, mus_init = subid.fit_hybrid_peaks_greedy_unified(t, frefs, method=METHOD, max_bases_total=1000, residual_tol=0.02, verbose=False)
    ts_opt, te_opt, mus_opt = subid.refine_bases_unified(t, P, ts_init, te_init, mus_init, method=METHOD, verbose=False)
    
    Phi_vel = subid.build_phi_matrix(t, ts_opt, te_opt, method=METHOD, mus=mus_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:,k] = cumtrapz(Phi_vel[:,k], t, initial=0)
    
    A_temp = np.hstack([Phi_pos, np.ones((len(t),1))])
    
    if subid.USE_DYNAMIC_ALPHA:
        alpha_temp = subid.compute_dynamic_ridge_alpha(A_temp, kappa_max=100.0)
    else:
        alpha_temp = subid.GLOBAL_ALPHA
        
    scales_list_temp = []
    for d in range(D_dim):
        w = subid.solve_ridge_weighted(A_temp, P[:, d], alpha=alpha_temp)
        scales_list_temp.append(w[:-1])
    scales_temp = np.column_stack(scales_list_temp)
    
    ts_m, te_m, mus_m, scales_m = subid.merge_bases_unified(ts_opt, te_opt, mus_opt, scales_temp, proximity_tol=0.04)
    ts_final, te_final, mus_final = subid.refine_bases_unified(t, P, ts_m, te_m, mus_m, method=METHOD, verbose=False)
    
    K_final = ts_final.size
    
    Phi_vel_final = subid.build_phi_matrix(t, ts_final, te_final, method=METHOD, mus=mus_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final): Phi_pos_final[:,k] = cumtrapz(Phi_vel_final[:,k], t, initial=0)
    
    A_final = np.hstack([Phi_pos_final, np.ones((len(t),1))])
    
    if subid.USE_DYNAMIC_ALPHA:
        alpha_final = subid.compute_dynamic_ridge_alpha(A_final, kappa_max=100.0)
    else:
        alpha_final = subid.GLOBAL_ALPHA
        
    scales_list_final = []
    p0_list = []
    for d in range(D_dim):
        w = subid.solve_ridge_weighted(A_final, P[:, d], alpha=alpha_final)
        scales_list_final.append(w[:-1])
        p0_list.append(w[-1]) 
    
    scales_final = np.column_stack(scales_list_final)
    
    P_fit = np.zeros_like(P)
    for d in range(D_dim):
        w_full = np.concatenate([scales_list_final[d], [p0_list[d]]])
        P_fit[:, d] = A_final.dot(w_full)
    
    V_fit = Phi_vel_final.dot(scales_final)
    vt_our = np.linalg.norm(V_fit, axis=1)
    
    print("Final time for Sub-ID:", time.time()-t0)
    print(f"  -> [OURS] Finished. K={K_final}.")
    
    if is_3d:
        RESULTS['Our'] = (P_fit[:,0], P_fit[:,1], P_fit[:,2], vt_our, ts_final, te_final, scales_final[:,0], scales_final[:,1], scales_final[:,2], mus_final)
        save_decomposition_data('decomposition_data.npz', t, REFS, 
                                {'vx': V_fit[:,0], 'vy': V_fit[:,1],'vz': V_fit[:,2], 'vt': vt_our, 'x': P_fit[:,0], 'y': P_fit[:,1], 'z': P_fit[:,2]},
                                {'starts': ts_final, 'ends': te_final, 'sx': scales_final[:,0], 'sy': scales_final[:,1], 'sz': scales_final[:,2], 'mu': mus_final}, 
                                time.time()-t0, size="3D")
    else:
        RESULTS['Our'] = (P_fit[:,0], P_fit[:,1], vt_our, ts_final, te_final, scales_final[:,0], scales_final[:,1], mus_final)
        save_decomposition_data('decomposition_data.npz', t, REFS, 
                                {'vx': V_fit[:,0], 'vy': V_fit[:,1], 'vt': vt_our, 'x': P_fit[:,0], 'y': P_fit[:,1]},
                                {'starts': ts_final, 'ends': te_final, 'sx': scales_final[:,0], 'sy': scales_final[:,1], 'mu': mus_final}, 
                                time.time()-t0)
    
    # -------------------------------------------------------------------------
    # RUN 2: JASON (Decompose2D)
    # -------------------------------------------------------------------------
    if is_3d:
        print("Jason not implmented in 3D yet. Skipping...")
    else:
        print("\n> Running JASON (Decompose2D)...")
        t0_jason = time.time()
        max_dx = np.max(xs) - np.min(xs) + 20
        max_dy = np.max(ys) - np.min(ys) + 20
        xr = (-max_dx, max_dx)
        yr = (-max_dy, max_dy)
        bestInit = 1e10
        for p in range(15):  
            print("Probando Scattershot p =", p+1)
            try:
                best, params_try = movement_decompose_2d.decompose_2D(t, np.column_stack([vx, vy]), n_sub_movement=p+1, x_rng=xr, y_rng=yr, iters=20)
            except Exception as e:
                print(f"Error en decompose_2D con p={p+1}: {e}")
                break
            
            params = params_try
            ts_j = params[:,0]; dur_j = params[:,1]; te_j = ts_j + dur_j
            sx_j = params[:,2]; sy_j = params[:,3]
            
            Phi_j_v = subid.build_phi_matrix(t, ts_j, te_j, method='minjerk')
            Phi_j_p = np.zeros_like(Phi_j_v)
            for k in range(len(ts_j)): Phi_j_p[:,k] = cumtrapz(Phi_j_v[:,k], t, initial=0)
            
            fit_x_j = Phi_j_p.dot(sx_j) + xs[0]; fit_y_j = Phi_j_p.dot(sy_j) + ys[0]
            vx_j = Phi_j_v.dot(sx_j); vy_j = Phi_j_v.dot(sy_j); vt_j = np.sqrt(vx_j**2 + vy_j**2)
            
            v_real = np.sqrt(vx**2 + vy**2)
            error = np.sqrt(np.mean((v_real - vt_j) ** 2))
            
            if error < bestInit:
                bestInit = error
                print("Data saved for Jason with error:", error)
                RESULTS['Jason'] = (fit_x_j, fit_y_j, vt_j, ts_j, te_j, sx_j, sy_j)
                save_decomposition_data('decomposition_data_Jason.npz', t, REFS,
                                {'vx': vx_j, 'vy': vy_j, 'vt': vt_j, 'x': fit_x_j, 'y': fit_y_j},
                                {'starts': ts_j, 'ends': te_j, 'sx': sx_j, 'sy': sy_j}, 
                                time.time()-t0_jason)
            
            print("Error:"+ str(error) +"%.")
            if error <= 3: 
                print("Error is:", error)
                break
        print(f"Best error after scattershot: {error:.4f}")
        print(f"  -> [JASON] Finished.")

    # -------------------------------------------------------------------------
    # RUN 4: GOWDA (Scattershot MinJerk 2015)
    # -------------------------------------------------------------------------
    print("\n> Running GOWDA...")
    t0_gow = time.time()
    
    try:
        if is_3d:
            hand_kin_gowda = np.column_stack([xs, ys, zs, vx, vy, vz])
            vel_indices = [3, 4, 5]
            is_3d_gowda = True
        else:
            hand_kin_gowda = np.column_stack([xs, ys, vx, vy])
            vel_indices = [2, 3]
            is_3d_gowda = False

        dt_ms = (t[1] - t[0]) * 1000.0 if len(t) > 1 else 20.0
        
        movements_gow, submovement_recon_gow, segments_gow, costs_gow, fits_gow, fit_costs_gow, runtime_gow, n_iter_gow, n_evals_gow = \
            gowda_algorithm.decompose_submovements_v2(
                hand_kin_gowda, 
                vel_inds=vel_indices,
                method='greedy',     
                fn_type='min_jerk', 
                bin_size_ms=dt_ms, 
                verbose=False
            )
        
        ts_gow_list, dur_gow_list, sx_gow_list, sy_gow_list, sz_gow_list = [], [], [], [], []
        for k_seg, mov_array in movements_gow.items():
            if mov_array is not None and len(mov_array) > 0:
                for param_row in mov_array:
                    ts_gow_list.append(param_row[0])
                    dur_gow_list.append(param_row[1])
                    sx_gow_list.append(param_row[2])
                    sy_gow_list.append(param_row[3])
                    
                    if is_3d_gowda and len(param_row) > 4:
                        sz_gow_list.append(param_row[4])
                    else:
                        sz_gow_list.append(0.0)
                        
        ts_gow = np.array(ts_gow_list)
        dur_gow = np.array(dur_gow_list)
        te_gow = ts_gow + dur_gow
        sx_gow = np.array(sx_gow_list)
        sy_gow = np.array(sy_gow_list)
        sz_gow = np.array(sz_gow_list)
        
        exec_time_gow = time.time() - t0_gow
        print(f"Gowda finished in {exec_time_gow:.2f}s.")
        
        if is_3d_gowda:
            Phi_gow_v = subid.build_phi_matrix(t, ts_gow, te_gow, method='minjerk')
            Phi_gow_p = np.zeros_like(Phi_gow_v)
            for k in range(len(ts_gow)): Phi_gow_p[:,k] = cumtrapz(Phi_gow_v[:,k], t, initial=0)
            
            fit_x_gow = Phi_gow_p.dot(sx_gow) + xs[0]
            fit_y_gow = Phi_gow_p.dot(sy_gow) + ys[0]
            fit_z_gow = Phi_gow_p.dot(sz_gow) + zs[0]
            vx_gow = Phi_gow_v.dot(sx_gow)
            vy_gow = Phi_gow_v.dot(sy_gow)
            vz_gow = Phi_gow_v.dot(sz_gow)
            vt_gow = np.sqrt(vx_gow**2 + vy_gow**2 + vz_gow**2)
            
            RESULTS['Gowda'] = (fit_x_gow, fit_y_gow, fit_z_gow, vt_gow, ts_gow, te_gow, sx_gow, sy_gow, sz_gow)
            
            save_decomposition_data('decomposition_data_Gowda.npz', t, REFS,
                                    {'vx': vx_gow, 'vy': vy_gow, 'vz': vz_gow, 'vt': vt_gow, 'x': fit_x_gow, 'y': fit_y_gow, 'z': fit_z_gow},
                                    {'starts': ts_gow, 'ends': te_gow, 'sx': sx_gow, 'sy': sy_gow, 'sz': sz_gow}, 
                                    time.time()-t0_gow, size="3D")
        else:
            Phi_gow_v = subid.build_phi_matrix(t, ts_gow, te_gow, method='minjerk')
            Phi_gow_p = np.zeros_like(Phi_gow_v)
            for k in range(len(ts_gow)): Phi_gow_p[:,k] = cumtrapz(Phi_gow_v[:,k], t, initial=0)
            
            fit_x_gow = Phi_gow_p.dot(sx_gow) + xs[0]
            fit_y_gow = Phi_gow_p.dot(sy_gow) + ys[0]
            vx_gow = Phi_gow_v.dot(sx_gow)
            vy_gow = Phi_gow_v.dot(sy_gow)
            vt_gow = np.sqrt(vx_gow**2 + vy_gow**2)
            
            RESULTS['Gowda'] = (fit_x_gow, fit_y_gow, vt_gow, ts_gow, te_gow, sx_gow, sy_gow)
            
            save_decomposition_data('decomposition_data_Gowda.npz', t, REFS,
                                    {'vx': vx_gow, 'vy': vy_gow, 'vt': vt_gow, 'x': fit_x_gow, 'y': fit_y_gow},
                                    {'starts': ts_gow, 'ends': te_gow, 'sx': sx_gow, 'sy': sy_gow}, 
                                    time.time()-t0_gow)
                                    
        print(f"  -> [GOWDA] Finished. Found {len(ts_gow)} submovements.")
        
    except Exception as e:
        print(f"  -> [!] Error en Gowda: {e}")

    # -------------------------------------------------------------------------
    # RUN 3: SSSUMO
    # -------------------------------------------------------------------------
    print("\n> Running SSSUMO (Inferencia.py)...")
    t0_ssumo = time.time()
    
    current_fps = 1.0 / (t_S[1] - t_S[0])
    
    sssumo_data_tuple, primitives_list = inferencia.decompose_numpy_velocity(vt_S, original_fps=current_fps)
    t_ssumo, vt_recon = sssumo_data_tuple
    RESULTS['SSSUMO'] = (sssumo_data_tuple, primitives_list)
    print("Final time for SSSUMO:", time.time()-t0_ssumo)
    print(f"  -> [SSSUMO] Found {len(primitives_list)} submovements.")
    
    starts_s = [p[0][0] for p in primitives_list]
    ends_s = [p[0][-1] for p in primitives_list]
    amps_s = [np.max(p[1]) for p in primitives_list]
    
    if len(vt_recon) != len(t): vt_recon = np.interp(t, t_ssumo, vt_recon)
    
    if is_3d:
        save_decomposition_data('decomposition_data_sssumo.npz', t, REFS,
                            {'vx': np.zeros_like(vt_recon), 'vy': np.zeros_like(vt_recon),'vz': np.zeros_like(vt_recon), 'vt': vt_recon, 
                            'x': np.zeros_like(vt_recon), 'y': np.zeros_like(vt_recon), 'z': np.zeros_like(vt_recon)},
                            {'starts': np.array(starts_s), 'ends': np.array(ends_s), 
                            'sx': np.array(amps_s), 'sy': np.zeros_like(amps_s), 'sz': np.zeros_like(amps_s)}, 
                            time.time()-t0_ssumo, size='3D')
    else:
        save_decomposition_data('decomposition_data_sssumo.npz', t, REFS,
                                {'vx': np.zeros_like(vt_recon), 'vy': np.zeros_like(vt_recon), 'vt': vt_recon, 
                                'x': np.zeros_like(vt_recon), 'y': np.zeros_like(vt_recon)},
                                {'starts': np.array(starts_s), 'ends': np.array(ends_s), 
                                'sx': np.array(amps_s), 'sy': np.zeros_like(amps_s)}, 
                                time.time()-t0_ssumo)

    # -------------------------------------------------------------------------
    # COMPARE & PLOT
    # -------------------------------------------------------------------------
    if is_3d:
        plot_comparison_3way(t, {'x':xs, 'y':ys, 'z':zs, 'vt':vt}, RESULTS, title_suffix=DATA_SOURCE, size='3D', method=METHOD)
    else:
        plot_comparison_3way(t, {'x':xs, 'y':ys, 'vt':vt}, RESULTS, title_suffix=DATA_SOURCE, method=METHOD)