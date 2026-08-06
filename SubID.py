import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re
import zarr
import pandas as pd
from pathlib import Path
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.signal import savgol_filter, find_peaks, peak_widths, butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

if not os.path.exists('Plots/Plots_AutomaticRidge_MultiDataset'):
    os.makedirs('Plots/Plots_AutomaticRidge_MultiDataset')

np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# 1. DATA LOADING FUNCTIONS
# =============================================================================

def import_data(zarr_root_str: str, lowpass_cutoff=4.0, target_hz=100.0, butter_order=4):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = target_hz
    lowpass_cutoff = lowpass_cutoff 
    butter_order = butter_order
    
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
    print(f"Loading Zarr (Filtered): {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 4.0 
    butter_order = 4
    
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
    
    target_hz, lowpass_cutoff, butter_order, pad_time = 100.0, 1.0, 4, 0.5 
    
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
            for i in range(3): states[:, i] = filtfilt(b, a, states[:, i], axis=0, padlen=pad_len)

        target_times_list.append(target_times)
        states_list.append(states)
        n_list += len(target_times)
        episode_ends_list.append(n_list)

    return (states_raw, times_raw, episode_ends_raw, np.concatenate(states_list, axis=0), 
            np.concatenate(target_times_list, axis=0), np.array(episode_ends_list, dtype=np.int64))

def load_data(dir_name):
    """
    Loads data from CSV files in the specified directory.
    Parameters:
        dir_name (str): Directory path containing the CSV files.
                        In each file expect:
                            Cols 0 & 1 to be X-Y position
                            Col 3 to be pen-pressure (only colecting data when it is >0)
                            Col 4 to be time
    Returns:
        position_filtered (list): List of position data arrays after filtering.
        velocity (list): List of velocity data arrays.
        time (list): List of time data arrays.

    Raises:
        ValueError: If the directory does not contain any CSV files.

    """

    # Get a list of files in the directory
    files = os.listdir(dir_name)
    # Filter only the CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    # Raise an error if no CSV files are found
    if not csv_files:
        raise ValueError('Must specify a directory to load the csv files from')
    # Extract block and trial information from file names
    blocks = []
    trials = []
    file_names = []
    for file_name in csv_files:
        file_names.append(file_name)
        match = re.search(r'tb_.*block(\d*)_trial(\d*).csv', file_name) #checking for correct file name
        block = int(match.group(1))
        trial = int(match.group(2))
        blocks.append(block)
        trials.append(trial)

    # We have lists of blocks and trials and looks for max to see how much blocks and trials we have in this folder
    max_block = max(blocks)
    max_trial = max(trials)
    position_filtered = []
    velocity = []
    time = []

    # Process data for each block and trial
    for block in range(1, max_block + 1):
        for trial in range(1, max_trial + 1):
            trial_index = [i for i, (_block, _trial) in enumerate(zip(blocks, trials))
                           if _block == block and _trial == trial]
            if not trial_index:
                continue

            data = np.loadtxt(os.path.join(dir_name, csv_files[trial_index[0]]), delimiter=',')
            pressure = data[:, 3]
            position = data[pressure > 0, :2] / 1000
            _time = data[pressure > 0, 4] / 1000  # seconds
            _time = _time - _time[0]
            dt = np.median(np.diff(_time))
            b, a = butter(2, 5 / ((1 / dt) / 2))
            _position_filtered = filtfilt(b, a, position, axis=0)
            _velocity = np.vstack([[0, 0], np.diff(_position_filtered, axis=0) / dt])

            #organizing the data in correct variables for future functions
            time.append(_time)
            position_filtered.append(_position_filtered)
            velocity.append(_velocity)

    return position_filtered, velocity, time

# =============================================================================
# 2. MINIMUM JERK ALGORITHM & DYNAMIC RIDGE
# =============================================================================

def get_synthetic_params_cascaded(K):
    """
    Generates synthetic parameters (starts, ends, scales) using cascaded logic.
    """
    t_starts, t_ends = [], []
    last_t0 = np.random.uniform(0.04, 0.08)
    last_duration = np.random.uniform(0.3, 0.6) 
    last_t1 = last_t0 + last_duration
    t_starts.append(last_t0)
    t_ends.append(last_t1)
    
    for i in range(1, K):
        prev_midpoint = last_t0 + (last_t1 - last_t0) / 2.0
        delta_start = np.random.uniform(-0.0, 0.1)
        curr_t0 = prev_midpoint + delta_start
        desired_duration = np.random.uniform(0.3, 0.8)
        curr_t1 = curr_t0 + desired_duration
        if curr_t1 < last_t1 + 0.1:
            curr_t1 = last_t1 + np.random.uniform(0.1, 0.3)
        t_starts.append(curr_t0)
        t_ends.append(curr_t1)
        last_t0, last_t1 = curr_t0, curr_t1

    mag_x = np.random.uniform(50.0, 300.0, K)
    sign_x = np.random.choice([-1, 1], K)
    scales_x = mag_x * sign_x
    
    mag_y = np.random.uniform(50.0, 300.0, K)
    sign_y = np.random.choice([-1, 1], K)
    scales_y = mag_y * sign_y
    p0 = np.random.uniform(100, 400, 2)
    return np.array(t_starts), np.array(t_ends), scales_x, scales_y, p0

def compute_minjerk_base(t, t0, t1):
    t = np.asarray(t)
    base = np.zeros_like(t)
    D = t1 - t0
    if D <= 1e-6: return base
    tau = (t - t0) / D
    mask = (tau >= 0) & (tau <= 1)
    if not np.any(mask): return base
    tau_val = tau[mask]
    base[mask] = (1.0 / D) * (30 * tau_val**2 - 60 * tau_val**3 + 30 * tau_val**4)
    return base

def build_phi_matrix(t, t_starts, t_ends):
    K, N = t_starts.size, t.size
    Phi = np.zeros((N, K))
    for k in range(K):
        if t_ends[k] > t_starts[k]:
            Phi[:, k] = compute_minjerk_base(t, t_starts[k], t_ends[k])
    return Phi

def solve_ridge_weighted(A, b, alpha=0.1):
    if alpha <= 1e-9:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x
    N_features = A.shape[1]
    Lambda = np.eye(N_features) * alpha
    Lambda[-1, -1] = 0.0 
    AtA, Atb = A.T @ A, A.T @ b
    try: x = np.linalg.solve(AtA + Lambda, Atb)
    except np.linalg.LinAlgError: x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x

""" def compute_dynamic_ridge_alpha(Phi_vel, kappa_max=100.0):
    K = Phi_vel.shape[1]
    if K < 2: return 0.0 
    norms = np.linalg.norm(Phi_vel, axis=0)
    norms[norms == 0] = 1e-12 
    sigma2 = np.mean(norms**2)
    Phi_norm = Phi_vel / norms
    G = Phi_norm.T @ Phi_norm
    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min, lambda_max = max(0.0, np.min(eigenvalues)), np.max(eigenvalues)
    kappa_current = lambda_max / lambda_min if lambda_min > 1e-12 else np.inf
    if kappa_current <= kappa_max: return 0.0
    lambda_star = sigma2 * (lambda_max - kappa_max * lambda_min) / (kappa_max - 1)
    return max(0.0, lambda_star * 0.01) """
def compute_dynamic_ridge_alpha(Phi_vel, kappa_max=100.0):
    K = Phi_vel.shape[1]
    if K < 2: return 0.0 
    N_active_per_base = np.sum(Phi_vel > 1e-6, axis=0)
    N_active_per_base[N_active_per_base == 0] = 1 
    
    energy_per_base = np.sum(Phi_vel**2, axis=0) / N_active_per_base
    sigma2 = np.mean(energy_per_base)
    
    norms = np.linalg.norm(Phi_vel, axis=0)
    norms[norms == 0] = 1e-12 
    Phi_norm = Phi_vel / norms
    G = Phi_norm.T @ Phi_norm
    
    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min = max(0.0, np.min(eigenvalues))
    lambda_max = np.max(eigenvalues)
    
    kappa_current = lambda_max / lambda_min if lambda_min > 1e-12 else np.inf
    
    if kappa_current <= kappa_max: 
        return 0.0
    denominator = max(kappa_max - 1, 1e-6)
    lambda_star = sigma2 * (lambda_max - kappa_max * lambda_min) / denominator
    return max(0.0, lambda_star)

def fit_hybrid_peaks_greedy_minjerk(t, frefs, max_bases_total=100, residual_tol=0.03, verbose=False):
    if isinstance(frefs, list): F = np.column_stack(frefs)
    else: F = np.asarray(frefs)
    if F.ndim == 1: F = F.reshape(-1, 1)

    vt = np.linalg.norm(F, axis=1)
    vt = np.nan_to_num(vt)
    max_vt = np.max(vt)
    N = t.size
    dt = (t[-1] - t[0]) / (N - 1) if N > 1 else 0.01
    
    starts_found, ends_found = [], []
    
    # --- PHASE A: Initial Detection ---
    if verbose: print("  [MinJerk] Phase A: Detecting main peaks...")
    dist_param = max(1, N // 60) 
    peaks_indices, _ = find_peaks(vt, height=max_vt * 0.05, distance=dist_param)
    if verbose: print(f"    Found {len(peaks_indices)} initial peaks.")
    
    if len(peaks_indices) > 0:
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.85)
        for i, idx in enumerate(peaks_indices):
            t_left = left_ips[i] * dt + t[0]
            t_right = right_ips[i] * dt + t[0]
            center_time = (t_left + t_right) / 2.0
            width_fwhm = t_right - t_left
            
            duration = np.clip(width_fwhm * 1.8, 0.2, 3.5)
            starts_found.append(center_time - (duration / 2.0))
            ends_found.append(center_time + (duration / 2.0))

    if len(starts_found) > 0:
        Phi = build_phi_matrix(t, np.array(starts_found), np.array(ends_found))
        s, _ = nnls(Phi, vt)
        residual = vt - Phi.dot(s)
    else:
        residual = vt.copy()

    rms = np.sqrt(np.mean(residual**2))
    if verbose: print(f"    Initial RMS after Phase A: {rms:.4f} and error to avoid adding more bases: {max_vt * residual_tol:.4f}")
    
    # --- PHASE B: Grid Search ---
    candidate_durations = np.arange(0.2, 3.5, 0.01)
    
    if rms > max_vt * residual_tol:
        if verbose: print(f"  [MinJerk] Phase B: Grid Search Duration...")
        
        for k in range(max_bases_total - len(starts_found)):
            idx_max = np.argmax(residual)
            if residual[idx_max] < max_vt * residual_tol: break
            t_res_peak = t[idx_max]

            if len(starts_found) > 0:
                centers = (np.array(starts_found) + np.array(ends_found)) / 2.0
                if np.min(np.abs(centers - t_res_peak)) < 0.05:
                    residual[max(0, idx_max-10):min(N, idx_max+10)] = 0
                    continue
            
            best_dur = None
            best_score = -np.inf
            
            for dur in candidate_durations:
                t0_try = t_res_peak - dur/2.0
                t1_try = t_res_peak + dur/2.0
                
                is_nested = False
                for s_ex, e_ex in zip(starts_found, ends_found):
                    if (s_ex <= t0_try + 0.1) and (e_ex >= t1_try - 0.1):
                        is_nested = True; break
                if is_nested: continue 

                base_tmp = compute_minjerk_base(t, t0_try, t1_try)
                dot_prod = np.dot(residual, base_tmp)
                if dot_prod <= 0: continue
                    
                norm_sq = np.dot(base_tmp, base_tmp) + 1e-9
                score = (dot_prod**2) / norm_sq
                
                if score > best_score:
                    best_score = score
                    best_dur = dur
            
            if best_dur is None: 
                residual[max(0, idx_max-10):min(N, idx_max+10)] = 0
                continue

            starts_found.append(t_res_peak - (best_dur / 2.0))
            ends_found.append(t_res_peak + (best_dur / 2.0))
            
            Phi_new = build_phi_matrix(t, np.array(starts_found), np.array(ends_found))
            s_new, _ = nnls(Phi_new, vt)
            residual = vt - Phi_new.dot(s_new)

    return np.array(starts_found), np.array(ends_found)

def compute_position_error_minjerk(params, t, P_ref):
    """
    Dimension-agnostic error function (works for 2D and 3D) with Dynamic Ridge.
    """
    N, D = P_ref.shape 
    K = params.size // 2
    t_starts, t_ends = params[:K], params[K:]
    MAX_DURATION = 5.0 
    
    if np.any(t_starts >= t_ends - 0.02) or np.any((t_ends - t_starts) > MAX_DURATION): return 1e12

    Phi_vel = build_phi_matrix(t, t_starts, t_ends)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(K): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)

    alpha_dynamic = compute_dynamic_ridge_alpha(Phi_pos, kappa_max=100.0)
    A = np.hstack([Phi_pos, np.ones((N, 1))])

    total_error = 0
    for d in range(D):
        w = solve_ridge_weighted(A, P_ref[:, d], alpha=alpha_dynamic)
        fit_d = A.dot(w)
        total_error += np.sum((fit_d - P_ref[:, d])**2)
        
    return total_error

def refine_bases_minjerk(t, P_ref, t_starts_init, t_ends_init, verbose=False, step_title="Global Opt"):
    print(f"\n{step_title}: Optimizing {t_starts_init.size} bases (Dynamic Ridge)...")
    K = t_starts_init.size
    params_init = np.concatenate([t_starts_init, t_ends_init])
    bounds = [(t[0]-0.5, t[-1]+0.5)] * (2 * K)
        
    t0 = time.time()
    res = minimize(compute_position_error_minjerk, params_init, args=(t, P_ref),
                method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'disp': verbose})
    
    if verbose: print(f"  Done in {time.time()-t0:.2f}s. Cost: {res.fun:.4e}")
    params_opt = res.x
    ts, te = params_opt[:K], params_opt[K:]
    idx = np.argsort(ts)
    return ts[idx], te[idx]

def merge_bases_minjerk(t_starts, t_ends, scales, proximity_tol=0.05):
    """
    Dimension-agnostic function to merge submovements.
    """
    K, D = scales.shape
    if K < 2: return t_starts, t_ends, scales
    
    centers, durations = (t_starts + t_ends) / 2.0, t_ends - t_starts
    new_ts, new_te, new_scales = [], [], []
    visited = np.zeros(K, dtype=bool)
    
    for i in range(K):
        if visited[i]: continue
        group = [i]
        visited[i] = True
        
        for j in range(i+1, K):
            if not visited[j] and abs(centers[i] - centers[j]) < proximity_tol:
                if max(durations[i], durations[j]) / min(durations[i], durations[j]) < 1.5:
                    group.append(j)
                    visited[j] = True
                    
        idx_g = np.array(group)
        mags = np.linalg.norm(scales[idx_g], axis=1)
        total_mag = np.sum(mags) + 1e-9
        
        new_ts.append(np.sum(t_starts[idx_g] * mags) / total_mag)
        new_te.append(np.sum(t_ends[idx_g] * mags) / total_mag)
        new_scales.append(np.sum(scales[idx_g], axis=0))
        
    return np.array(new_ts), np.array(new_te), np.array(new_scales)

# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def plot_velocity_and_bases(t, v_ref, v_fit, t_starts, t_ends, scales_dim, dim_color='tab:red', title='Velocity dim'):
    N = t.size
    Phi = build_phi_matrix(t, t_starts, t_ends) if t_starts.size > 0 else np.zeros((N, 0))
    scaled = Phi * scales_dim[None, :] if Phi.size > 0 else np.zeros((N, 0))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='Reconstructed')

    has_base_label = False
    for k in range(scaled.shape[1]):
        plt.fill_between(t, 0, scaled[:, k], alpha=0.3, color=dim_color, label='MinJerk Bases' if not has_base_label else None, linewidth=0)
        plt.plot(t, scaled[:, k], color=dim_color, alpha=0.8, lw=1) 
        has_base_label = True
        
    plt.legend(loc='upper right', ncol=3, fontsize='small')
    plt.xlabel('Time (t)'); plt.ylabel('Velocity'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots/Plots_AutomaticRidge_MultiDataset", title.replace("\n", "").replace(":", "").replace(" ", "_") + ".png"))
    plt.close()

def plot_velocity_and_bases_signed(t, v_ref, v_fit, t_starts, t_ends, Phi_vel, scales_dim, pos_color='tab:red', neg_color='tab:blue', title='Velocity Decomposition'):
    N = t.size
    scaled_vel = Phi_vel * scales_dim[None, :] if Phi_vel.size > 0 else np.zeros((N, 0))
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='v_ref')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='v_fit')
    
    has_pos, has_neg = False, False
    for k in range(scaled_vel.shape[1]):
        curve = scaled_vel[:, k]
        if np.any(curve >= 0):
            plt.fill_between(t, 0, curve, where=curve>=0, facecolor=pos_color, alpha=0.3, interpolate=True, label='Pos' if not has_pos else None)
            if not has_pos: has_pos = True
            plt.plot(t, np.ma.masked_less(curve, 0), color=pos_color, lw=1, alpha=0.6)
        if np.any(curve < 0):
            plt.fill_between(t, 0, curve, where=curve<0, facecolor=neg_color, alpha=0.3, interpolate=True, label='Neg' if not has_neg else None)
            if not has_neg: has_neg = True
            plt.plot(t, np.ma.masked_greater(curve, 0), color=neg_color, lw=1, alpha=0.6)

    plt.xlabel('Time (s)'); plt.ylabel('Velocity'); plt.title(title)
    plt.legend(loc='upper right', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("Plots/Plots_AutomaticRidge_MultiDataset", title.replace(" ", "_") + ".png"))
    plt.close()

def plot_position_and_integrated_bases(t, p_ref, p_fit, t_starts, t_ends, Phi_pos, scales_dim, pos_color='tab:red', neg_color='tab:blue', title='Position dim'):
    N = t.size
    scaled_pos = Phi_pos * scales_dim[None, :] if Phi_pos.size > 0 else np.zeros((N, 0))
    plt.figure(figsize=(10, 5))
    plt.plot(t, p_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, p_fit, '--', lw=2, color='green', label='Reconstructed')
    
    for k in range(scaled_pos.shape[1]):
        if np.any(scaled_pos[:, k] >= 0): plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]>=0, facecolor=pos_color, alpha=0.2)
        if np.any(scaled_pos[:, k] < 0): plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]<0, facecolor=neg_color, alpha=0.2)

    plt.xlabel('Time (t)'); plt.ylabel('Position'); plt.title(title)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots/Plots_AutomaticRidge_MultiDataset", title.replace(" ", "_") + ".png"))
    plt.close()

def plot_3d_trajectory_pos(P_ref, P_fit):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P_ref[:,0], P_ref[:,1], P_ref[:,2], 'k-', label='Original', lw=1)
    ax.plot(P_fit[:,0], P_fit[:,1], P_fit[:,2], 'r--', label='MinJerk Fit', lw=2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Trajectory: Real vs Fitted')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("Plots/Plots_AutomaticRidge_MultiDataset", "3D_Trajectory.png"))
    plt.close()

def plot_2d_trajectory_pos(P_ref, P_fit):
    plt.figure(figsize=(6, 6))
    plt.plot(P_ref[:,0], P_ref[:,1], 'k-', lw=2, label='Original')
    plt.plot(P_fit[:,0], P_fit[:,1], 'r--', lw=2, label='Fitted')
    plt.legend()
    plt.title('2D Trajectory: Real vs Fitted')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots/Plots_AutomaticRidge_MultiDataset", "2D_Trajectory.png"))
    plt.close()


# =============================================================================
# 4. MAIN - AUTOMATIC PROCESSING
# =============================================================================

if __name__ == '__main__':
    
    # --- SELECT DATA ---
    DATA_SOURCE = 'LETTERS' # 'PUSHT' or 'PUSHTReal2d' or 'SUBJECT' or 'PUSHTReal3d' or 'SYNTHETIC' or 'LETTERS' or 'MOVING3D' or 'SPATULA'
    
    PUSHT_SIMULATED = "data/pusht_real/real_pusht_20230105/replay_buffer.zarr"
    PUSHT_REAL_2D = "data/EPFL_data/adrian_adrian2D_2026-03-13-13-50/data.zarr"
    SUBJECT_STROKE = "data/subject_stroke/subject08day1post"
    PUSHT_REAL_3D = "data/EPFL_data/adrian_adrian3D_2026-03-13-16-04/data.zarr"
    MOVING_DATA = "data/moving_object/object_moving_tangential_velocity_data.csv"
    LETTERS_PATH = "data/Handwriting/character_A_minjerk.zarr" #!Here change the letter
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
        K_TARGET = 3
        
        ts_gt, te_gt, sx_gt, sy_gt, p0_gt = get_synthetic_params_cascaded(K_TARGET)
        
        t = np.linspace(0, DURATION, N_SAMPLES)
        Phi_gt = build_phi_matrix(t, ts_gt, te_gt)
        
        vx = Phi_gt.dot(sx_gt)
        vy = Phi_gt.dot(sy_gt)
        vt = np.sqrt(vx**2 + vy**2)
        
        xs = cumtrapz(vx, t, initial=0.0) + p0_gt[0]
        ys = cumtrapz(vy, t, initial=0.0) + p0_gt[1]
        P = np.column_stack([xs, ys])
        
        V_ref = np.zeros_like(P)
        for i in range(2):
            V_ref[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref, axis=1)
        
        print(f"Generated {K_TARGET} submovements over {DURATION}s.")
        
    elif DATA_SOURCE == 'PUSHT':
        #if not os.path.exists(PUSHT_SIMULATED): exit(f"Zarr not found at {PUSHT_SIMULATED}")
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
        
        V_ref = np.zeros_like(P)
        for i in range(2):
            V_ref[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref, axis=1)
        
    elif DATA_SOURCE == 'PUSHTReal2d':
        (states_raw, times_raw, episode_ends_raw, 
        states_processed_all, target_times_all, episode_ends_processed) = import_data(PUSHT_REAL_2D)
        
        EPISODE_IDX = 0
        start_idx = 0 if EPISODE_IDX == 0 else episode_ends_processed[EPISODE_IDX-1]
        end_idx = episode_ends_processed[EPISODE_IDX]
        
        episode_data = states_processed_all[start_idx:end_idx, :].astype(float)
        t_raw = target_times_all[start_idx:end_idx]
        t = t_raw - t_raw[0]
        
        N = t.size
        xs = episode_data[:, 0] * 100
        ys = episode_data[:, 1] * 100
            
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
        vt = np.sqrt(vx**2 + vy**2)
        
        P = np.column_stack([xs, ys])
        
        V_ref = np.zeros_like(P)
        for i in range(2):
            V_ref[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref, axis=1)
        print(f"Loaded Episode {EPISODE_IDX}. Samples: {N}.")
    
    elif DATA_SOURCE == 'SPATULA' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data(SPATULA_PATH,lowpass_cutoff=1.0)
        
        EPISODE_IDX = 1
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
        
        vt_ref = np.linalg.norm(V_ref_3d, axis=1)
        
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]
        
    elif DATA_SOURCE == 'SUBJECT':
        position_filtered, velocity_list, time_list = load_data(SUBJECT_STROKE)
        #if position_filtered is None or not position_filtered: exit("No data loaded.")
        EPISODE_IDX = 1
        P_raw = position_filtered[EPISODE_IDX]
        V = velocity_list[EPISODE_IDX]
        t_raw = time_list[EPISODE_IDX]
        t = t_raw - t_raw[0]
        xs, ys = P_raw[:, 0], P_raw[:, 1]
        vx, vy = V[:, 0], V[:, 1]
        vt = np.sqrt(vx**2 + vy**2)
        P = np.column_stack([xs, ys])
        
        V_ref = np.zeros_like(P)
        for i in range(2):
            V_ref[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref, axis=1)
        
    elif DATA_SOURCE == 'LETTERS':
        print(f"Loading Zarr: {LETTERS_PATH}")
        #if not os.path.exists(LETTERS_PATH): exit(f"Zarr not found at {LETTERS_PATH}")
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
        vt_ref = np.linalg.norm(V_ref_2d, axis=1)
            
        vt = np.linalg.norm(V_ref_2d, axis=1)
        vx = V_ref_2d[:, 0]
        vy = V_ref_2d[:, 1]
        print("Valores de vx y vy (primeros 5):", vx[:5], vy[:5])
        
    elif DATA_SOURCE == 'PUSHTReal3d' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data_filtered(PUSHT_REAL_3D)
        
        EPISODE_IDX = 1
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
        
        vt_ref = np.linalg.norm(V_ref_3d, axis=1)
        
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]
        
    elif DATA_SOURCE == 'MOVING3D' :
        (states_raw, times_raw, episode_ends_raw, 
        states_proc, times_proc, episode_ends_proc) = import_data_CSV(MOVING_DATA)
        
        EPISODE_IDX = 3
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
        vt_ref = np.linalg.norm(V_ref_3d, axis=1)
        
        vt = np.linalg.norm(V_ref_3d, axis=1)
        vx = V_ref_3d[:, 0]
        vy = V_ref_3d[:, 1]
        vz = V_ref_3d[:, 2]

    # --- SETUP V_ref BASED ON DIMENSIONS ---
    is_3d = (zs is not None)
    if is_3d:
        V_ref = np.column_stack([vx, vy, vz])
    else:
        V_ref = np.column_stack([vx, vy])

    D_dim = P.shape[1]
    print(f"  -> Detected Spatial Dimension: {D_dim}D")

    # --- MINJERK ALGORITHM START (Dynamic and Dimensionally Agnostic) ---
    print("\nSTEP 1: Getting initial guess (MinJerk Greedy)...")
    t_init_contador = time.time()
    
    # 1. Greedy Initialization (Uses velocities of each dimension)
    frefs = [V_ref[:, d] for d in range(D_dim)]
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, frefs, max_bases_total=1000, residual_tol=0.03, verbose=True)

    # 2. Global Optimization (Uses the dimension-agnostic function)
    ts_opt, te_opt = refine_bases_minjerk(t, P, ts_init, te_init, verbose=False)
    
    # 3. Merging (Calculates temporal scales with Dynamic Ridge)
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    alpha_dynamic_temp = compute_dynamic_ridge_alpha(A_temp[:, :-1], kappa_max=100.0)
    print(f"Pre-Merge Dynamic Ridge Alpha: {alpha_dynamic_temp}")
    
    scales_list_temp = []
    for d in range(D_dim):
        w = solve_ridge_weighted(A_temp, P[:, d], alpha=alpha_dynamic_temp)
        scales_list_temp.append(w[:-1])
    scales_temp = np.column_stack(scales_list_temp)
    
    print("\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
    ts_final, te_final = refine_bases_minjerk(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    
    K_final = ts_final.size
    
    # 4. Final Reconstruction
    Phi_vel_final = build_phi_matrix(t, ts_final, te_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final): Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)

    A_final = np.hstack([Phi_pos_final, np.ones((len(t), 1))])
    alpha_dynamic_final = compute_dynamic_ridge_alpha(A_final[:, :-1], kappa_max=100.0)
    print(f"Final Dynamic Ridge Alpha: {alpha_dynamic_final}")
    
    scales_list_final = []
    p0_list = []
    
    for d in range(D_dim):
        w = solve_ridge_weighted(A_final, P[:, d], alpha=alpha_dynamic_final)
        scales_list_final.append(w[:-1])
        p0_list.append(w[-1]) 
        
    scales_final = np.column_stack(scales_list_final)
    
    P_fit = np.zeros_like(P)
    for d in range(D_dim):
        w_full = np.concatenate([scales_list_final[d], [p0_list[d]]])
        P_fit[:, d] = A_final.dot(w_full)
    
    V_fit = Phi_vel_final.dot(scales_final)
    vt_fit = np.linalg.norm(V_fit, axis=1)
    scales_mag = np.linalg.norm(scales_final, axis=1)
    
    print(f"Total Time: {time.time() - t_init_contador:.2f}s")
    rmse = np.sqrt(np.mean(np.sum((P - P_fit)**2, axis=1)))
    print(f"RMSE {D_dim}D: {rmse:.5f}")

    # --- PLOTS ---
    print(f"\nGenerating Plots (MinJerk {D_dim}D)...")
    
    dim_names = ['X', 'Y', 'Z'] if is_3d else ['X', 'Y']
    pos_colors = ['tab:red', 'tab:green', 'tab:orange']
    neg_colors = ['tab:cyan', 'tab:purple', 'tab:blue']
    
    for d in range(D_dim):
        plot_position_and_integrated_bases(t, P[:, d], P_fit[:, d], ts_final, te_final, Phi_pos_final, scales_final[:, d], 
                                        pos_color=pos_colors[d], neg_color=neg_colors[d], title=f'Pos {dim_names[d]} (MinJerk)')
        plot_velocity_and_bases_signed(t, V_ref[:, d], V_fit[:, d], ts_final, te_final, Phi_vel_final, scales_final[:, d], 
                                    pos_color=pos_colors[d], neg_color=neg_colors[d], title=f'Vel {dim_names[d]} (MinJerk)')

    plot_velocity_and_bases(t, vt_ref, vt_fit, ts_final, te_final, scales_mag, 
                            dim_color='tab:orange', title='Final Tangential Velocity (MinJerk)')

    if is_3d:
        plot_3d_trajectory_pos(P, P_fit)
    else:
        plot_2d_trajectory_pos(P, P_fit)
    
    print(f"\nAll plots saved in 'Plots/Plots_AutomaticRidge_MultiDataset'")

    print(f"\nFinal Parameters (MinJerk {D_dim}D):")
    if is_3d:
        print(f"{'ID':<3} | {'Start':<8} | {'End':<8} | {'Dur':<8} | {'Sx':<8} | {'Sy':<8} | {'Sz':<8}")
    else:
        print(f"{'ID':<3} | {'Start':<8} | {'End':<8} | {'Dur':<8} | {'Sx':<8} | {'Sy':<8}")
        
    sort_idx = np.argsort(ts_final)
    for k in range(K_final):
        idx = sort_idx[k]
        dur = te_final[idx] - ts_final[idx]
        row = [k, ts_final[idx], te_final[idx], dur] + [scales_final[idx, d] for d in range(D_dim)]
        if is_3d:
            print(f"{row[0]:<3} | {row[1]:.3f}    | {row[2]:.3f}    | {row[3]:.3f}    | {row[4]:.3f} | {row[5]:.3f}    | {row[6]:.3f}")
        else:
            print(f"{row[0]:<3} | {row[1]:.3f}    | {row[2]:.3f}    | {row[3]:.3f}    | {row[4]:.3f} | {row[5]:.3f}")