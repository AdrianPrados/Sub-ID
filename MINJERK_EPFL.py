import numpy as np
import matplotlib.pyplot as plt
import time
import os
import zarr 
from pathlib import Path
from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.signal import find_peaks, peak_widths, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# --- CONFIGURACIÓN ---
if not os.path.exists('Plots_EPFL'):
    os.makedirs('Plots_EPFL')

np.set_printoptions(precision=4, suppress=True)

# REGULARIZATION FACTOR (L2 / RIDGE)
# Higher value = smaller and more stable scales (less overfitting).
# Value 0.0 = standard least squares (original behavior).
RIDGE_ALPHA = 0.2

# -----------------------------------------------------------------------------
# 0. DATA IMPORT & FILTERING (NEW)
# -----------------------------------------------------------------------------

def import_data(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 4 #! Generated with the metrics of velocity_profileEPFL.py
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    
    # Check for state key (adapted to support your specific state1 or generic state) 
    states_raw = zarr_data["data/flat_spatula"][:]  
    
    # Force float64 from the beginning for the whole time matrix
    times_raw = np.asarray(zarr_data["data/time"][:], dtype=np.float64).ravel()
    
    dt = 1.0 / float(target_hz)
    state_dim = states_raw.shape[1]

    target_times_list = []
    states_list = []
    episode_ends_list = []
    n_list = 0

    for trial in range(episode_ends_raw.shape[0]):
        if trial == 0: 
            dex_start = 0
        else: 
            dex_start = episode_ends_raw[trial - 1]
            
        dex_end = episode_ends_raw[trial] 
        dex_states = np.arange(dex_start, dex_end)

        # --- VECTORIZED SOLUTION: ABSOLUTE MONOTONICITY ---
        t_episode = times_raw[dex_states].copy()
        
        # 1. Temporarily normalize to zero (saves us if times are Unix Epoch).
        t_zero = t_episode[0]
        t_episode = t_episode - t_zero
        
        # 2. Prevent time from going backward
        t_episode = np.maximum.accumulate(t_episode)
        
        # 3. Force strict growth by adding a microscopic total slope of 1 millisecond.
        t_episode = t_episode + np.linspace(0, 1e-3, len(t_episode))
        
        # 4. Restore original base time.
        t_episode = t_episode + t_zero
        # ------------------------------------------------

        # Create target_times
        t_start_ep = t_episode[0]
        t_end_ep = t_episode[-1]
        target_times = np.arange(t_start_ep + dt, t_end_ep - dt, dt)
        
        states = np.zeros((len(target_times), state_dim))

        # 1. POSITION INTERPOLATION (Adapts to 2D Push-T or 3D datasets)
        dim_pos = min(3, state_dim)
        for i in range(dim_pos):
            f_interp = interp1d(t_episode, states_raw[dex_states, i], kind="cubic", assume_sorted=True)
            states[:, i] = f_interp(target_times)

        # 2. ROTATION INTERPOLATION (Only if dataset has at least 7 dimensions)
        """ if state_dim >= 7:
            rot_in = Rotation.from_quat(states_raw[dex_states, 3:7])
            slerp = Slerp(t_episode, rot_in)
            states[:, 3:7] = slerp(target_times).as_quat() """

        # 3. BUTTERWORTH FILTERING
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

    return (states_raw, times_raw, episode_ends_raw, 
            states_processed_all, target_times_all, episode_ends_processed)


def plot_basis_histograms(ts, te, scales_mag, out_dir="Plots_EPFL"):
    """
    Plotea la distribución biológica de las primitivas extraídas:
    Duraciones, Amplitudes espaciales y Periodos Refractarios.
    """
    # 1. Ordenar por tiempo de inicio para que la secuencia temporal sea correcta
    sort_idx = np.argsort(ts)
    ts_sorted = ts[sort_idx]
    te_sorted = te[sort_idx]
    mags_sorted = scales_mag[sort_idx]

    # 2. Calcular las métricas
    durations = te_sorted - ts_sorted
    amplitudes = mags_sorted
    
    # Periodo refractario (Tiempo entre el inicio del movimiento N y el N+1)
    if len(ts_sorted) > 1:
        refractory_periods = np.diff(ts_sorted)
    else:
        refractory_periods = np.array([])

    # 3. Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Histograma 1: Duraciones ---
    axes[0].hist(durations, bins=15, color='tab:blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(durations), color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {np.mean(durations):.2f}s')
    axes[0].set_title('Submovement Durations')
    axes[0].set_xlabel('Duration (s)')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # --- Histograma 2: Amplitudes (Escala Espacial) ---
    axes[1].hist(amplitudes, bins=15, color='tab:green', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(amplitudes), color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {np.mean(amplitudes):.4f}m')
    axes[1].set_title('Submovement Amplitudes (Scale)')
    axes[1].set_xlabel('Amplitude Magnitude')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    # --- Histograma 3: Periodos Refractarios ---
    if len(refractory_periods) > 0:
        axes[2].hist(refractory_periods, bins=15, color='tab:orange', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(refractory_periods), color='red', linestyle='dashed', linewidth=2, 
                        label=f'Mean: {np.mean(refractory_periods):.2f}s')
    axes[2].set_title('Refractory Periods (Start-to-Start)')
    axes[2].set_xlabel('Time Interval (s)')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    if len(refractory_periods) > 0: axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Basis_Histograms.svg"))
    plt.close()
    print("  > Histogramas de primitivas guardados como 'Basis_Histograms.svg'")


# -----------------------------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS CORE (MINIMUM JERK)
# -----------------------------------------------------------------------------

def compute_minjerk_base(t, t0, t1):
    t = np.asarray(t)
    base = np.zeros_like(t)
    
    D = t1 - t0
    if D <= 1e-6: return base
    
    # Tau [0, 1]
    tau = (t - t0) / D
    mask = (tau >= 0) & (tau <= 1)
    
    if not np.any(mask): return base
    
    tau_val = tau[mask]
    
    # MinKerj Polonomyal
    val = (1.0 / D) * (30 * tau_val**2 - 60 * tau_val**3 + 30 * tau_val**4)
    
    base[mask] = val
    return base

def build_phi_matrix(t, t_starts, t_ends):
    """
    Construye la matriz Phi para MinJerks.
    """
    K = t_starts.size
    N = t.size
    Phi = np.zeros((N, K))
    
    for k in range(K):
        if t_ends[k] > t_starts[k]:
            Phi[:, k] = compute_minjerk_base(t, t_starts[k], t_ends[k])
    return Phi

def solve_ridge_weighted(A, b, alpha=0.1):
    """
    Solves Ax = b using Ridge Regression (L2 Regularization).
    Prevents giant coefficients that cancel each other out.
    """
    if alpha <= 1e-9:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x

    N_features = A.shape[1]
    
    # Matriz de Regularización (Identidad escalada por alpha)
    Lambda = np.eye(N_features) * alpha
    Lambda[-1, -1] = 0.0 
    
    # Ecuación Normal Regularizada
    AtA = A.T @ A
    Atb = A.T @ b
    
    try:
        x = np.linalg.solve(AtA + Lambda, Atb)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
    return x

# -----------------------------------------------------------------------------
# 2. ALGORITMOS DE AJUSTE
# -----------------------------------------------------------------------------

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
    
    # --- FASE A: Detección Inicial ---
    if verbose: print("  [MinJerk] Fase A: Detecting main peaks...")
    dist_param = max(1, N // 60) 
    peaks_indices, _ = find_peaks(vt, height=max_vt * 0.05, distance=dist_param)
    if verbose: print(f"    Found {len(peaks_indices)} initial peaks.")
    
    if len(peaks_indices) > 0:
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.7)
        for i, idx in enumerate(peaks_indices):
            t_left = left_ips[i] * dt + t[0]
            t_right = right_ips[i] * dt + t[0]
            center_time = (t_left + t_right) / 2.0
            width_fwhm = t_right - t_left
            
            duration = np.clip(width_fwhm * 1.8, 0.4, 3.5)
            starts_found.append(center_time - (duration / 2.0))
            ends_found.append(center_time + (duration / 2.0))

    if len(starts_found) > 0:
        Phi = build_phi_matrix(t, np.array(starts_found), np.array(ends_found))
        s, _ = nnls(Phi, vt)
        residual = vt - Phi.dot(s)
    else:
        residual = vt.copy()

    rms = np.sqrt(np.mean(residual**2))
    if verbose: print(f"    Initial RMS after Fase A: {rms:.4f} and error to avoid adding more bases: {max_vt * residual_tol:.4f}")
    
    # --- FASE B: Grid Search ---
    candidate_durations = np.arange(0.4, 1.5, 0.01)
    
    if rms > max_vt * residual_tol:
        if verbose: print(f"  [MinJerk] Fase B: Grid Search Duration...")
        
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

def compute_position_error_minjerk_regularized(params, t, P_ref):
    N, D = P_ref.shape
    K = params.size // 2
    t_starts = params[:K]
    t_ends = params[K:]
    
    MAX_DURATION = 5.0 
    
    if np.any(t_starts >= t_ends - 0.02): return 1e12
    if np.any((t_ends - t_starts) > MAX_DURATION): return 1e12

    Phi_vel = build_phi_matrix(t, t_starts, t_ends)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(K):
        Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
        
    A = np.hstack([Phi_pos, np.ones((N, 1))])
    
    Sx = solve_ridge_weighted(A, P_ref[:, 0], alpha=RIDGE_ALPHA)
    Sy = solve_ridge_weighted(A, P_ref[:, 1], alpha=RIDGE_ALPHA)
    
    fit_x = A.dot(Sx)
    fit_y = A.dot(Sy)
    
    return np.sum((fit_x - P_ref[:,0])**2) + np.sum((fit_y - P_ref[:,1])**2)

def refine_bases_minjerk(t, P_ref, t_starts_init, t_ends_init, verbose=False, step_title="Global Opt"):
    print(f"\n{step_title}: Optimizing {t_starts_init.size} bases (Ridge Alpha={RIDGE_ALPHA})...")
    K = t_starts_init.size
    params_init = np.concatenate([t_starts_init, t_ends_init])
    bounds = []
    
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))       
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))   
    
    t0 = time.time()
    res = minimize(compute_position_error_minjerk_regularized, 
            params_init, 
            args=(t, P_ref),
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 20000, 'disp': verbose})
    
    if verbose: print(f"  Done in {time.time()-t0:.2f}s. Cost: {res.fun:.4e}")
    
    params_opt = res.x
    ts = params_opt[:K]
    te = params_opt[K:]
    idx = np.argsort(ts)
    return ts[idx], te[idx]

def merge_bases_minjerk(t_starts, t_ends, scales, proximity_tol=0.05):
    K, D = scales.shape
    if K < 2: return t_starts, t_ends, scales
    
    centers = (t_starts + t_ends) / 2.0
    durations = t_ends - t_starts
    
    new_ts, new_te, new_scales = [], [], []
    visited = np.zeros(K, dtype=bool)
    
    for i in range(K):
        if visited[i]: continue
        group = [i]
        visited[i] = True
        
        for j in range(i+1, K):
            if visited[j]: continue
            if abs(centers[i] - centers[j]) < proximity_tol:
                ratio = max(durations[i], durations[j]) / min(durations[i], durations[j])
                if ratio < 1.5:
                    group.append(j)
                    visited[j] = True
                    
        idx_g = np.array(group)
        mags = np.linalg.norm(scales[idx_g], axis=1)
        total_mag = np.sum(mags) + 1e-9
        
        w_ts = np.sum(t_starts[idx_g] * mags) / total_mag
        w_te = np.sum(t_ends[idx_g] * mags) / total_mag
        w_scales = np.sum(scales[idx_g], axis=0)
        
        new_ts.append(w_ts)
        new_te.append(w_te)
        new_scales.append(w_scales)
        
    return np.array(new_ts), np.array(new_te), np.array(new_scales)

# -----------------------------------------------------------------------------
# 4. PLOTTING HELPERS
# -----------------------------------------------------------------------------

def plot_velocity_and_bases(t, v_ref, v_fit, t_starts, t_ends, scales_dim, 
                            dim_color='tab:red', title='Velocity dim'):
    N = t.size
    Phi = build_phi_matrix(t, t_starts, t_ends) if t_starts.size > 0 else np.zeros((N, 0))
    scaled = Phi * scales_dim[None, :] if Phi.size > 0 else np.zeros((N, 0))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='Reconstructed')

    has_base_label = False
    for k in range(scaled.shape[1]):
        plt.fill_between(t, 0, scaled[:, k], alpha=0.3, color=dim_color,
                        label='MinJerk Bases' if not has_base_label else None, linewidth=0)
        plt.plot(t, scaled[:, k], color=dim_color, alpha=0.8, lw=1) 
        has_base_label = True
        
    plt.legend(loc='upper right', ncol=3, fontsize='small')
    plt.xlabel('Time (t)'); plt.ylabel('Velocity'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    filename = title.replace("\n", " ").replace(":", "").replace(" ", "_") + ".svg"
    plt.savefig(os.path.join("Plots_EPFL", filename))
    plt.close()

def plot_velocity_and_bases_signed(t, v_ref, v_fit, t_starts, t_ends, Phi_vel, scales_dim, 
                                pos_color='tab:red', neg_color='tab:blue', title='Velocity Decomposition'):
    N = t.size
    scaled_vel = Phi_vel * scales_dim[None, :] if Phi_vel.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='v_ref')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='v_fit')
    
    has_pos, has_neg = False, False
    for k in range(scaled_vel.shape[1]):
        curve = scaled_vel[:, k]
        if np.any(curve >= 0):
            plt.fill_between(t, 0, curve, where=curve>=0, facecolor=pos_color, alpha=0.3, interpolate=True,
                            label='Pos' if not has_pos else None)
            if not has_pos: has_pos = True
            plt.plot(t, np.ma.masked_less(curve, 0), color=pos_color, lw=1, alpha=0.6)
        if np.any(curve < 0):
            plt.fill_between(t, 0, curve, where=curve<0, facecolor=neg_color, alpha=0.3, interpolate=True,
                            label='Neg' if not has_neg else None)
            if not has_neg: has_neg = True
            plt.plot(t, np.ma.masked_greater(curve, 0), color=neg_color, lw=1, alpha=0.6)

    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)'); plt.title(title)
    plt.legend(loc='upper right', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("Plots_EPFL", title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_position_and_integrated_bases(t, p_ref, p_fit, t_starts, t_ends, Phi_pos, scales_dim, 
                                    pos_color='tab:red', neg_color='tab:blue', title='Position dim'):
    N = t.size
    scaled_pos = Phi_pos * scales_dim[None, :] if Phi_pos.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    plt.plot(t, p_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, p_fit, '--', lw=2, color='green', label='Reconstructed')
    
    for k in range(scaled_pos.shape[1]):
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]>=0, facecolor=pos_color, alpha=0.2)
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]<0, facecolor=neg_color, alpha=0.2)

    plt.xlabel('Time (t)'); plt.ylabel('Position'); plt.title(title)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots_EPFL", title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_2d_trajectory_pos(xs, ys, x_fit, y_fit):
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'k-', lw=2, label='Original')
    plt.plot(x_fit, y_fit, 'r--', lw=2, label='Fitted (Reg)')
    plt.legend()
    plt.title(f'2D Trajectory (Ridge alpha={RIDGE_ALPHA})')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots_EPFL", "2D_Trajectory_Reg.svg"))
    plt.close()

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # ZARR PATH
    #! 2D Adrian
    ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/EPFL_data/adrian_adrian2D_2026-03-13-13-50/data.zarr"
    #!2D James
    #ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/EPFL_data/adrian_james2D_2026-03-13-14-38/data.zarr"
    print(f"--- 1. Loading and Filtering Push-T Data from real users... ---")
    
    if os.path.exists(ZARR_PATH):
        # Using the new filtering function
        (states_raw, times_raw, episode_ends_raw, 
        states_processed_all, target_times_all, episode_ends_processed) = import_data(ZARR_PATH)
        
        EPISODE_IDX = 0
        start_idx = 0 if EPISODE_IDX == 0 else episode_ends_processed[EPISODE_IDX-1]
        end_idx = episode_ends_processed[EPISODE_IDX]
        
        # Extract processed (filtered and resampled) data for the specific episode
        episode_data = states_processed_all[start_idx:end_idx, :].astype(float)
        t_raw = target_times_all[start_idx:end_idx]
        t = t_raw - t_raw[0]
        
        #! Filtro temporal ---
        mask = t <= 15.0
        t = t[mask]
        episode_data = episode_data[mask, :]
        
        
        N = t.size
        
        # Values are already filtered through the Butter filter from import_data
        xs = episode_data[:, 0]
        ys = episode_data[:, 1]
            
        vx_ref = np.gradient(xs, t)
        vy_ref = np.gradient(ys, t)
        vt_ref = np.sqrt(vx_ref**2 + vy_ref**2)
        
        P = np.column_stack([xs, ys])
        print(f"Loaded Episode {EPISODE_IDX}. Samples: {N}.")
    else:
        print("ERROR: Zarr file not found.")
        exit()
    
    init_time = time.time()

    # --- 1. Greedy Init ---
    print("\nSTEP 1: Getting initial guess (MinJerk Greedy)...")
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vx_ref, vy_ref], max_bases_total=1000, residual_tol=0.02, verbose=True)

    # Plot Initial Guess
    Phi_init = build_phi_matrix(t, ts_init, te_init)
    s_init, _ = nnls(Phi_init, vt_ref)
    vt_fit_init = Phi_init.dot(s_init)
    
    plot_velocity_and_bases(t, vt_ref, vt_fit_init, ts_init, te_init, s_init, 
                            dim_color='tab:cyan', title='STEP 1b: Initial Guess (Fitted to vt_ref)')
    
    # --- 2. Global Optimization (CON RIDGE INTERNO) ---
    ts_opt, te_opt = refine_bases_minjerk(t, P, ts_init, te_init, verbose=False)
    
    # --- 3. Calcular Escalas Intermedias (CON RIDGE) ---
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    
    Sx_temp = solve_ridge_weighted(A_temp, P[:, 0], alpha=RIDGE_ALPHA)
    Sy_temp = solve_ridge_weighted(A_temp, P[:, 1], alpha=RIDGE_ALPHA)
    scales_temp = np.column_stack([Sx_temp[:-1], Sy_temp[:-1]])
    
    ts_final = ts_opt.copy()
    te_final = te_opt.copy()
    
    print(f"\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
    
    # Refinamiento Final
    ts_final, te_final = refine_bases_minjerk(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    K_final = ts_final.size
    
    # --- 4. Cálculo FINAL de escalas (CON RIDGE) ---
    Phi_vel_final = build_phi_matrix(t, ts_final, te_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final):
        Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)

    A_final = np.hstack([Phi_pos_final, np.ones((N, 1))])
    
    Sx_final = solve_ridge_weighted(A_final, P[:, 0], alpha=RIDGE_ALPHA)
    Sy_final = solve_ridge_weighted(A_final, P[:, 1], alpha=RIDGE_ALPHA)
    
    scales_x = Sx_final[:-1]
    scales_y = Sy_final[:-1]
    scales_final_mat = np.column_stack([scales_x, scales_y])
    
    # Reconstrucción
    fit_x = A_final.dot(Sx_final)
    fit_y = A_final.dot(Sy_final)
    P_fit = np.column_stack([fit_x, fit_y])
    
    V_fit = Phi_vel_final.dot(scales_final_mat)
    vx_fit, vy_fit = V_fit[:, 0], V_fit[:, 1]
    vt_fit = np.sqrt(vx_fit**2 + vy_fit**2)
    scales_mag = np.linalg.norm(scales_final_mat, axis=1)
    
    end_time = time.time()
    
    # --- PLOTS ---
    print("\nGenerating Plots...")
    plot_position_and_integrated_bases(t, xs, fit_x, ts_final, te_final, Phi_pos_final, scales_x, 
                                    pos_color='tab:red', neg_color='tab:cyan', title='Pos X (Regularized)')
    plot_position_and_integrated_bases(t, ys, fit_y, ts_final, te_final, Phi_pos_final, scales_y,
                                    pos_color='tab:green', neg_color='tab:purple', title='Pos Y (Regularized)')
    plot_velocity_and_bases_signed(t, vx_ref, vx_fit, ts_final, te_final, Phi_vel_final, scales_x, 
                                pos_color='tab:red', neg_color='tab:cyan', title='Vel X (Regularized)')
    plot_velocity_and_bases_signed(t, vy_ref, vy_fit, ts_final, te_final, Phi_vel_final, scales_y, 
                                pos_color='tab:green', neg_color='tab:purple', title='Vel Y (Regularized)')
    plot_velocity_and_bases(t, vt_ref, vt_fit, ts_final, te_final, scales_mag, 
                            dim_color='tab:orange', title='Final Tangential Velocity (Regularized)')
    plot_2d_trajectory_pos(xs, ys, fit_x, fit_y)
    
    plot_basis_histograms(ts_final, te_final, scales_mag)
    
    rmse_pos = np.sqrt(np.mean((P - P_fit)**2, axis=0))
    
    print("\nSummary:")
    print(f"  Regularization Alpha = {RIDGE_ALPHA}")
    print(f"  Used basis functions (K) = {K_final}")
    print(f"  RMSE (X): {rmse_pos[0]:.6f}")
    print(f"  RMSE (Y): {rmse_pos[1]:.6f}")

    print("\nFinal Parameters (MinJerk Regularized):")
    print(f"{'ID':<3} | {'Start':<8} | {'End':<8} | {'Dur':<8} | {'Sx':<8} | {'Sy':<8}")
    
    sort_idx = np.argsort(ts_final)
    for k in range(K_final):
        idx = sort_idx[k]
        dur = te_final[idx] - ts_final[idx]
        sx = scales_x[idx]
        sy = scales_y[idx]
        print(f"{k:<3} | {ts_final[idx]:.3f}    | {te_final[idx]:.3f}    | {dur:.3f}    | {sx:.3f}    | {sy:.3f}")