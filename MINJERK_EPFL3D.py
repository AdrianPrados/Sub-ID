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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- CONFIGURACIÓN ---
if not os.path.exists('Plots_EPFL_3D'):
    os.makedirs('Plots_EPFL_3D')

np.set_printoptions(precision=4, suppress=True)

# REGULARIZATION FACTOR (L2 / RIDGE)
RIDGE_ALPHA = 0.05

# -----------------------------------------------------------------------------
# 0. DATA IMPORT & FILTERING (EPFL STYLE)
# -----------------------------------------------------------------------------

def import_data(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 20.0 
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    states_raw = zarr_data["data/flat_spatula"][:]  
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
        t_zero = t_episode[0]
        t_episode = t_episode - t_zero
        t_episode = np.maximum.accumulate(t_episode)
        t_episode = t_episode + np.linspace(0, 1e-3, len(t_episode))
        t_episode = t_episode + t_zero
        # ------------------------------------------------

        t_start_ep = t_episode[0]
        t_end_ep = t_episode[-1]
        target_times = np.arange(t_start_ep + dt, t_end_ep - dt, dt)
        
        states = np.zeros((len(target_times), state_dim))

        # 1. POSITION INTERPOLATION (3D)
        dim_pos = min(3, state_dim)
        for i in range(dim_pos):
            f_interp = interp1d(t_episode, states_raw[dex_states, i], kind="cubic", assume_sorted=True)
            states[:, i] = f_interp(target_times)

        # 2. BUTTERWORTH FILTERING
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


# -----------------------------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS CORE (MINIMUM JERK)
# -----------------------------------------------------------------------------

def compute_minjerk_base(t, t0, t1):
    t = np.asarray(t)
    base = np.zeros_like(t)
    
    D = t1 - t0
    if D <= 1e-6: return base
    
    tau = (t - t0) / D
    mask = (tau >= 0) & (tau <= 1)
    if not np.any(mask): return base
    
    tau_val = tau[mask]
    val = (1.0 / D) * (30 * tau_val**2 - 60 * tau_val**3 + 30 * tau_val**4)
    base[mask] = val
    return base

def build_phi_matrix(t, t_starts, t_ends):
    K = t_starts.size
    N = t.size
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
    Lambda[-1, -1] = 0.0  # No regularizar intercepto
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    try:
        x = np.linalg.solve(AtA + Lambda, Atb)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x

# -----------------------------------------------------------------------------
# 2. ALGORITMOS DE AJUSTE (3D)
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
    
    # Fase A: Detección Inicial
    dist_param = max(1, N // 60) 
    peaks_indices, _ = find_peaks(vt, height=max_vt * 0.05, distance=dist_param)
    
    if len(peaks_indices) > 0:
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.7)
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
    
    # Fase B: Grid Search
    candidate_durations = np.arange(0.2, 2.5, 0.02)
    
    if rms > max_vt * residual_tol:
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

def compute_position_error_minjerk_3d(params, t, P_ref):
    N, D = P_ref.shape # D=3 for 3D
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
    
    total_error = 0
    for d in range(D):
        w = solve_ridge_weighted(A, P_ref[:, d], alpha=RIDGE_ALPHA)
        fit_d = A.dot(w)
        total_error += np.sum((fit_d - P_ref[:, d])**2)
    
    return total_error

def refine_bases_minjerk_3d(t, P_ref, t_starts_init, t_ends_init, verbose=False, step_title="Global Opt"):
    print(f"\n{step_title}: Optimizing {t_starts_init.size} bases (Ridge Alpha={RIDGE_ALPHA})...")
    K = t_starts_init.size
    params_init = np.concatenate([t_starts_init, t_ends_init])
    bounds = []
    
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))       
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))   
    
    t0 = time.time()
    res = minimize(compute_position_error_minjerk_3d, 
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

def merge_bases_minjerk_3d(t_starts, t_ends, scales, proximity_tol=0.05):
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
# 4. PLOTTING HELPERS (3D)
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
    plt.savefig(os.path.join("Plots_EPFL_3D", filename))
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
    plt.savefig(os.path.join("Plots_EPFL_3D", title.replace(" ", "_") + ".svg"))
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
    plt.savefig(os.path.join("Plots_EPFL_3D", title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_3d_trajectory_pos(P_ref, P_fit):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P_ref[:,0], P_ref[:,1], P_ref[:,2], 'k-', label='Original', lw=1)
    ax.plot(P_fit[:,0], P_fit[:,1], P_fit[:,2], 'r--', label='MinJerk Fit', lw=2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Trajectory: Real vs Fitted (MinJerk)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("Plots_EPFL_3D", "3D_Trajectory.svg"))
    plt.close()

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # ZARR PATH - Replace with your valid 3D Zarr dataset path
    ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/EPFL_data/adrian_adrian3D_2026-03-13-16-04/data.zarr"
    print(f"--- 1. Loading and Filtering 3D Data from real users... ---")
    
    if os.path.exists(ZARR_PATH):
        (states_raw, times_raw, episode_ends_raw, 
        states_processed_all, target_times_all, episode_ends_processed) = import_data(ZARR_PATH)
        
        EPISODE_IDX = 1
        start_idx = 0 if EPISODE_IDX == 0 else episode_ends_processed[EPISODE_IDX-1]
        end_idx = episode_ends_processed[EPISODE_IDX]
        
        # Extract processed (filtered and resampled) data for the specific episode
        episode_data = states_processed_all[start_idx:end_idx, :].astype(float)
        t_raw = target_times_all[start_idx:end_idx]
        t = t_raw - t_raw[0]
        N = t.size
        
        # 3D Position
        P = episode_data[:, :3]
            
        V_ref_3d = np.zeros_like(P)
        for i in range(3):
            V_ref_3d[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref_3d, axis=1)
        
        print(f"Loaded Episode {EPISODE_IDX}. Samples: {N}.")
    else:
        print("ERROR: Zarr file not found. Generating dummy 3D data for testing...")
        # Generamos datos dummy si no encuentra el Zarr
        t = np.linspace(0, 5, 500)
        N = len(t)
        P = np.column_stack([np.sin(t), np.cos(t), t * 0.5])
        V_ref_3d = np.zeros_like(P)
        for i in range(3): V_ref_3d[:, i] = np.gradient(P[:, i], t)
        vt_ref = np.linalg.norm(V_ref_3d, axis=1)
    
    init_time = time.time()

    # --- 1. Greedy Init ---
    print("\nSTEP 1: Getting initial guess (MinJerk Greedy)...")
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vt_ref], max_bases_total=1000, residual_tol=0.04, verbose=True)

    Phi_init = build_phi_matrix(t, ts_init, te_init)
    s_init, _ = nnls(Phi_init, vt_ref)
    vt_fit_init = Phi_init.dot(s_init)
    
    plot_velocity_and_bases(t, vt_ref, vt_fit_init, ts_init, te_init, s_init, 
                            dim_color='tab:cyan', title='STEP 1b: Initial Guess (Fitted to vt_ref)')
    
    # --- 2. Global Optimization (3D Ridge) ---
    ts_opt, te_opt = refine_bases_minjerk_3d(t, P, ts_init, te_init, verbose=False)
    
    # --- 3. Calcular Escalas Intermedias (3D RIDGE) ---
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    
    scales_list_temp = []
    for d in range(3):
        w = solve_ridge_weighted(A_temp, P[:, d], alpha=RIDGE_ALPHA)
        scales_list_temp.append(w[:-1])
    scales_temp = np.column_stack(scales_list_temp)
    
    print(f"\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk_3d(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
    
    # Refinamiento Final
    ts_final, te_final = refine_bases_minjerk_3d(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    K_final = ts_final.size
    
    # --- 4. Cálculo FINAL de escalas (3D RIDGE) ---
    Phi_vel_final = build_phi_matrix(t, ts_final, te_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final):
        Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)

    A_final = np.hstack([Phi_pos_final, np.ones((N, 1))])
    
    scales_list_final = []
    p0_list = []
    for d in range(3):
        w = solve_ridge_weighted(A_final, P[:, d], alpha=RIDGE_ALPHA)
        scales_list_final.append(w[:-1])
        p0_list.append(w[-1])
        
    scales_final_mat = np.column_stack(scales_list_final)
    
    # Reconstrucción 3D
    P_fit = np.zeros_like(P)
    for d in range(3):
        w_full = np.concatenate([scales_list_final[d], [p0_list[d]]])
        P_fit[:, d] = A_final.dot(w_full)
    
    V_fit = Phi_vel_final.dot(scales_final_mat)
    vt_fit = np.linalg.norm(V_fit, axis=1)
    scales_mag = np.linalg.norm(scales_final_mat, axis=1)
    
    end_time = time.time()
    
    # --- PLOTS ---
    print("\nGenerating Plots...")
    plot_position_and_integrated_bases(t, P[:, 0], P_fit[:, 0], ts_final, te_final, Phi_pos_final, scales_final_mat[:, 0], 
                                    pos_color='tab:red', neg_color='tab:cyan', title='Pos X (Regularized)')
    plot_position_and_integrated_bases(t, P[:, 1], P_fit[:, 1], ts_final, te_final, Phi_pos_final, scales_final_mat[:, 1],
                                    pos_color='tab:green', neg_color='tab:purple', title='Pos Y (Regularized)')
    plot_position_and_integrated_bases(t, P[:, 2], P_fit[:, 2], ts_final, te_final, Phi_pos_final, scales_final_mat[:, 2],
                                    pos_color='tab:orange', neg_color='tab:blue', title='Pos Z (Regularized)')

    plot_velocity_and_bases_signed(t, V_ref_3d[:, 0], V_fit[:, 0], ts_final, te_final, Phi_vel_final, scales_final_mat[:, 0], 
                                pos_color='tab:red', neg_color='tab:cyan', title='Vel X (Regularized)')
    plot_velocity_and_bases_signed(t, V_ref_3d[:, 1], V_fit[:, 1], ts_final, te_final, Phi_vel_final, scales_final_mat[:, 1], 
                                pos_color='tab:green', neg_color='tab:purple', title='Vel Y (Regularized)')
    plot_velocity_and_bases_signed(t, V_ref_3d[:, 2], V_fit[:, 2], ts_final, te_final, Phi_vel_final, scales_final_mat[:, 2], 
                                pos_color='tab:orange', neg_color='tab:blue', title='Vel Z (Regularized)')
                                
    plot_velocity_and_bases(t, vt_ref, vt_fit, ts_final, te_final, scales_mag, 
                            dim_color='tab:orange', title='Final Tangential Velocity (Regularized)')
                            
    plot_3d_trajectory_pos(P, P_fit)
    
    rmse_pos = np.sqrt(np.mean((P - P_fit)**2, axis=0))
    
    print("\nSummary:")
    print(f"  Regularization Alpha = {RIDGE_ALPHA}")
    print(f"  Used basis functions (K) = {K_final}")
    print(f"  RMSE (X): {rmse_pos[0]:.6f}")
    print(f"  RMSE (Y): {rmse_pos[1]:.6f}")
    print(f"  RMSE (Z): {rmse_pos[2]:.6f}")

    print("\nFinal Parameters (MinJerk 3D Regularized):")
    print(f"{'ID':<3} | {'Start':<8} | {'End':<8} | {'Dur':<8} | {'Sx':<8} | {'Sy':<8} | {'Sz':<8}")
    
    sort_idx = np.argsort(ts_final)
    for k in range(K_final):
        idx = sort_idx[k]
        dur = te_final[idx] - ts_final[idx]
        sx = scales_final_mat[idx, 0]
        sy = scales_final_mat[idx, 1]
        sz = scales_final_mat[idx, 2]
        print(f"{k:<3} | {ts_final[idx]:.3f}    | {te_final[idx]:.3f}    | {dur:.3f}    | {sx:.3f}    | {sy:.3f}    | {sz:.3f}")