import numpy as np
import zarr
import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

from scipy.optimize import nnls, minimize
from scipy.signal import find_peaks, peak_widths, butter, filtfilt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
MAX_ACTIVE_PRIMITIVES = 6
# Structure: [IS_ACTIVE, DeltaT, Dur, Sx, Sy, Sz]
PRIMITIVE_DIM = 6  
# Value for "dT" when no next action exists (Far Horizon)
PAD_DT_VALUE = 1.0 

# Hyperparameters for MinJerk and filtering
RIDGE_ALPHA = 0.01
TARGET_HZ = 100.0
LOWPASS_CUTOFF = 5.0
BUTTER_ORDER = 4

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
    Lambda[-1, -1] = 0.0 
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
    
    dist_param = max(1, N // 60) 
    peaks_indices, _ = find_peaks(vt, height=max_vt * 0.05, distance=dist_param)
    
    if len(peaks_indices) > 0:
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.6)
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
    
    candidate_durations = np.arange(0.2, 1.5, 0.01)
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
    
    total_error = 0
    for d in range(D):
        S_d = solve_ridge_weighted(A, P_ref[:, d], alpha=RIDGE_ALPHA)
        fit_d = A.dot(S_d)
        total_error += np.sum((fit_d - P_ref[:, d])**2)
        
    return total_error

def refine_bases_minjerk(t, P_ref, t_starts_init, t_ends_init, verbose=False):
    K = t_starts_init.size
    params_init = np.concatenate([t_starts_init, t_ends_init])
    bounds = []
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))       
    for _ in range(K): bounds.append((t[0]-0.5, t[-1]+0.5))   
    res = minimize(compute_position_error_minjerk_regularized, params_init, args=(t, P_ref),
            method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'disp': verbose})
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
                    group.append(j); visited[j] = True
        idx_g = np.array(group)
        mags = np.linalg.norm(scales[idx_g], axis=1)
        total_mag = np.sum(mags) + 1e-9
        w_ts = np.sum(t_starts[idx_g] * mags) / total_mag
        w_te = np.sum(t_ends[idx_g] * mags) / total_mag
        w_scales = np.sum(scales[idx_g], axis=0)
        new_ts.append(w_ts); new_te.append(w_te); new_scales.append(w_scales)
    return np.array(new_ts), np.array(new_te), np.array(new_scales)


# -----------------------------------------------------------------------------
# MAIN DATASET GENERATION
# -----------------------------------------------------------------------------
def create_object_moving_dataset(input_csv_path, output_zarr_path):
    print(f"--- Generating Dataset: State Only 3D + MINJERK + MASKING ---")
    print(f"Input CSV: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # We group by Participant and Trial to process each episode independently
    grouped = df.groupby(['participant', 'trial_n'])
    
    data_vec_obs = [] 
    data_actions = [] 
    episode_ends = []
    total_steps = 0
    
    # Butterworth Filter Design
    nyq = 0.5 * TARGET_HZ
    normal_cutoff = LOWPASS_CUTOFF / nyq
    b, a = butter(BUTTER_ORDER, normal_cutoff, btype='low', analog=False)

    for (part_id, trial_id), df_subset in tqdm(grouped, desc="Processing Trials"):
        
        # --- A. PREPARE EPISODE DATA ---
        df_subset = df_subset.sort_values('time')
        
        t_raw = df_subset['time'].values
        xs_raw = df_subset['x'].values
        ys_raw = df_subset['y'].values
        zs_raw = df_subset['z'].values
        
        # 1. Filtro de duplicados
        _, unique_idx = np.unique(t_raw, return_index=True)
        t_clean = t_raw[unique_idx]
        xs_clean = xs_raw[unique_idx]
        ys_clean = ys_raw[unique_idx]
        zs_clean = zs_raw[unique_idx]

        t_clean = t_clean - t_clean[0]
        
        # 2. Interpolación a Target HZ
        dt = 1.0 / TARGET_HZ
        target_times = np.arange(0.0, t_clean[-1], dt)
        
        f_x = interp1d(t_clean, xs_clean, kind="cubic", fill_value="extrapolate")
        f_y = interp1d(t_clean, ys_clean, kind="cubic", fill_value="extrapolate")
        f_z = interp1d(t_clean, zs_clean, kind="cubic", fill_value="extrapolate")
        
        xs_interp = f_x(target_times)
        ys_interp = f_y(target_times)
        zs_interp = f_z(target_times)

        # 3. Filtrado Butterworth
        pad_len = min(len(target_times) - 1, 3 * max(len(a), len(b))) 
        xs_smooth = filtfilt(b, a, xs_interp, padlen=pad_len)
        ys_smooth = filtfilt(b, a, ys_interp, padlen=pad_len)
        zs_smooth = filtfilt(b, a, zs_interp, padlen=pad_len)

        t = target_times
        N = t.size
        
        # Physical State Shape: (N, 3)
        ep_state = np.column_stack([xs_smooth, ys_smooth, zs_smooth]).astype(np.float32)

        # --- B. MINJERK EXTRACTION ---
        vx_ref = np.gradient(xs_smooth, dt)
        vy_ref = np.gradient(ys_smooth, dt)
        vz_ref = np.gradient(zs_smooth, dt)
        P_ref = ep_state

        ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vx_ref, vy_ref, vz_ref], 
                                                        max_bases_total=100, 
                                                        residual_tol=0.03, verbose=False)
        primitives = []
        if len(ts_init) > 0:
            ts_opt, te_opt = refine_bases_minjerk(t, P_ref, ts_init, te_init, verbose=False)
            
            Phi_f = build_phi_matrix(t, ts_opt, te_opt)
            Phi_pos_f = np.zeros_like(Phi_f)
            for k in range(Phi_pos_f.shape[1]): 
                Phi_pos_f[:, k] = cumtrapz(Phi_f[:, k], t, initial=0.0)
            
            A_f = np.hstack([Phi_pos_f, np.ones((N, 1))])
            Sx_f = solve_ridge_weighted(A_f, P_ref[:, 0], RIDGE_ALPHA)
            Sy_f = solve_ridge_weighted(A_f, P_ref[:, 1], RIDGE_ALPHA)
            Sz_f = solve_ridge_weighted(A_f, P_ref[:, 2], RIDGE_ALPHA)
            
            # Merge logic for 3D
            scales_temp = np.column_stack([Sx_f[:-1], Sy_f[:-1], Sz_f[:-1]])
            ts_final, te_final, scales_final = merge_bases_minjerk(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
            ts_final, te_final = refine_bases_minjerk(t, P_ref, ts_final, te_final, verbose=False)
            
            # Re-solve for final merged bases
            Phi_f_m = build_phi_matrix(t, ts_final, te_final)
            Phi_pos_f_m = np.zeros_like(Phi_f_m)
            for k in range(Phi_pos_f_m.shape[1]): 
                Phi_pos_f_m[:, k] = cumtrapz(Phi_f_m[:, k], t, initial=0.0)
                
            A_f_m = np.hstack([Phi_pos_f_m, np.ones((N, 1))])
            Sx_f_m = solve_ridge_weighted(A_f_m, P_ref[:, 0], RIDGE_ALPHA)
            Sy_f_m = solve_ridge_weighted(A_f_m, P_ref[:, 1], RIDGE_ALPHA)
            Sz_f_m = solve_ridge_weighted(A_f_m, P_ref[:, 2], RIDGE_ALPHA)
            
            for k in range(len(ts_final)):
                primitives.append({
                    'ts': ts_final[k],
                    'te': te_final[k],
                    'dur': te_final[k] - ts_final[k],
                    'sx': Sx_f_m[k], 
                    'sy': Sy_f_m[k],
                    'sz': Sz_f_m[k]
                })
            primitives.sort(key=lambda x: x['ts'])

        # --- C. VECTOR CONSTRUCTION ---
        # Memory Dim = 6 primitives * 6 params = 36 dimensions
        mem_dim = MAX_ACTIVE_PRIMITIVES * PRIMITIVE_DIM 
        
        # State Dim: Physical(3) + Memory(36) = 39 dimensions
        state_dim = ep_state.shape[1] + mem_dim             
        
        ep_vec_out = np.zeros((N, state_dim), dtype=np.float32)
        
        # Action Dim: 5 (dt_next, dur, sx, sy, sz)
        ep_act_out = np.zeros((N, 5), dtype=np.float32)

        for i_t in range(N):
            current_t = t[i_t]
            
            # --- 1. MEMORY INPUT (OBSERVATION) ---
            active_now = [p for p in primitives if p['ts'] <= current_t < p['te']]
            active_now.sort(key=lambda x: x['ts'])
            
            mem_vector = np.zeros(mem_dim, dtype=np.float32)
            
            for k in range(min(len(active_now), MAX_ACTIVE_PRIMITIVES)):
                p = active_now[k]
                dt_active = p['ts'] - current_t 
                base_idx = k * PRIMITIVE_DIM 
                
                mem_vector[base_idx]     = 1.0       # IS_ACTIVE
                mem_vector[base_idx + 1] = dt_active # Relative Start
                mem_vector[base_idx + 2] = p['dur']  # Duration
                mem_vector[base_idx + 3] = p['sx']   # Scale X
                mem_vector[base_idx + 4] = p['sy']   # Scale Y
                mem_vector[base_idx + 5] = p['sz']   # Scale Z
            
            # CONCATENATE: [Phys(3) + Mem(36)]
            ep_vec_out[i_t] = np.concatenate([
                ep_state[i_t], 
                mem_vector
            ])
            
            # --- 2. TARGET OUTPUT (ACTION) ---
            next_prim = None
            for p in primitives:
                if p['ts'] > current_t:
                    next_prim = p
                    break
            
            if next_prim is not None:
                dt_next = next_prim['ts'] - current_t
                ep_act_out[i_t] = [
                    dt_next,
                    next_prim['dur'],
                    next_prim['sx'],
                    next_prim['sy'],
                    next_prim['sz']
                ]
            else:
                ep_act_out[i_t] = [
                    PAD_DT_VALUE, 0.0, 0.0, 0.0, 0.0
                ]

        data_vec_obs.append(ep_vec_out)
        data_actions.append(ep_act_out)
        
        total_steps += N
        episode_ends.append(total_steps)

    # --- D. SAVE ---
    print(f"\nSaving dataset to: {output_zarr_path}")
    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path)
        
    root = zarr.open(output_zarr_path, 'w')
    
    # Concatenate
    all_vec = np.concatenate(data_vec_obs, axis=0)
    all_act = np.concatenate(data_actions, axis=0)
    all_ends = np.array(episode_ends, dtype=np.int64)
    
    # Create Groups
    g_data = root.create_group('data')
    g_meta = root.create_group('meta')
    
    g_data.create_dataset('state', data=all_vec, chunks=(1000, all_vec.shape[1]))
    g_data.create_dataset('action', data=all_act, chunks=(1000, all_act.shape[1]))
    g_meta.create_dataset('episode_ends', data=all_ends)
    
    print("\nDataset Generated Successfully!")
    print(f"  Total steps: {total_steps} across {len(episode_ends)} episodes.")
    print(f"  Input Vector: {all_vec.shape} (data/state)")
    print(f"     Structure: [Phys(3) + Memory(36)]")
    print(f"     Memory Structure: 6 slots x [IS_ACTIVE, dt, dur, Sx, Sy, Sz]")
    print(f"  Output Action: {all_act.shape} (data/action)")
    print(f"  NOTE: Update 'shape_meta' in your config. State dim is now 39, Action is 5.")

if __name__ == "__main__":
    
    IN_PATH ="/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/COMPARATIVAS/sssumoMethod/data/object_moving_tangential_velocity_data.csv"
    OUT_PATH = "object_moving_minjerk_masked_3D.zarr"
    
    create_object_moving_dataset(IN_PATH, OUT_PATH)