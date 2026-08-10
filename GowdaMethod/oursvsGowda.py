import numpy as np
import matplotlib.pyplot as plt
import time
import os
import zarr  # <--- Necesario para PushT
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.signal import find_peaks, peak_widths, savgol_filter 

# Importamos los módulos externos
import movement_decompose_2d
import gowda_algorithm  # <--- El código traducido de Gowda

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'Plots_Comparison_PushT'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

np.set_printoptions(precision=4, suppress=True)

# REGULARIZATION FACTOR (L2 / RIDGE) para tu algoritmo
RIDGE_ALPHA = 0.05

# -----------------------------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS CORE (TU ALGORITMO)
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
    # MinJerk Polynomial (Velocity profile scaled by 1/D)
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
# 2. ALGORITMOS DE AJUSTE (TU ALGORITMO)
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
    
    if len(peaks_indices) > 0:
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.7)
        for i, idx in enumerate(peaks_indices):
            t_left = left_ips[i] * dt + t[0]
            t_right = right_ips[i] * dt + t[0]
            center_time = (t_left + t_right) / 2.0
            width_fwhm = t_right - t_left
            duration = np.clip(width_fwhm * 1.8, 0.3, 3.5)
            starts_found.append(center_time - (duration / 2.0))
            ends_found.append(center_time + (duration / 2.0))

    if len(starts_found) > 0:
        Phi = build_phi_matrix(t, np.array(starts_found), np.array(ends_found))
        s, _ = nnls(Phi, vt)
        residual = vt - Phi.dot(s)
    else:
        residual = vt.copy()

    rms = np.sqrt(np.mean(residual**2))
    
    # --- FASE B: Grid Search ---
    candidate_durations = np.arange(0.2, 1.5, 0.01)
    
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
            options={'maxiter': 2000, 'disp': verbose})
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
# 3. PLOTTING HELPERS (INCORPORADOS DEL OTRO CÓDIGO)
# -----------------------------------------------------------------------------

def plot_velocity_and_bases(t, v_ref, v_fit, t_starts, t_ends, scales_dim, 
                            dim_color='tab:red', title='Velocity dim', folder="Plots_Comparison"):
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
    plt.xlabel('Time (s)'); plt.ylabel('Velocity'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    filename = title.replace("\n", " ").replace(":", "").replace(" ", "_") + ".svg"
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_velocity_and_bases_signed(t, v_ref, v_fit, t_starts, t_ends, Phi_vel, scales_dim, 
                                pos_color='tab:red', neg_color='tab:blue', title='Velocity Decomposition', folder="Plots_Comparison"):
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
    plt.savefig(os.path.join(folder, title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_position_and_integrated_bases(t, p_ref, p_fit, t_starts, t_ends, Phi_pos, scales_dim, 
                                    pos_color='tab:red', neg_color='tab:blue', title='Position dim', folder="Plots_Comparison"):
    N = t.size
    scaled_pos = Phi_pos * scales_dim[None, :] if Phi_pos.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    plt.plot(t, p_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, p_fit, '--', lw=2, color='green', label='Reconstructed')
    
    for k in range(scaled_pos.shape[1]):
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]>=0, facecolor=pos_color, alpha=0.2)
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]<0, facecolor=neg_color, alpha=0.2)

    plt.xlabel('Time (s)'); plt.ylabel('Position'); plt.title(title)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(folder, title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_2d_trajectory_pos(xs, ys, x_fit, y_fit, title='2D Trajectory', folder="Plots_Comparison"):
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'k-', lw=2, label='Original')
    plt.plot(x_fit, y_fit, 'r--', lw=2, label='Fitted')
    plt.legend()
    plt.title(title)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(folder, title.replace(" ", "_") + ".svg"))
    plt.close()

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # --- SELECCIÓN DE DATOS ---
    DATA_SOURCE = 'PUSHT' 

    # Rutas (Ajusta según tu sistema)
    ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/pusht_real/real_pusht_20230105/replay_buffer.zarr"
    SUBJECT_PATH = r'../data/subject08day1pre'
    
    print(f"--- LOADING DATA (SOURCE: {DATA_SOURCE}) ---")
    
    t = None
    P = None
    xs_smooth, ys_smooth = None, None
    vx_ref, vy_ref, vt_ref = None, None, None
    N = 0

    if DATA_SOURCE == 'PUSHT':
        if os.path.exists(ZARR_PATH):
            zarr_data = zarr.open(ZARR_PATH, mode='r')
            episode_ends = zarr_data['meta/episode_ends'][:]
            states_all = zarr_data['data/robot_eef_pose'][:]
            time_stamps = zarr_data['data/timestamp'][:]
            
            EPISODE_IDX = 0
            start_idx = 0 if EPISODE_IDX == 0 else episode_ends[EPISODE_IDX-1]
            end_idx = episode_ends[EPISODE_IDX]
            
            episode_data = states_all[start_idx:end_idx, :].astype(float)
            t_raw = time_stamps[start_idx:end_idx]
            t = t_raw - t_raw[0]
            N = t.size
            
            xs_raw = episode_data[:, 0]
            ys_raw = episode_data[:, 1]
            
            xs_smooth = savgol_filter(xs_raw, 31, 3) if N > 31 else xs_raw
            ys_smooth = savgol_filter(ys_raw, 31, 3) if N > 31 else ys_raw
                
            vx_ref = np.gradient(xs_smooth, t)
            vy_ref = np.gradient(ys_smooth, t)
            vt_ref = np.sqrt(vx_ref**2 + vy_ref**2)
            
            P = np.column_stack([xs_smooth, ys_smooth])
            print(f"Loaded PushT Episode {EPISODE_IDX}. Samples: {N}.")
        else:
            exit(f"Zarr file not found: {ZARR_PATH}")

    elif DATA_SOURCE == 'SUBJECT':
        if os.path.exists(SUBJECT_PATH):
            try:
                position_filtered, velocity_list, time_list = movement_decompose_2d.load_data(SUBJECT_PATH)
                if not position_filtered: exit("No data loaded.")
                EPISODE_IDX = 0
                P = position_filtered[EPISODE_IDX]
                V = velocity_list[EPISODE_IDX]
                t_raw = time_list[EPISODE_IDX]
                t = t_raw - t_raw[0]
                N = t.size
                xs_smooth, ys_smooth = P[:, 0], P[:, 1]
                vx_ref, vy_ref = V[:, 0], V[:, 1]
                vt_ref = np.sqrt(vx_ref**2 + vy_ref**2)
                print(f"Loaded Subject Episode {EPISODE_IDX}. Samples: {N}.")
            except Exception as e: exit(f"Error loading data: {e}")
        else:
            exit(f"Data path not found: {SUBJECT_PATH}")
    else:
        exit("Invalid DATA_SOURCE selected.")

    
    # =========================================================================
    # PARTE 1: EJECUTAR TU ALGORITMO (OUR ALGORITHM)
    # =========================================================================
    print("\n" + "="*60)
    print(" >>> EJECUTANDO TU ALGORITMO (CALCULO DE K Y PLOTS) <<<")
    print("="*60)

    print("STEP 1: Initial Guess (MinJerk Greedy)...")
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vx_ref, vy_ref], max_bases_total=1000, residual_tol=0.03, verbose=True)

    ts_opt, te_opt = refine_bases_minjerk(t, P, ts_init, te_init, verbose=False)
    
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    Sx_temp = solve_ridge_weighted(A_temp, P[:, 0], alpha=RIDGE_ALPHA)
    Sy_temp = solve_ridge_weighted(A_temp, P[:, 1], alpha=RIDGE_ALPHA)
    scales_temp = np.column_stack([Sx_temp[:-1], Sy_temp[:-1]])
    
    print(f"\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
    ts_final, te_final = refine_bases_minjerk(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    
    K_final = ts_final.size
    print(f"\n[OUR ALGORITHM] Calculated K = {K_final}.")

    Phi_vel_final = build_phi_matrix(t, ts_final, te_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final): Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)
    A_final = np.hstack([Phi_pos_final, np.ones((N, 1))])
    Sx_final = solve_ridge_weighted(A_final, P[:, 0], alpha=RIDGE_ALPHA)
    Sy_final = solve_ridge_weighted(A_final, P[:, 1], alpha=RIDGE_ALPHA)
    
    scales_x = Sx_final[:-1]
    scales_y = Sy_final[:-1]
    scales_final_mat = np.column_stack([scales_x, scales_y])
    fit_x = A_final.dot(Sx_final)
    fit_y = A_final.dot(Sy_final)
    V_fit = Phi_vel_final.dot(scales_final_mat)
    vx_fit, vy_fit = V_fit[:, 0], V_fit[:, 1]
    vt_fit = np.sqrt(vx_fit**2 + vy_fit**2)
    scales_mag = np.linalg.norm(scales_final_mat, axis=1)

    print("Generating OUR Plots...")
    plot_position_and_integrated_bases(t, xs_smooth, fit_x, ts_final, te_final, Phi_pos_final, scales_x, 
                                    title='Our_Pos_X', folder=OUTPUT_DIR)
    plot_position_and_integrated_bases(t, ys_smooth, fit_y, ts_final, te_final, Phi_pos_final, scales_y,
                                    title='Our_Pos_Y', folder=OUTPUT_DIR)
    plot_velocity_and_bases_signed(t, vx_ref, vx_fit, ts_final, te_final, Phi_vel_final, scales_x, 
                                title='Our_Vel_X', folder=OUTPUT_DIR)
    plot_velocity_and_bases_signed(t, vy_ref, vy_fit, ts_final, te_final, Phi_vel_final, scales_y, 
                                title='Our_Vel_Y', folder=OUTPUT_DIR)
    plot_velocity_and_bases(t, vt_ref, vt_fit, ts_final, te_final, scales_mag, 
                            title='Our_Tangential_Vel', folder=OUTPUT_DIR)
    plot_2d_trajectory_pos(xs_smooth, ys_smooth, fit_x, fit_y, title='Our_2D_Trajectory', folder=OUTPUT_DIR)



    # =========================================================================
    # PARTE 3: EJECUTAR ALGORITMO GOWDA (2015)
    # =========================================================================
    print("\n" + "="*60)
    print(" >>> EJECUTANDO GOWDA 2015 (SCATTERSHOT MINJERK) <<<")
    print("="*60)
    
    t0_gow = time.time()
    
    # 1. Preparar la matriz de entrada Nx4: [pos_x, pos_y, vel_x, vel_y]
    # Gowda asume que las columnas de velocidad están en vel_inds. En este caso [2, 3].
    hand_kin_gowda = np.column_stack([xs_smooth, ys_smooth, vx_ref, vy_ref])
    
    # Calculamos DT a partir del vector de tiempo
    dt_ms = (t[1] - t[0]) * 1000.0 if len(t) > 1 else 10.0
    
    try:
        # Llamamos al wrapper principal.
        movements_gow, submovement_recon_gow, segments_gow, costs_gow, fits_gow, fit_costs_gow, runtime_gow, n_iter_gow, n_evals_gow = \
            gowda_algorithm.decompose_submovements_v2(
                hand_kin_gowda, 
                vel_inds=[2, 3],          # Índices de velocidad en Python
                method='greedy',     
                fn_type='min_jerk', 
                bin_size_ms=dt_ms, 
                verbose=False
            )
        
        # 2. Extraer parámetros de Gowda
        # Gowda devuelve un diccionario de segmentos, donde cada segmento tiene un array de parámetros
        # Estructura típica del parámetro en Gowda 2D: [t_start, duracion, amp_x, amp_y]
        ts_gow_list, dur_gow_list, sx_gow_list, sy_gow_list = [], [], [], []
        
        for k_seg, mov_array in movements_gow.items():
            if mov_array is not None and len(mov_array) > 0:
                for param_row in mov_array:
                    ts_gow_list.append(param_row[0])
                    dur_gow_list.append(param_row[1])
                    sx_gow_list.append(param_row[2])
                    sy_gow_list.append(param_row[3])
                    
        ts_gow = np.array(ts_gow_list)
        dur_gow = np.array(dur_gow_list)
        te_gow = ts_gow + dur_gow
        scales_x_gow = np.array(sx_gow_list)
        scales_y_gow = np.array(sy_gow_list)
        K_gow = ts_gow.size
        
        exec_time_gow = time.time() - t0_gow
        print(f"Gowda finished in {exec_time_gow:.2f}s. Extracted {K_gow} submovements.")
        
        # 3. Construir matrices Phi y reconstruir trayectorias
        Phi_vel_gow = build_phi_matrix(t, ts_gow, te_gow)
        Phi_pos_gow = np.zeros_like(Phi_vel_gow)
        for k in range(K_gow):
            Phi_pos_gow[:, k] = cumtrapz(Phi_vel_gow[:, k], t, initial=0.0)
            
        vx_fit_gow = Phi_vel_gow.dot(scales_x_gow)
        vy_fit_gow = Phi_vel_gow.dot(scales_y_gow)
        vt_fit_gow = np.sqrt(vx_fit_gow**2 + vy_fit_gow**2)
        
        # Sumamos P[0] asumiendo desplazamientos relativos, como en el método de Jason
        fit_x_gow = Phi_pos_gow.dot(scales_x_gow) + P[0, 0]
        fit_y_gow = Phi_pos_gow.dot(scales_y_gow) + P[0, 1]
        scales_mag_gow = np.sqrt(scales_x_gow**2 + scales_y_gow**2)

        # 4. Generar Plots estilo OUR TEST
        print("Generating GOWDA Plots...")
        plot_position_and_integrated_bases(t, xs_smooth, fit_x_gow, ts_gow, te_gow, Phi_pos_gow, scales_x_gow, 
                                        title='Gowda_Pos_X', folder=OUTPUT_DIR)
        plot_position_and_integrated_bases(t, ys_smooth, fit_y_gow, ts_gow, te_gow, Phi_pos_gow, scales_y_gow,
                                        title='Gowda_Pos_Y', folder=OUTPUT_DIR)
        plot_velocity_and_bases_signed(t, vx_ref, vx_fit_gow, ts_gow, te_gow, Phi_vel_gow, scales_x_gow, 
                                    title='Gowda_Vel_X', folder=OUTPUT_DIR)
        plot_velocity_and_bases_signed(t, vy_ref, vy_fit_gow, ts_gow, te_gow, Phi_vel_gow, scales_y_gow, 
                                    title='Gowda_Vel_Y', folder=OUTPUT_DIR)
        plot_velocity_and_bases(t, vt_ref, vt_fit_gow, ts_gow, te_gow, scales_mag_gow, 
                                title='Gowda_Tangential_Vel', folder=OUTPUT_DIR)
        plot_2d_trajectory_pos(xs_smooth, ys_smooth, fit_x_gow, fit_y_gow, title='Gowda_2D_Trajectory', folder=OUTPUT_DIR)

    except Exception as e:
        print(f"[!] Error ejecutando Gowda (¿Faltan funciones internas?): {e}")
        K_gow, ts_gow, te_gow, dur_gow, scales_x_gow, scales_y_gow = 0, [], [], [], [], []


    # --- COMPARACIÓN EN CONSOLA UNIFICADA ---
    print("\n" + "-"*110)
    print(f"{'':<4} | {'OUR ALGORITHM':<32} | {'DECOMPOSE_2D (Jason)':<32} | {'GOWDA (2015)':<32}")
    print(f"{'ID':<4} | {'Start':<5} {'Dur':<5} {'Sx':<7} {'Sy':<7} | {'Start':<5} {'Dur':<5} {'Sx':<7} {'Sy':<7} | {'Start':<5} {'Dur':<5} {'Sx':<7} {'Sy':<7}")
    print("-" * 110)
    
    idx_our = np.argsort(ts_final)
    #idx_dec = np.argsort(ts_dec)
    idx_gow = np.argsort(ts_gow) if K_gow > 0 else []
    
    max_k = max(K_final, K_gow)
    
    for k in range(max_k):
        s_our, s_dec, s_gow = "", "", ""
        
        if k < K_final:
            i = idx_our[k]
            s_our = f"{ts_final[i]:.3f} {te_final[i]-ts_final[i]:.3f} {scales_x[i]:>7.3f} {scales_y[i]:>7.3f}"
            
        if k < K_gow:
            g = idx_gow[k]
            s_gow = f"{ts_gow[g]:.3f} {dur_gow[g]:.3f} {scales_x_gow[g]:>7.3f} {scales_y_gow[g]:>7.3f}"
            
        print(f"{k:<4} | {s_our:<32} | {s_dec:<32} | {s_gow:<32}")
        
    print("-" * 110)
    print(f"Plots saved in '{OUTPUT_DIR}' folder.")