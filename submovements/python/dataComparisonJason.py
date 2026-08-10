import numpy as np
import time
import os
import zarr  # <--- Necesario para PushT
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.signal import find_peaks, peak_widths, savgol_filter # <--- Agregado savgol_filter

# Importamos el módulo externo
import movement_decompose_2d

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'Results_Decompose2D_Comparison'
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
# 5. MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # --- SELECCIÓN DE DATOS ---
    # 'PUSHT' o 'SUBJECT'
    DATA_SOURCE = 'PUSHT' 

    # Rutas (Ajusta según tu sistema)
    ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/pusht_real/real_pusht_20230105/replay_buffer.zarr"
    SUBJECT_PATH = r'../data/subject08day1pre'
    
    print(f"--- LOADING DATA (SOURCE: {DATA_SOURCE}) ---")
    
    # Variables a rellenar independientemente de la fuente
    t = None
    P = None
    xs_smooth, ys_smooth = None, None
    vx_ref, vy_ref, vt_ref = None, None, None
    N = 0

    if DATA_SOURCE == 'PUSHT':
        # Lógica importada de MINJERK_pushT.py
        if os.path.exists(ZARR_PATH):
            zarr_data = zarr.open(ZARR_PATH, mode='r')
            episode_ends = zarr_data['meta/episode_ends'][:]
            states_all = zarr_data['data/robot_eef_pose'][:]
            time_stamps = zarr_data['data/timestamp'][:]
            
            # Selecciona el episodio (ej. 3 como en tu ejemplo)
            EPISODE_IDX = 0
            start_idx = 0 if EPISODE_IDX == 0 else episode_ends[EPISODE_IDX-1]
            end_idx = episode_ends[EPISODE_IDX]
            
            episode_data = states_all[start_idx:end_idx, :].astype(float)
            t_raw = time_stamps[start_idx:end_idx]
            t = t_raw - t_raw[0]
            N = t.size
            
            xs_raw = episode_data[:, 0]
            ys_raw = episode_data[:, 1]
            
            # Filtro Savitzky-Golay (Importante para datos de robot)
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
        # Lógica original
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
    # PARTE 1: EJECUTAR TU ALGORITMO (Para obtener K_final)
    # =========================================================================
    print("\n" + "="*60)
    print(" >>> EJECUTANDO TU ALGORITMO (CALCULO DE K) <<<")
    print("="*60)

    # 1. Greedy Init
    print("STEP 1: Initial Guess (MinJerk Greedy)...")
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vx_ref, vy_ref], max_bases_total=1000, residual_tol=0.03, verbose=True)

    # 2. Global Optimization (Ridge)
    ts_opt, te_opt = refine_bases_minjerk(t, P, ts_init, te_init, verbose=False)
    
    # 3. Escalas Intermedias
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    Sx_temp = solve_ridge_weighted(A_temp, P[:, 0], alpha=RIDGE_ALPHA)
    Sy_temp = solve_ridge_weighted(A_temp, P[:, 1], alpha=RIDGE_ALPHA)
    scales_temp = np.column_stack([Sx_temp[:-1], Sy_temp[:-1]])
    
    # 4. Merging
    print(f"\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk(ts_opt, te_opt, scales_temp, proximity_tol=0.04)
    ts_final, te_final = refine_bases_minjerk(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    
    K_final = ts_final.size
    print(f"\n[OUR ALGORITHM] Calculated K = {K_final}. Passing to Decompose_2D...")


    # =========================================================================
    # PARTE 2: EJECUTAR DECOMPOSE_2D Y GUARDAR DATOS
    # =========================================================================
    print("\n" + "="*60)
    print(f" >>> EJECUTANDO DECOMPOSE_2D (K={K_final}) <<<")
    print("="*60)
    
    t0_dec = time.time()
    
    # Ajustar rangos según el dataset si es necesario
    # Para PushT los valores suelen ser pixeles (0-512) o normalizados, pero decompose_2D necesita rangos de búsqueda.
    # Usaremos rangos amplios por seguridad.
    if DATA_SOURCE == 'PUSHT':
         x_range = (np.min(xs_smooth)-20, np.max(xs_smooth)+20)
         y_range = (np.min(ys_smooth)-20, np.max(ys_smooth)+20)
    else:
         x_range = (-20, 20) 
         y_range = (-10, 10)
    
    V_input = np.column_stack([vx_ref, vy_ref]) # Reconstruir V para la función
    
    best_error, final_params = movement_decompose_2d.decompose_2D(
        t, V_input, n_sub_movement=K_final, x_rng=x_range, y_rng=y_range)
    
    exec_time_dec = time.time()-t0_dec
    print(f"Decompose_2D finished in {exec_time_dec:.2f}s. Best Error: {best_error:.4e}")
    
    # --- RECONSTRUCCIÓN DE SEÑALES DE DECOMPOSE_2D PARA GUARDAR ---
    
    K_dec = final_params.shape[0]
    ts_dec = final_params[:, 0]
    D_dec  = final_params[:, 1]
    te_dec = ts_dec + D_dec
    scales_x_dec = final_params[:, 2] # Ax
    scales_y_dec = final_params[:, 3] # Ay
    
    # Construcción de señales (usando las mismas funciones matemáticas)
    Phi_vel_dec = build_phi_matrix(t, ts_dec, te_dec)
    Phi_pos_dec = np.zeros_like(Phi_vel_dec)
    for k in range(K_dec):
        Phi_pos_dec[:, k] = cumtrapz(Phi_vel_dec[:, k], t, initial=0.0)
    
    # Reconstrucción (Nota: decompose_2D devuelve desplazamientos relativos, sumamos pos inicial P[0])
    vx_fit_dec = Phi_vel_dec.dot(scales_x_dec)
    vy_fit_dec = Phi_vel_dec.dot(scales_y_dec)
    vt_fit_dec = np.sqrt(vx_fit_dec**2 + vy_fit_dec**2)
    
    fit_x_dec = Phi_pos_dec.dot(scales_x_dec) + P[0, 0]
    fit_y_dec = Phi_pos_dec.dot(scales_y_dec) + P[0, 1]

    # --- GUARDAR DATOS (.npz) ---
    save_path = os.path.join(OUTPUT_DIR, "decomposition_data_Jason.npz")
    print(f"\nSaving DECOMPOSE_2D data to: {save_path}")
    
    # Cálculo rápido de métricas para el guardado
    def calc_r2_quick(y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    r2_vt = calc_r2_quick(vt_ref, vt_fit_dec)
    rmse_vt = np.sqrt(np.mean((vt_ref - vt_fit_dec)**2))

    np.savez(
        save_path,
        t=t,
        # Referencias
        vx_ref=vx_ref, vy_ref=vy_ref, vt_ref=vt_ref,
        px_ref=xs_smooth, py_ref=ys_smooth,
        # Resultados Decompose 2D
        vx_fit=vx_fit_dec, vy_fit=vy_fit_dec, vt_fit=vt_fit_dec,
        px_fit=fit_x_dec, py_fit=fit_y_dec,
        # Parámetros
        subm_starts=ts_dec,
        subm_ends=te_dec,
        subm_scales_x=scales_x_dec,
        subm_scales_y=scales_y_dec,
        # Metadatos
        metrics=np.array([r2_vt, rmse_vt, K_dec, exec_time_dec])
    )
    print("Data saved successfully (Plots omitted).")