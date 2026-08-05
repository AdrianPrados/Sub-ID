import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
from pathlib import Path
from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.signal import savgol_filter, find_peaks, peak_widths, butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# --- CONFIGURACIÓN ---
if not os.path.exists('Plots_AdrianData_MinJerk_3D'):
    os.makedirs('Plots_AdrianData_MinJerk_3D')

np.set_printoptions(precision=4, suppress=True)

# REGULARIZATION FACTOR (L2 / RIDGE)
RIDGE_ALPHA = 0.1  

# -----------------------------------------------------------------------------
# 1. FUNCIONES DE CARGA DE DATOS 
# -----------------------------------------------------------------------------

def import_data(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 1.0 
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    states_raw = zarr_data["data/state"][:]  
    # IMPORTANTE: Forzamos float64 desde el principio para toda la matriz de tiempo
    times_raw = np.asarray(zarr_data["data/time"][:], dtype=np.float64).ravel()
    
    dt = 1.0 / float(target_hz)

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

        # --- SOLUCIÓN VECTORIZADA: MONOTONÍA ABSOLUTA ---
        t_episode = times_raw[dex_states].copy()
        
        # 1. Normalizar a cero temporalmente. Esto nos salva si los tiempos son Unix Epoch.
        t_zero = t_episode[0]
        t_episode = t_episode - t_zero
        
        # 2. Impedir que el tiempo retroceda (corrige el ruido del sensor y el desorden).
        # Esto convierte algo como [0.1, 0.05, 0.2] en [0.1, 0.1, 0.2]
        t_episode = np.maximum.accumulate(t_episode)
        
        # 3. Forzar el crecimiento estricto añadiendo una pendiente microscópica total de 1 milisegundo.
        # Esto convierte los duplicados [0.1, 0.1, 0.2] en [0.100, 0.105, 0.201]
        # Garantiza cero duplicados de forma rapidísima.
        t_episode = t_episode + np.linspace(0, 1e-3, len(t_episode))
        
        # 4. Restaurar el tiempo base original.
        t_episode = t_episode + t_zero
        # ------------------------------------------------

        # 2. Crear target_times
        t_start_ep = t_episode[0]
        t_end_ep = t_episode[-1]
        target_times = np.arange(t_start_ep + dt, t_end_ep - dt, dt)
        
        states = np.zeros((len(target_times), 7))

        # 1. INTERPOLACIÓN POSICIÓN
        for i in range(3):
            # interp1d ya no se quejará de duplicados ni desórdenes
            f_interp = interp1d(t_episode, states_raw[dex_states, i], kind="cubic", assume_sorted=True)
            states[:, i] = f_interp(target_times)

        # 2. INTERPOLACIÓN ROTACIÓN
        rot_in = Rotation.from_quat(states_raw[dex_states, 3:7])
        slerp = Slerp(t_episode, rot_in)
        states[:, 3:7] = slerp(target_times).as_quat()

        # 3. FILTRADO
        b, a = butter(butter_order, lowpass_cutoff / (target_hz / 2), btype="low", analog=False)
        states[:, 0] = filtfilt(b, a, states[:, 0], axis=0)
        states[:, 1] = filtfilt(b, a, states[:, 1], axis=0)
        states[:, 2] = filtfilt(b, a, states[:, 2], axis=0)

        target_times_list.append(target_times)
        states_list.append(states)
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_processed_all = np.concatenate(states_list, axis=0)
    episode_ends_processed = np.array(episode_ends_list)

    return (states_raw, times_raw, episode_ends_raw, 
            states_processed_all, target_times_all, episode_ends_processed)

""" def import_data(zarr_root_str: str):
    zarr_root = Path(zarr_root_str)
    print(f"Loading Zarr: {zarr_root}")
    zarr_data = zarr.open(str(zarr_root), mode="r")
    
    target_hz = 100.0
    lowpass_cutoff = 1.0 
    butter_order = 4
    
    print(f"  > Processing params: Target Hz={target_hz}, Cutoff={lowpass_cutoff}Hz")

    episode_ends_raw = zarr_data["data/episode_ends"][:]  
    states_raw = zarr_data["data/state"][:]  
    times_raw = zarr_data["data/time"][:]  
    
    times_raw = np.asarray(times_raw).ravel()
    dt = 1.0 / float(target_hz)

    target_times_list = []
    states_list = []
    episode_ends_list = []
    n_list = 0

    for trial in range(episode_ends_raw.shape[0]):
        if trial == 0: dex_start = 0
        else: dex_start = episode_ends_raw[trial - 1]
        dex_end = episode_ends_raw[trial] - 1
        dex_states = np.arange(dex_start, dex_end)

        # --- FILTRO DE DUPLICADOS ---
        # Extraer tiempos, encontrar los únicos y mantener solo esos índices
        #t_episode = times_raw[dex_states_raw]
        #_, unique_idx = np.unique(t_episode, return_index=True)
        #dex_states = dex_states_raw[unique_idx]
        # ----------------------------

        # 2. Crear target_times usando los límites de tiempo limpios
        t_start_ep = times_raw[dex_states[0]]
        t_end_ep = times_raw[dex_states[-1]]
        target_times = np.arange(t_start_ep + dt, t_end_ep - dt, dt)
        
        states = np.zeros((len(target_times), 7))


        # 1. INTERPOLACIÓN POSICIÓN
        for i in range(3):
            f_interp = interp1d(times_raw[dex_states], states_raw[dex_states, i], kind="cubic")
            states[:, i] = f_interp(target_times)

        # 2. INTERPOLACIÓN ROTACIÓN
        rot_in = Rotation.from_quat(states_raw[dex_states, 3:7])
        slerp = Slerp(times_raw[dex_states], rot_in)
        states[:, 3:7] = slerp(target_times).as_quat()

        # 3. FILTRADO
        b, a = butter(butter_order, lowpass_cutoff / (target_hz / 2), btype="low", analog=False)
        states[:, 0] = filtfilt(b, a, states[:, 0], axis=0)
        states[:, 1] = filtfilt(b, a, states[:, 1], axis=0)
        states[:, 2] = filtfilt(b, a, states[:, 2], axis=0)

        target_times_list.append(target_times)
        states_list.append(states)
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_processed_all = np.concatenate(states_list, axis=0)
    episode_ends_processed = np.array(episode_ends_list)

    return (states_raw, times_raw, episode_ends_raw, 
            states_processed_all, target_times_all, episode_ends_processed) """

# -----------------------------------------------------------------------------
# 2. MOTOR MATEMÁTICO: MINIMUM JERK
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
    
    # MinJerk Polynomial derivative (Velocity profile)
    # P(tau) = 10t^3 - 15t^4 + 6t^5 (Position 0 to 1)
    # V(tau) = (1/D) * (30t^2 - 60t^3 + 30t^4)
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
    """
    if alpha <= 1e-9:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x

    N_features = A.shape[1]
    
    # Matriz de Regularización (Identidad escalada por alpha)
    # La última columna es el Intercepto, NO queremos regularizarla.
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
# 3. ALGORITMOS DE AJUSTE (ADAPTADOS A MINJERK 3D)
# -----------------------------------------------------------------------------

def fit_hybrid_peaks_greedy_minjerk(t, frefs, max_bases_total=100, residual_tol=0.03, verbose=False):
    """
    Versión adaptada del greedy de LGNB pero usando MinJerk (sin bucle de Mu).
    """
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
        widths, _, left_ips, right_ips = peak_widths(vt, peaks_indices, rel_height=0.65)
        for i, idx in enumerate(peaks_indices):
            t_left = left_ips[i] * dt + t[0]
            t_right = right_ips[i] * dt + t[0]
            center_time = (t_left + t_right) / 2.0
            width_fwhm = t_right - t_left
            
            # MinJerk suele ser más ancho que la FWHM gaussiana pura
            duration = np.clip(width_fwhm * 1.8, 0.2, 3.5)
            starts_found.append(center_time - (duration / 2.0))
            ends_found.append(center_time + (duration / 2.0))

    # Residuo inicial
    if len(starts_found) > 0:
        Phi = build_phi_matrix(t, np.array(starts_found), np.array(ends_found))
        s, _ = nnls(Phi, vt)
        residual = vt - Phi.dot(s)
    else:
        residual = vt.copy()

    rms = np.sqrt(np.mean(residual**2))
    
    #Added for the plot on the paper:
    """ Phi_init = build_phi_matrix(t, ts_init, te_init)
    s_init, _ = nnls(Phi_init, vt_ref) """
    vt_fit_init = Phi.dot(s)
    
    plot_velocity_and_bases(t, vt, vt_fit_init, np.array(starts_found), np.array(ends_found), s, 
                            dim_color='tab:cyan', title='STEP 0b:')
    
    # --- FASE B: Grid Search ---
    candidate_durations = np.arange(0.2, 3.5, 0.02)
    
    if rms > max_vt * residual_tol:
        if verbose: print(f"  [MinJerk] Fase B: Grid Search Duration...")
        
        for k in range(max_bases_total - len(starts_found)):
            idx_max = np.argmax(residual)
            if residual[idx_max] < max_vt * residual_tol: break
            t_res_peak = t[idx_max]

            # Check proximidad a existentes
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
            
            vt_new = Phi_new.dot(s_new)
            
            # --- MODIFICACIÓN VISUAL: Asignamos colores ---
            # Todas las bases existentes (len-1) serán azules, la última añadida será roja
            iter_colors = ['tab:blue'] * (len(starts_found) - 1) + ['tab:red']
            
            plot_velocity_and_bases_paper(t, vt, vt_new, np.array(starts_found), np.array(ends_found), s_new, 
                            dim_color=iter_colors, title=f'Iterative addition of MinJerk - {len(starts_found)} bases')

    return np.array(starts_found), np.array(ends_found)

def compute_position_error_minjerk_3d(params, t, P_ref):
    """
    Función de coste 3D usando Ridge Regression.
    """
    N, D = P_ref.shape # D=3 (X,Y,Z)
    K = params.size // 2
    t_starts = params[:K]
    t_ends = params[K:]
    
    MAX_DURATION = 5.0 
    
    # Penalizaciones temporales
    if np.any(t_starts >= t_ends - 0.02): return 1e12
    if np.any((t_ends - t_starts) > MAX_DURATION): return 1e12

    # Construcción base
    Phi_vel = build_phi_matrix(t, t_starts, t_ends)
    
    # Integración para posición
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(K):
        Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
        
    # Añadir columna de unos para el offset inicial (Intercept)
    A = np.hstack([Phi_pos, np.ones((N, 1))])
    
    # --- RIDGE REGRESSION IMPLEMENTATION (Sum of errors per dimension) ---
    total_error = 0
    for d in range(D):
        w = solve_ridge_weighted(A, P_ref[:, d], alpha=RIDGE_ALPHA)
        fit_d = A.dot(w)
        total_error += np.sum((fit_d - P_ref[:, d])**2)
    
    return total_error

def refine_bases_minjerk(t, P_ref, t_starts_init, t_ends_init, verbose=False, step_title="Global Opt"):
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
            options={'maxiter': 2000, 'disp': verbose})
    
    if verbose: print(f"  Done in {time.time()-t0:.2f}s. Cost: {res.fun:.4e}")
    
    params_opt = res.x
    ts = params_opt[:K]
    te = params_opt[K:]
    idx = np.argsort(ts)
    return ts[idx], te[idx]

def merge_bases_minjerk_3d(t_starts, t_ends, scales, proximity_tol=0.05):
    """
    Mezcla bases cercanas. scales tiene shape (K, D).
    """
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
        # Magnitud vectorial para ponderar
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
# 4. PLOTTING HELPERS (MISMOS QUE LGNB, RENOMBRADOS)
# -----------------------------------------------------------------------------
def plot_velocity_and_bases_paper(t, v_ref, v_fit, t_starts, t_ends, scales_dim, 
                            gt_starts=None, gt_ends=None, gt_scales=None,
                            dim_color='tab:red', title='Velocity dimension'):
    N = t.size
    Phi = build_phi_matrix(t, t_starts, t_ends) if t_starts.size > 0 else np.zeros((N, 0))
    scaled = Phi * scales_dim[None, :] if Phi.size > 0 else np.zeros((N, 0))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='Reconstructed')

    # --- MODIFICACIÓN: Soporte para múltiples colores ---
    if isinstance(dim_color, str):
        colors = [dim_color] * scaled.shape[1]
    else:
        colors = dim_color

    has_old_label = False
    has_new_label = False
    
    for k in range(scaled.shape[1]):
        c = colors[k]
        label_str = None
        
        # Asignar leyenda dinámicamente según si usamos 1 color o 2 colores
        if isinstance(dim_color, str):
            if not has_old_label:
                label_str = 'MinJerk Bases'
                has_old_label = True
        else:
            if c == 'tab:red' and not has_new_label:
                label_str = 'New Basis'
                has_new_label = True
            elif c == 'tab:blue' and not has_old_label:
                label_str = 'Previous Bases'
                has_old_label = True

        plt.fill_between(t, 0, scaled[:, k], alpha=0.3, color=c,
                        label=label_str, linewidth=0)
        plt.plot(t, scaled[:, k], color=c, alpha=0.8, lw=1) 
        
    plt.legend(loc='upper right', ncol=3, fontsize='small')
    plt.xlabel('Time (t)'); plt.ylabel('Velocity'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    
    # Me aseguro de que el directorio existe por si acaso
    if not os.path.exists("Plots_AdrianData_MinJerk_3D"):
        os.makedirs("Plots_AdrianData_MinJerk_3D")
        
    plt.savefig(os.path.join("Plots_AdrianData_MinJerk_3D", title.replace("\n", "").replace(":", "").replace(" ", "_") + ".svg"))
    plt.close()


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
    plt.savefig(os.path.join("Plots_AdrianData_MinJerk_3D", title.replace("\n", "").replace(":", "").replace(" ", "_") + ".svg"))
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
    plt.savefig(os.path.join("Plots_AdrianData_MinJerk_3D", title.replace(" ", "_") + ".svg"))
    plt.close()

def plot_position_and_integrated_bases(t, p_ref, p_fit, t_starts, t_ends, Phi_pos, scales_dim, 
                                    pos_color='tab:red', neg_color='tab:blue', title='Position dim'):
    N = t.size
    scaled_pos = Phi_pos * scales_dim[None, :] if Phi_pos.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    plt.plot(t, p_ref, 'k-', lw=2.5, label='Original')
    plt.plot(t, p_fit, '--', lw=2, color='green', label='Reconstructed')
    
    for k in range(scaled_pos.shape[1]):
        if np.any(scaled_pos[:, k] >= 0):
            plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]>=0, facecolor=pos_color, alpha=0.2)
        if np.any(scaled_pos[:, k] < 0):
            plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k]<0, facecolor=neg_color, alpha=0.2)

    plt.xlabel('Time (t)'); plt.ylabel('Position'); plt.title(title)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join("Plots_AdrianData_MinJerk_3D", title.replace(" ", "_") + ".svg"))
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
    plt.savefig(os.path.join("Plots_AdrianData_MinJerk_3D", "3D_Trajectory.svg"))
    #plt.close()
    plt.show()

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. LOAD DATA (ADRIAN DATASET)
    #zarr_path = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/adrian_data/spatula_pose_raw.zarr"
    zarr_path = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/adrian_data/pushing_2026-02-20-16-16/spatula_pose_raw.zarr"
    #zarr_path = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/adrian_data/small_scoop_2026-01-28-17-27/small_scoop_2026-01-28-17-27/spatula_pose_raw.zarr"
    # Usamos la funcion import_data importada del estilo LGNB
    (states_raw, times_raw, episode_ends_raw, 
    states_proc, times_proc, episode_ends_proc) = import_data(zarr_path)
    
    EPISODE_IDX = 1
    print(f"\nMinJerk Analysis - Episode {EPISODE_IDX} (Ridge: {RIDGE_ALPHA})...")

    # --- DATOS PROCESADOS (FILTRADOS) ---
    start_proc = 0 if EPISODE_IDX == 0 else episode_ends_proc[EPISODE_IDX-1]
    end_proc = episode_ends_proc[EPISODE_IDX]
    t_proc_ep = times_proc[start_proc:end_proc]
    p_proc_ep = states_proc[start_proc:end_proc, :3] # XYZ
    
    t = t_proc_ep - t_proc_ep[0]
    #t= t[85:320] # Corto para acelerar pruebas
    #P = p_proc_ep[85:320, :3] # XYZ (corto para acelerar pruebas)
    P=p_proc_ep
    
    
    
    # Derivar velocidad del P procesado para la inicialización
    V_ref_3d = np.zeros_like(P)
    for i in range(3):
        V_ref_3d[:, i] = np.gradient(P[:, i], t)
    vt_ref = np.linalg.norm(V_ref_3d, axis=1)
    
    """ # --- 2. Matar el rebote final de la velocidad ---
    # Sabemos que los últimos 19 valores son los que vuelven a subir.
    puntos_rebote = 50 

    # Creamos una curva de atenuación que va de 1.0 bajando suavemente a 0.0
    curva_frenado = np.cos(np.linspace(0, np.pi/2, puntos_rebote))

    # Multiplicamos la velocidad 3D de esos últimos puntos por la curva
    for i in range(3):
        V_ref_3d[-puntos_rebote:, i] = V_ref_3d[-puntos_rebote:, i] * curva_frenado

    # --- 3. Calcular la velocidad tangencial final ---
    # Al recalcularla ahora, los valores irán bajando desde ~0.0242 hasta 0.0
    vt_ref = np.linalg.norm(V_ref_3d, axis=1)
    
    print(vt_ref) """
    
    # --- PIPELINE ---
    
    # 1. INITIAL GUESS (GREEDY MINJERK)
    print("STEP 1: Getting initial guess (MinJerk Greedy)...")
    t_init_contador = time.time()
    ts_init, te_init = fit_hybrid_peaks_greedy_minjerk(t, [vt_ref], verbose=True, residual_tol=0.0)

    # Plot Initial Guess
    Phi_init = build_phi_matrix(t, ts_init, te_init)
    s_init, _ = nnls(Phi_init, vt_ref)
    vt_fit_init = Phi_init.dot(s_init)
    
    plot_velocity_and_bases(t, vt_ref, vt_fit_init, ts_init, te_init, s_init, 
                            dim_color='tab:cyan', title='STEP 1b: Initial Guess (Fitted to vt_ref)')
    
    # 2. GLOBAL OPTIMIZATION (3D)
    # Refina tiempos (ts, te) minimizando error en XYZ
    ts_opt, te_opt = refine_bases_minjerk(t, P, ts_init, te_init, verbose=False)
    
    # 3. MERGING & CLEANUP
    # Calcular escalas intermedias 3D para hacer el merge
    Phi_vel = build_phi_matrix(t, ts_opt, te_opt)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(Phi_pos.shape[1]): Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
    
    A_temp = np.hstack([Phi_pos, np.ones((len(t), 1))])
    
    # Resolver Ridge temporalmente para cada dimensión
    scales_list_temp = []
    for d in range(3):
        w = solve_ridge_weighted(A_temp, P[:, d], alpha=RIDGE_ALPHA)
        scales_list_temp.append(w[:-1]) # Sin intercepto
    curr_scales = np.column_stack(scales_list_temp)
    
    print(f"\nSTEP 3: Merging & Re-Optimization...")
    ts_final, te_final, scales_final = merge_bases_minjerk_3d(ts_opt, te_opt, curr_scales, proximity_tol=0.04)
    
    # Refinamiento Final
    ts_final, te_final = refine_bases_minjerk(t, P, ts_final, te_final, verbose=False, step_title="Final Refine")
    
    K_final = ts_final.size
    
    # 4. CALCULO FINAL DE ESCALAS (3D RIDGE)
    Phi_vel_final = build_phi_matrix(t, ts_final, te_final)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K_final):
        Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)

    A_final = np.hstack([Phi_pos_final, np.ones((len(t), 1))])
    
    scales_list_final = []
    p0_list = []
    
    # Loop por dimensiones X, Y, Z
    for d in range(3):
        w = solve_ridge_weighted(A_final, P[:, d], alpha=RIDGE_ALPHA)
        scales_list_final.append(w[:-1])
        p0_list.append(w[-1]) # Intercepto
        
    scales_final = np.column_stack(scales_list_final)
    
    # Reconstrucción
    P_fit = np.zeros_like(P)
    for d in range(3):
        w_full = np.concatenate([scales_list_final[d], [p0_list[d]]])
        P_fit[:, d] = A_final.dot(w_full)
    
    V_fit = Phi_vel_final.dot(scales_final)
    vt_fit = np.linalg.norm(V_fit, axis=1)
    scales_mag = np.linalg.norm(scales_final, axis=1)
    
    t_end_contador = time.time()
    print("Total Time: {:.2f}s".format(t_end_contador - t_init_contador))
    
    # --- PLOTS ---
    print("\nGenerating Plots (MinJerk 3D)...")
    rmse = np.sqrt(np.mean(np.sum((P - P_fit)**2, axis=1)))
    print(f"RMSE 3D: {rmse:.5f}")

    plot_3d_trajectory_pos(P, P_fit)
    
    plot_position_and_integrated_bases(t, P[:, 0], P_fit[:, 0], ts_final, te_final, Phi_pos_final, scales_final[:, 0], 
                                    pos_color='tab:red', neg_color='tab:cyan', title='Pos X (MinJerk)')
    plot_position_and_integrated_bases(t, P[:, 1], P_fit[:, 1], ts_final, te_final, Phi_pos_final, scales_final[:, 1],
                                    pos_color='tab:green', neg_color='tab:purple', title='Pos Y (MinJerk)')
    plot_position_and_integrated_bases(t, P[:, 2], P_fit[:, 2], ts_final, te_final, Phi_pos_final, scales_final[:, 2],
                                    pos_color='tab:orange', neg_color='tab:blue', title='Pos Z (MinJerk)')

    plot_velocity_and_bases_signed(t, V_ref_3d[:, 0], V_fit[:, 0], ts_final, te_final, Phi_vel_final, scales_final[:, 0], 
                                pos_color='tab:red', neg_color='tab:cyan', title='Vel X (MinJerk)')
    plot_velocity_and_bases_signed(t, V_ref_3d[:, 1], V_fit[:, 1], ts_final, te_final, Phi_vel_final, scales_final[:, 1], 
                                pos_color='tab:green', neg_color='tab:purple', title='Vel Y (MinJerk)')
    plot_velocity_and_bases_signed(t, V_ref_3d[:, 2], V_fit[:, 2], ts_final, te_final, Phi_vel_final, scales_final[:, 2], 
                                pos_color='tab:orange', neg_color='tab:blue', title='Vel Z (MinJerk)')

    plot_velocity_and_bases(t, vt_ref, vt_fit, ts_final, te_final, scales_mag, 
                            dim_color='tab:orange', title='Final Tangential Velocity (MinJerk)')
    
    print(f"\nAll plots saved in 'Plots_AdrianData_MinJerk_3D'")

    print("\nFinal Parameters (MinJerk 3D):")
    print(f"{'ID':<3} | {'Start':<8} | {'End':<8} | {'Dur':<8} | {'Sx':<8} | {'Sy':<8} | {'Sz':<8}")
    
    sort_idx = np.argsort(ts_final)
    for k in range(K_final):
        idx = sort_idx[k]
        dur = te_final[idx] - ts_final[idx]
        sx = scales_final[idx, 0]
        sy = scales_final[idx, 1]
        sz = scales_final[idx, 2]
        print(f"{k:<3} | {ts_final[idx]:.3f}    | {te_final[idx]:.3f}    | {dur:.3f}    | {sx:.3f}    | {sy:.3f}    | {sz:.3f}")