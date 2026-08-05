import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import os
from scipy.integrate import cumtrapz
from scipy.optimize import nnls, minimize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

MIN_T = 0.000001
np.set_printoptions(precision=4, suppress=True)

# -----------------------------------------------------------
# CORE: MINIMUM JERK
# -----------------------------------------------------------
def compute_single_base(t, mode, sigma, min_t=MIN_T):
    """
    Computes a Minimum Jerk velocity profile.
    
    Parameters (Mapping):
      - mode: Central time of the movement (Peak Time).
      - sigma: TOTAL duration of the movement (Duration).
    """
    t = np.asarray(t)
    
    #Time limits
    duration = max(sigma, 1e-4) 
    t_start = mode - duration / 2.0
    t_end   = mode + duration / 2.0
    
    val = np.zeros_like(t)
    
    mask = (t >= t_start) & (t <= t_end)
    
    if not np.any(mask):
        return val
    
    tau = (t[mask] - t_start) / duration
    
    # Normalized Minimum Jerk velocity profile (Integral = 1)
    # Standard form of velocity: v(tau) = 30*t^2 - 60*t^3 + 30*t^4
    # To ensure the integral equals 1 (and works as a scalable basis), we divide by the duration.
    shape = 30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)
    val[mask] = shape / duration
    
    return val

def build_phi_matrix(t, modes, sigmas):
    """Matrix (N x K)"""
    t = np.asarray(t)
    modes = np.asarray(modes)
    if modes.size == 0:
        return np.zeros((t.size, 0))
        
    # Broadcasting sigmas
    if np.isscalar(sigmas):
        sigmas = np.full(modes.shape, sigmas, dtype=float)
    else:
        sigmas = np.asarray(sigmas)
        
    K = modes.size
    Phi = np.zeros((t.size, K))
    for k in range(K):
        Phi[:, k] = compute_single_base(t, modes[k], sigmas[k])
    return Phi

def smooth_trajectory_savgol(P, window_length, polyorder):
    if window_length % 2 == 0:
        window_length += 1 
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        window_length = window_length + 1 if window_length % 2 == 0 else window_length

    Px_smoothed = savgol_filter(P[:, 0], window_length, polyorder)
    Py_smoothed = savgol_filter(P[:, 1], window_length, polyorder)
    return np.column_stack([Px_smoothed, Py_smoothed])



def fit_sparse_by_predicted_reduction(t, frefs,
                                    candidate_grid_step=0.02,
                                    candidate_sigmas=None,
                                    top_sigmas_per_mode=1,
                                    max_bases=50,
                                    min_pred_abs=1e-6, min_pred_rel=1e-4,
                                    min_mode_dist_factor=0.5,
                                    prune_thresh=1e-4,
                                    merge_mode_tol_factor=0.5,
                                    verbose=False):
    t_global_start = time.time()
    if isinstance(frefs, list):
        F = np.column_stack(frefs)
    else:
        F = np.asarray(frefs)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        elif F.ndim == 2 and F.shape[1] == t.size:
            F = F.T
            
    N, D = F.shape
    tmin, tmax = float(np.min(t)), float(np.max(t))
    span = tmax - tmin
    
    # Grid de modos (tiempos centrales)
    step = candidate_grid_step * max(span, 1.0)
    if step <= 0: step = span / 20.0
    start = max(tmin + step * 0.5, MIN_T + step * 0.5)
    candidate_modes = np.arange(start, tmax, step)
    C = candidate_modes.size
    
    if C == 0:
        return np.array([]), np.array([]), [np.zeros(0) for _ in range(D)], [np.zeros(N) for _ in range(D)], {}

    # Grid  MinJerk
    if candidate_sigmas is None:
        candidate_sigmas = np.linspace(0.1, 0.8, 20)
    candidate_sigmas = np.unique(np.clip(candidate_sigmas, 0.01, span*1.5))
    S = candidate_sigmas.size
    
    
    B_mode_sigma = []
    for i, m in enumerate(candidate_modes):
        Bms = np.zeros((N, S))
        for j, s in enumerate(candidate_sigmas):
            Bms[:, j] = compute_single_base(t, m, s)
        B_mode_sigma.append(Bms)
        
    Residual_init = F.copy()
    bnorms_per_mode = [np.sum(B**2, axis=0) + 1e-12 for B in B_mode_sigma]
    
    # Greedy pre-selection
    candidates = []
    for i, m in enumerate(candidate_modes):
        Bms = B_mode_sigma[i]
        bnorms = bnorms_per_mode[i]
        BtR = Bms.T.dot(Residual_init)
        pred_red = np.sum((BtR**2) / bnorms[:, None], axis=1)
        
        top_k = min(max(1, int(top_sigmas_per_mode)), S)
        idx_sorted = np.argsort(pred_red)[::-1][:top_k]
        for idx in idx_sorted:
            candidates.append((m, candidate_sigmas[idx]))
            
    cand_modes_exp = np.array([c[0] for c in candidates])
    cand_sigmas_exp = np.array([c[1] for c in candidates])
    M = cand_modes_exp.size

    B_full = np.zeros((N, M))
    for j in range(M):
        B_full[:, j] = compute_single_base(t, cand_modes_exp[j], cand_sigmas_exp[j])
        
    bnorm2 = np.sum(B_full * B_full, axis=0) + 1e-12
    selected = []
    selected_mask = np.zeros(M, dtype=bool)
    Residual = F.copy()
    
    time_step = (t[1] - t[0]) if N > 1 else 0.0
    min_mode_dist = max(min_mode_dist_factor * time_step, 1e-8)
    
    while len(selected) < max_bases:
        BtR = B_full.T.dot(Residual)
        pred_red = np.sum((BtR**2) / bnorm2[:, None], axis=1)
        pred_red_masked = pred_red.copy()
        pred_red_masked[selected_mask] = 0.0
        
        if len(selected) > 0:
            sel_modes = cand_modes_exp[np.array(selected)]
            dists = np.abs(cand_modes_exp[:, None] - sel_modes[None, :])
            min_dists = np.min(dists, axis=1)
            pred_red_masked[min_dists < min_mode_dist] = 0.0
            
        best_j = int(np.argmax(pred_red_masked))
        best_value = float(pred_red_masked[best_j])
        total_RSS = float(np.sum(Residual**2))
        rel_best = best_value / (total_RSS + 1e-12)
        
        if (best_value < min_pred_abs) or (rel_best < min_pred_rel):
            break
            
        selected.append(best_j)
        selected_mask[selected[-1]] = True
        
        sel_idx = np.array(selected, dtype=int)
        Bsel = B_full[:, sel_idx]
        A_sel = np.zeros((Bsel.shape[1], D))
        for d in range(D):
            A_sel[:, d], _ = nnls(Bsel, F[:, d])
        Residual = F - Bsel.dot(A_sel)
        
        if len(selected) == max_bases:
            break

    if len(selected) == 0:
        return np.array([]), np.array([]), [np.zeros(0) for _ in range(D)], [np.zeros(N) for _ in range(D)], {}

    # Pruning final
    sel_idx = np.array(selected, dtype=int)
    Bsel = B_full[:, sel_idx]
    A_sel = np.zeros((Bsel.shape[1], D))
    for d in range(D):
        A_sel[:, d], _ = nnls(Bsel, F[:, d])
        
    row_norms = np.linalg.norm(A_sel, axis=1)
    keep = np.where(row_norms >= prune_thresh)[0]
    if keep.size == 0: keep = np.array([np.argmax(row_norms)])
    
    sel_idx = sel_idx[keep]
    A_sel = A_sel[keep, :]
    
    modes_init = cand_modes_exp[sel_idx].copy()
    sigmas_init = cand_sigmas_exp[sel_idx].copy()
    scales_init = A_sel.T
    
    # Merge
    def merge_close(modes, sigmas, scales, tol):
        order = np.argsort(modes)
        modes = modes[order].tolist()
        sigmas = sigmas[order].tolist()
        scales = scales[:, order].copy()
        
        merged_modes, merged_sigmas, merged_scales = [], [], []
        i = 0
        Kcur = len(modes)
        while i < Kcur:
            j = i + 1
            group = [i]
            while j < Kcur and abs(modes[j] - modes[i]) < tol:
                group.append(j); j += 1
            
            if len(group) == 1:
                merged_modes.append(modes[i])
                merged_sigmas.append(sigmas[i])
                merged_scales.append(scales[:, i])
            else:
                group_scales = scales[:, group]
                weights = np.linalg.norm(group_scales, axis=0) + 1e-12
                wm = np.sum(np.array(modes)[group] * weights) / np.sum(weights)
                ws = np.sum(np.array(sigmas)[group] * weights) / np.sum(weights)
                merged_modes.append(float(wm))
                merged_sigmas.append(float(ws))
                merged_scales.append(np.sum(group_scales, axis=1))
            i = j
            
        merged_modes = np.array(merged_modes)
        merged_sigmas = np.array(merged_sigmas)
        merged_scales = np.column_stack(merged_scales) if len(merged_scales) > 0 else np.zeros((scales.shape[0], 0))
        return merged_modes, merged_sigmas, merged_scales

    merge_tol = max(merge_mode_tol_factor * (t[1] - t[0]) if N > 1 else merge_mode_tol_factor * 1e-3, 1e-8)
    modes_merged, sigmas_merged, scales_merged = merge_close(modes_init, sigmas_init, scales_init, merge_tol)
    
    # Final output
    modes_ref, sigmas_ref, scales_ref = modes_merged, sigmas_merged, scales_merged
    B_final = build_phi_matrix(t, modes_ref, sigmas_ref)
    Fhat = B_final.dot(scales_ref.T)
    scales_list = [scales_ref[d, :] for d in range(D)]
    fs_est = [Fhat[:, d] for d in range(D)]
    
    return modes_ref, sigmas_ref, scales_list, fs_est, {}


def merge_bases_by_proximity(modes, sigmas, scales_x, scales_y, mode_tol=1e-4, sigma_tol=1e-4):
    K = modes.size
    if K == 0: return modes, sigmas, scales_x, scales_y
    
    scales = np.column_stack([scales_x, scales_y])
    order = np.argsort(modes)
    modes_s = modes[order].copy()
    sigmas_s = sigmas[order].copy()
    scales_s = scales[order, :].copy()

    merged_modes, merged_sigmas, merged_scales = [], [], []

    i = 0
    while i < K:
        current_mode = modes_s[i]
        current_sigma = sigmas_s[i]
        group_indices = [i]
        j = i + 1
        while j < K and abs(modes_s[j] - current_mode) < mode_tol and abs(sigmas_s[j] - current_sigma) < sigma_tol:
            group_indices.append(j); j += 1
        
        group_scales = scales_s[group_indices, :]
        new_scales_summed = np.sum(group_scales, axis=0)
        
        weights = np.linalg.norm(group_scales, axis=1) + 1e-12
        new_mode = np.sum(modes_s[group_indices] * weights) / np.sum(weights)
        new_sigma = np.sum(sigmas_s[group_indices] * weights) / np.sum(weights)
        
        if len(group_indices) > 1:
            print(f"\n  [MERGE] Fusión detectada de {len(group_indices)} bases:")
            print(f"    {'OrigIdx':^7} | {'Mode':^8} | {'Sigma':^8} | {'Scale X':^9} | {'Scale Y':^9}")
            print(f"    {'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}")
            
            for g_idx in group_indices:
                orig_idx = order[g_idx] # Índice original antes de ordenar
                m_val = modes_s[g_idx]
                s_val = sigmas_s[g_idx]
                sx_val = scales_s[g_idx, 0]
                sy_val = scales_s[g_idx, 1]
                print(f"    {orig_idx:^7} | {m_val:8.4f} | {s_val:8.4f} | {sx_val:9.4f} | {sy_val:9.4f}")
            
            print(f"    {'-'*50}")
            print(f"    => RESULT| {new_mode:8.4f} | {new_sigma:8.4f} | {new_scales_summed[0]:9.4f} | {new_scales_summed[1]:9.4f}")
            print(f"    {'-'*50}")

        merged_modes.append(new_mode)
        merged_sigmas.append(new_sigma)
        merged_scales.append(new_scales_summed)
        i = j

    merged_modes = np.array(merged_modes)
    merged_sigmas = np.array(merged_sigmas)
    merged_scales = np.array(merged_scales)
    
    return merged_modes, merged_sigmas, merged_scales[:, 0], merged_scales[:, 1]


# -----------------------------------------------------------
# GLOBAL OPTIMIZATION (POSITION ERROR)
# -----------------------------------------------------------
def compute_position_error(params, t, P_ref):
    N, D = P_ref.shape
    K = params.size // 2
    modes = params[:K]
    sigmas = params[K:]
    
    if np.any(sigmas < 0.001): return 1e12

    Phi_vel = build_phi_matrix(t, modes, sigmas)
    Phi_pos = np.zeros_like(Phi_vel)
    for k in range(K):
        Phi_pos[:, k] = cumtrapz(Phi_vel[:, k], t, initial=0.0)
        
    A_matrix = np.hstack([Phi_pos, np.ones((N, 1))])
    
    try:
        S_x_with_intercept, _, _, _ = np.linalg.lstsq(A_matrix, P_ref[:, 0], rcond=None)
        S_y_with_intercept, _, _, _ = np.linalg.lstsq(A_matrix, P_ref[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return 1e12

    P_fit_x = A_matrix.dot(S_x_with_intercept)
    P_fit_y = A_matrix.dot(S_y_with_intercept)
    P_fit = np.column_stack([P_fit_x, P_fit_y])
    
    return np.sum((P_fit - P_ref)**2)

def refine_bases_for_position_error(t, P_ref, modes_init, sigmas_init, step_title="Global Optimization", verbose=False):
    print(f"\n{step_title}: Optimizing {modes_init.size} Min-Jerk bases...")
    params_init = np.concatenate([modes_init, sigmas_init])
    K = modes_init.size
    t_min, t_max = np.min(t), np.max(t)
    span = t_max - t_min
    
    mode_bounds = [(t_min - 0.2, t_max + 0.2) for _ in range(K)]
    sigma_bounds = [(0.01, span * 1.5) for _ in range(K)]
    bounds = mode_bounds + sigma_bounds
    
    cost_func = lambda params: compute_position_error(params, t, P_ref)
    
    t0 = time.time()
    result = minimize(cost_func, params_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'disp': verbose})
    
    params_final = result.x
    modes_final = params_final[:K]
    sigmas_final = params_final[K:]
    
    sort_idx = np.argsort(modes_final)
    return modes_final[sort_idx], sigmas_final[sort_idx]


# -----------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------
def plot_velocity_and_bases(t, v_ref, v_fit, modes, sigmas, scales_dim, 
                            dim_color='tab:red', title='Velocity dim', plot_mode='individual'):
    N = t.size
    Phi = build_phi_matrix(t, modes, sigmas) if modes.size > 0 else np.zeros((N, 0))
    scaled = Phi * scales_dim[None, :] if Phi.size > 0 else np.zeros((N, 0))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='v_ref (Original)')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='v_fit (Fitted)')

    if plot_mode == 'stacked':
        cumulative_sum = np.zeros(N)
        for k in range(scaled.shape[1]):
            next_sum = cumulative_sum + scaled[:, k]
            plt.fill_between(t, cumulative_sum, next_sum, alpha=0.7, color=dim_color, linewidth=0)
            cumulative_sum = next_sum
    elif plot_mode == 'individual':
        has_base_label = False
        for k in range(scaled.shape[1]):
            plt.fill_between(t, 0, scaled[:, k], alpha=0.3, color=dim_color, linewidth=0)
            plt.plot(t, scaled[:, k], color=dim_color, alpha=0.8, lw=1) 
            has_base_label = True
            
    if modes.size > 0:
        for m in modes:
            if t[0] <= m <= t[-1]: plt.axvline(m, color='k', linestyle=':', alpha=0.2)
    
    plt.legend(loc='upper right', fontsize='small')
    plt.xlabel('Time (t)'); plt.ylabel('Velocity'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.show()

def plot_2d_trajectory_pos(xs, ys, x_fit, y_fit):
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'k-', lw=2, label='Original')
    plt.plot(x_fit, y_fit, 'r--', lw=2, label='Fitted (MinJerk)')
    plt.legend()
    plt.title('2D Trajectory'); plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.show()
    
def plot_velocity_and_bases_signed(t, v_ref, v_fit, modes, Phi_vel, scales_dim, pos_color='tab:red', neg_color='tab:blue', title='Velocity Decomposition'):
    N = t.size
    # Calculamos las bases escaladas (velocidad * escala)
    scaled_vel = Phi_vel * scales_dim[None, :] if Phi_vel.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    

    plt.plot(t, v_ref, 'k-', lw=2.5, label='v_ref (Original)')
    plt.plot(t, v_fit, '--', lw=2, color='green', label='v_fit (Reconstructed)')


    has_pos, has_neg = False, False
    for k in range(scaled_vel.shape[1]):
        curve = scaled_vel[:, k]
        

        if np.any(curve >= 0):
            plt.fill_between(t, 0, curve, 
                            where=curve >= 0, 
                            facecolor=pos_color, alpha=0.3, interpolate=True,
                            label='Positive Base' if not has_pos else None)
            if not has_pos: has_pos = True
            

            plt.plot(t, np.ma.masked_less(curve, 0), color=pos_color, lw=1, alpha=0.6)


        if np.any(curve < 0):
            plt.fill_between(t, 0, curve, 
                            where=curve < 0, 
                            facecolor=neg_color, alpha=0.3, interpolate=True,
                            label='Negative Base' if not has_neg else None)
            if not has_neg: has_neg = True
            
            plt.plot(t, np.ma.masked_greater(curve, 0), color=neg_color, lw=1, alpha=0.6)

    if modes.size > 0:
        for m in modes:
            if t[0] <= m <= t[-1]: 
                plt.axvline(m, color='k', linestyle=':', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_position_and_integrated_bases(t, p_ref, p_fit, modes_v, Phi_pos, scales_dim, 
                                    pos_color='tab:red', neg_color='tab:blue', title='Position'):
    N = t.size
    scaled_pos = Phi_pos * scales_dim[None, :] if Phi_pos.size > 0 else np.zeros((N, 0))

    plt.figure(figsize=(10, 5))
    plt.plot(t, p_ref, 'k-', lw=2.5, label='p_ref')
    plt.plot(t, p_fit, '--', lw=2, color='green', label='p_fit')
    
    for k in range(scaled_pos.shape[1]):
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k] >= 0, 
                        facecolor=pos_color, alpha=0.2, interpolate=True)
        plt.fill_between(t, scaled_pos[:, k], 0, where=scaled_pos[:, k] < 0, 
                        facecolor=neg_color, alpha=0.2, interpolate=True)

    plt.xlabel('Time'); plt.ylabel('Position'); plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.show()

def plot_velocity_comparison(t, v_ref, v_fit, title='Velocity Comparison'):
    rmse = np.sqrt(np.mean((v_ref - v_fit)**2))
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_ref, 'k-', lw=2.5, label='v_ref')
    plt.plot(t, v_fit, 'r--', lw=2, label='v_fit (derivative of P_fit)')
    plt.title(f"{title}\nRMSE: {rmse:.6f}")
    plt.xlabel('Time'); plt.ylabel('Velocity'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(); plt.show()


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)

    # --- Data Loading ---
    FILE_PATH = os.path.dirname(os.path.realpath(__file__)) if '__file__' in locals() else '.'
    DATASET_LETTER_FILE = os.path.join(FILE_PATH, "./data/G.npy")
    
    if os.path.exists(DATASET_LETTER_FILE):
        data = np.load(DATASET_LETTER_FILE, allow_pickle=True)
        traj = data[0]
        xs, ys = traj[:, 0], traj[:, 1]
        print(f"Loaded {DATASET_LETTER_FILE}")
    else:
        print("Generating synthetic trajectory.")
        Ndemo = 300
        t_demo = np.linspace(MIN_T, 1.0, Ndemo)
        xs = np.sin(2*np.pi*t_demo) + 0.2*np.sin(10*np.pi*t_demo)
        ys = np.cos(2*np.pi*t_demo)
    
    N = xs.shape[0]
    t = np.linspace(MIN_T, 6.0, N)
    
    def compute_velocities_from_positions(t, pos):
        vd = np.zeros_like(pos)
        for d in range(pos.shape[1]): vd[:, d] = np.gradient(pos[:, d], t)
        return vd 
    
    P = np.column_stack([xs, ys])
    print("Pre-processing (Savitzky-Golay)...")
    P = smooth_trajectory_savgol(P, window_length=15, polyorder=3)
    xs, ys = P[:, 0], P[:, 1]
    
    V = compute_velocities_from_positions(t, P)
    V = smooth_trajectory_savgol(V, window_length=15, polyorder=3)
    vx_ref, vy_ref = V[:, 0], V[:, 1]
    vt_ref = np.nan_to_num(np.sqrt(vx_ref**2 + vy_ref**2))

    # --- STEP 1: Initialization ---
    candidate_durations = np.linspace(0.1, 1.3, 50) 
    print("STEP 1: Initial seed (Min-Jerk greedy selection)...")
    modes_init, sigmas_init, scales_v_list_init, fs_est_v_init, _ = fit_sparse_by_predicted_reduction(
        t, [vt_ref], 
        candidate_grid_step=0.035,
        candidate_sigmas=candidate_durations,
        top_sigmas_per_mode=2,
        max_bases=60,
        min_pred_abs=1e-5,
        prune_thresh=0.1,
        verbose=False
    )
    if modes_init.size == 0: raise SystemExit
    
    # Plot initial seed
    vt_fit_initial = fs_est_v_init[0]
    scales_vt_init = scales_v_list_init[0]
    rmse_vt_initial = np.sqrt(np.mean((vt_ref - vt_fit_initial)**2))
    plot_velocity_and_bases(t, vt_ref, vt_fit_initial, modes_init, sigmas_init, scales_vt_init, 
                            dim_color='tab:cyan', 
                            title=f'STEP 1b: Initial Min-Jerk Seed (RMSE={rmse_vt_initial:.4f})', plot_mode='individual')
    
    # --- STEP 3: Global Optimization ---
    modes_v_opt, sigmas_v_opt = refine_bases_for_position_error(
        t, P, modes_init, sigmas_init, 
        step_title="STEP 3: Global Optimization (Min-Jerk)", verbose=False
    )
    
    # --- STEP 4: Merge & Re-Optimize ---
    print(f"\nSTEP 4: Iterative Merge & Re-Optimization...")
    modes_curr, sigmas_curr = modes_v_opt, sigmas_v_opt
    
    # Calculo inicial escalas
    Phi_temp = build_phi_matrix(t, modes_curr, sigmas_curr)
    Phi_pos_temp = np.zeros_like(Phi_temp)
    for k in range(modes_curr.size): Phi_pos_temp[:, k] = cumtrapz(Phi_temp[:, k], t, initial=0.0)
    A_temp = np.hstack([Phi_pos_temp, np.ones((N, 1))])
    sx_curr = np.linalg.lstsq(A_temp, P[:, 0], rcond=None)[0][:-1]
    sy_curr = np.linalg.lstsq(A_temp, P[:, 1], rcond=None)[0][:-1]

    MAX_MERGE_CYCLES = 5
    cycle_count = 0
    while cycle_count < MAX_MERGE_CYCLES:
        cycle_count += 1
        modes_merged, sigmas_merged, sx_merged, sy_merged = merge_bases_by_proximity(
            modes_curr, sigmas_curr, sx_curr, sy_curr, mode_tol=0.02, sigma_tol=0.05 
        )
        if modes_merged.size == modes_curr.size:
            print(f"  [Cycle {cycle_count}] Convergence reached ({modes_merged.size} bases).")
            modes_v_reopt, sigmas_v_reopt = modes_curr, sigmas_curr
            break
        print(f"  [Cycle {cycle_count}] Merged {modes_curr.size} -> {modes_merged.size} bases. Re-optimizing...")
        
        modes_reopt, sigmas_reopt = refine_bases_for_position_error(
            t, P, modes_merged, sigmas_merged, step_title=f"    Cycle {cycle_count} Opt", verbose=False
        )
        
        # Update scales
        Phi_iter = build_phi_matrix(t, modes_reopt, sigmas_reopt)
        Phi_pos_iter = np.zeros_like(Phi_iter)
        for k in range(modes_reopt.size): Phi_pos_iter[:, k] = cumtrapz(Phi_iter[:, k], t, initial=0.0)
        A_iter = np.hstack([Phi_pos_iter, np.ones((N, 1))])
        sx_curr = np.linalg.lstsq(A_iter, P[:, 0], rcond=None)[0][:-1]
        sy_curr = np.linalg.lstsq(A_iter, P[:, 1], rcond=None)[0][:-1]
        modes_curr, sigmas_curr = modes_reopt, sigmas_reopt
        if cycle_count == MAX_MERGE_CYCLES: modes_v_reopt, sigmas_v_reopt = modes_curr, sigmas_curr

    # --- Final Reconstruction ---
    modes_v, sigmas_v = modes_v_reopt, sigmas_v_reopt
    K = modes_v.size
    Phi_vel_final = build_phi_matrix(t, modes_v, sigmas_v)
    Phi_pos_final = np.zeros_like(Phi_vel_final)
    for k in range(K): Phi_pos_final[:, k] = cumtrapz(Phi_vel_final[:, k], t, initial=0.0)

    A_final = np.hstack([Phi_pos_final, np.ones((N, 1))])
    sol_x = np.linalg.lstsq(A_final, P[:, 0], rcond=None)[0]
    scales_x, p0_x = sol_x[:-1], sol_x[-1]
    sol_y = np.linalg.lstsq(A_final, P[:, 1], rcond=None)[0]
    scales_y, p0_y = sol_y[:-1], sol_y[-1]
    
    P_fit = np.array([p0_x, p0_y])[None, :] + Phi_pos_final.dot(np.column_stack([scales_x, scales_y]))
    x_fit, y_fit = P_fit[:, 0], P_fit[:, 1]
    
    V_fit_from_P = compute_velocities_from_positions(t, P_fit)
    vx_fit_from_P, vy_fit_from_P = V_fit_from_P[:, 0], V_fit_from_P[:, 1]
    vt_fit_final = np.sqrt(vx_fit_from_P**2 + vy_fit_from_P**2)

    # --- STANDARD PLOTS ---
    print("\n--- Final Results ---")
    plot_2d_trajectory_pos(xs, ys, x_fit, y_fit)
    plot_position_and_integrated_bases(t, xs, x_fit, modes_v, Phi_pos_final, scales_x, 
                                    title='Position X (Min-Jerk Decomposition)')
    
    plot_position_and_integrated_bases(t, ys, y_fit, modes_v, Phi_pos_final, scales_y, 
                                    title='Position Y (Min-Jerk Decomposition)', pos_color='tab:green', neg_color='tab:purple')
    
    plot_velocity_and_bases_signed(t, vx_ref, vx_fit_from_P, modes_v, Phi_vel_final, scales_x, 
                                pos_color='tab:red', neg_color='tab:cyan',
                                title='Velocity X (Min-Jerk Decomposition)')

    plot_velocity_and_bases_signed(t, vy_ref, vy_fit_from_P, modes_v, Phi_vel_final, scales_y, 
                                pos_color='tab:green', neg_color='tab:purple',
                                title='Velocity Y (Min-Jerk Decomposition)')
    
    plot_velocity_comparison(t, vx_ref, vx_fit_from_P, 
                            title='Velocity X (Original vs. d(P_fit)/dt)')
    plot_velocity_comparison(t, vy_ref, vy_fit_from_P, 
                            title='Velocity Y (Original vs. d(P_fit)/dt)')
    
    # =========================================================================
    # === PLOTS ========
    # =========================================================================
    print("Exact decomposition...")

    # 1. Objective Velocity
    v_target_exact = vt_fit_final 

    # 2. Pure Min Jerk
    Phi = build_phi_matrix(t, modes_v, sigmas_v)
    raw_forces = np.zeros((K, len(t)))
    
    for k in range(K):
        
        mag_scale = np.sqrt(scales_x[k]**2 + scales_y[k]**2)
        raw_forces[k, :] = Phi[:, k] * mag_scale

    # 3. Normalization factor
    sum_raw_forces = np.sum(raw_forces, axis=0)
    ratio_t = np.zeros_like(v_target_exact)
    mask = sum_raw_forces > 1e-12
    ratio_t[mask] = v_target_exact[mask] / sum_raw_forces[mask]

    # 4. Final bases
    final_bases = raw_forces * ratio_t[None, :]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.nipy_spectral(np.linspace(0, 0.9, K))

    for k in range(K):
        curve = final_bases[k, :]
        if np.max(curve) > 0.005 * np.max(v_target_exact):
            c = "orange"
            plt.fill_between(t, 0, curve, color=c, alpha=0.3)
            plt.plot(t, curve, color=c, lw=1.5, label=f'Base {k}')
            idx = np.argmax(curve)
            plt.text(t[idx], curve[idx], f'{k}', fontsize=9, color=c, fontweight='bold', ha='center', va='bottom')

    plt.plot(t, v_target_exact, 'k--', lw=3, alpha=1.0, label='Speed Learned')
    plt.title("Basis Functions for `speed` (BaseExact(t)=BaseMinJerk(t)×(VReal(t))/∑Bases(t))")
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
    plt.show()
    
    print("Comparative generation: Avsolute Magnitud vs. Projected Contribution...")


    v_norm = vt_fit_final + 1e-12
    u_x = vx_fit_from_P / v_norm
    u_y = vy_fit_from_P / v_norm
    

    Phi = build_phi_matrix(t, modes_v, sigmas_v)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    

    ax1.plot(t, vt_ref, 'k-', lw=3, alpha=1.0, label='Reference Velocity')
    ax1.plot(t, vt_fit_final, 'r--', lw=2, label='Learned Velocity')
    
    for k in range(K):
        v_k_x = Phi[:, k] * scales_x[k]
        v_k_y = Phi[:, k] * scales_y[k]
        proj_k = v_k_x * u_x + v_k_y * u_y
        
        ax1.fill_between(t, 0, proj_k, where=(proj_k >= 0), interpolate=True, color='tab:orange', alpha=0.3)
        ax1.fill_between(t, 0, proj_k, where=(proj_k < 0), interpolate=True, color='tab:blue', alpha=0.3)
        
        idx_peak = np.argmax(np.abs(proj_k))
        if np.abs(proj_k[idx_peak]) > 0.05:
            peak_color = 'darkred' if proj_k[idx_peak] > 0 else 'darkblue'
            ax1.text(t[idx_peak], proj_k[idx_peak], f'{k}', fontsize=8, color=peak_color, ha='center', fontweight='bold')
        
        ax1.plot(t, np.ma.masked_where(proj_k < 0, proj_k), color='tab:orange', lw=1, alpha=0.6)
        ax1.plot(t, np.ma.masked_where(proj_k >= 0, proj_k), color='tab:blue', lw=1, alpha=0.6)

    ax1.set_title("A. Real contribution of basis functions\nOrange = V > 0 | Blue = V < 0", fontsize=11, fontweight='bold')
    ax1.set_ylabel('Velocity (m/s)'); ax1.grid(True, linestyle=':', alpha=0.5); ax1.axhline(0, color='k', lw=0.5)
    
    # Panel B: Magnitud Absoluta
    ax2.plot(t, vt_ref, 'k-', lw=2, alpha=1.0, label='Reference Velocity')
    sum_magnitudes = np.zeros_like(t)
    
    for k in range(K):
        mag_scale = np.sqrt(scales_x[k]**2 + scales_y[k]**2)
        mag_k = Phi[:, k] * mag_scale
        sum_magnitudes += mag_k
        
        ax2.fill_between(t, 0, mag_k, color='tab:green', alpha=0.2)
        ax2.plot(t, mag_k, color='tab:green', lw=1, alpha=0.8)
        
        peak_idx = np.argmax(mag_k)
        if mag_k[peak_idx] > 0.05 * np.max(vt_ref):
            ax2.text(t[peak_idx], mag_k[peak_idx], f'{k}', fontsize=8, color='darkgreen', ha='center', va='bottom')

    ax2.plot(t, sum_magnitudes, 'g:', lw=3, alpha=1.0, label='Scalar Sum (Sum of Magnitudes)')
    ax2.plot(t, vt_fit_final, 'r--', lw=2, label='Vector Sum (Norm of Sum)')
    
    ax2.plot(t, Phi.dot(scales_x), 'b', alpha=0.5, label='Vx Model')
    ax2.plot(t, Phi.dot(scales_y), 'orange', alpha=0.5, label='Vy Model')

    ax2.set_title("B. Speed Comparison: Scalar vs Vector Sum (MinJerk)", fontsize=11, fontweight='bold')
    ax2.set_ylabel('Velocity (m/s)'); ax2.set_xlabel('Time (s)')
    ax2.grid(True, linestyle=':', alpha=0.5); ax2.legend(loc='upper right', fontsize='small')
    plt.tight_layout(); plt.show()
    
    print("\nFinal Parameters:")
    print(f"{'Idx':<3} | {'Center (s)':<10} | {'Duration (s)':<12} | {'Scale X':<10} | {'Scale Y':<10}")
    for k in range(K):
        print(f"{k:<3} | {modes_v[k]:<10.4f} | {sigmas_v[k]:<12.4f} | {scales_x[k]:<10.4f} | {scales_y[k]:<10.4f}")