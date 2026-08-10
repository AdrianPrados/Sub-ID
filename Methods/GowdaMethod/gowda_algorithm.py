""" import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint

# =====================================================================
# 1. FUNCIONES MATEMÁTICAS CORE (GOWDA / JASON FRIEDMAN)
# =====================================================================

def MJxy(t0, D, Ax, Ay, t, calc_gradient=False):
    t = np.asarray(t)
    nt = (t - t0) / D
    nt_sq = nt**2
    nt_cubed = nt**3
    nt_fourth = nt**4

    shape = (-60 * nt_cubed + 30 * nt_fourth + 30 * nt_sq)
    Bx = (Ax / D) * shape
    By = (Ay / D) * shape

    A_tang = np.sqrt((Ax / D)**2 + (Ay / D)**2) + 1e-12 
    B = A_tang * shape

    if not calc_gradient:
        return Bx, By, B

    K = len(t)
    Jx = np.zeros((4, K))
    Jy = np.zeros((4, K))
    J = np.zeros((4, K))

    shape_dt0 = -(1.0 / D**2) * (120 * nt_cubed - 180 * nt_sq + 60 * nt)
    shape_dD = (1.0 / D**2) * (-150 * nt_fourth + 240 * nt_cubed - 90 * nt_sq)

    Jx[0, :] = Ax * shape_dt0
    Jx[1, :] = Ax * shape_dD
    Jx[2, :] = (1.0 / D) * shape

    Jy[0, :] = Ay * shape_dt0
    Jy[1, :] = Ay * shape_dD
    Jy[3, :] = (1.0 / D) * shape
    
    J[0, :] = A_tang * D * shape_dt0 
    J[1, :] = A_tang * D * shape_dD
    J[2, :] = (Ax / (A_tang * D**2)) * shape
    J[3, :] = (Ay / (A_tang * D**2)) * shape

    return Bx, By, B, Jx, Jy, J

def calc_error_metric(v_true, tangvel_true, v_pred):
    tangpred = np.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2)
    sumtrajsq = np.sum(v_true[:, 0]**2) + np.sum(v_true[:, 1]**2) + np.sum(tangvel_true**2)
    
    error_x = np.sum((v_pred[:, 0] - v_true[:, 0])**2)
    error_y = np.sum((v_pred[:, 1] - v_true[:, 1])**2)
    error_t = np.sum((tangpred - tangvel_true)**2)
    
    return (error_x + error_y + error_t) / (sumtrajsq + 1e-12)

def calculateerrorMJxy(parameters, times, vel, tangvel, timedelta):
    numsubmovements = len(parameters) // 4
    trajectoryx = vel[:, 0]
    trajectoryy = vel[:, 1]
    
    K = len(times)
    predictedx = np.zeros((numsubmovements, K))
    predictedy = np.zeros((numsubmovements, K))
    predicted = np.zeros((numsubmovements, K))
    
    Jx_total = np.zeros((4 * numsubmovements, K))
    Jy_total = np.zeros((4 * numsubmovements, K))
    J_total = np.zeros((4 * numsubmovements, K))
    
    for k in range(numsubmovements):
        T0 = parameters[k*4]
        D = parameters[k*4 + 1]
        Dx = parameters[k*4 + 2]
        Dy = parameters[k*4 + 3]
        
        mask = (times > T0) & (times < T0 + D)
        thisrng = np.where(mask)[0]
        
        if len(thisrng) > 0:
            t_eval = times[thisrng]
            Bx, By, B, Jx_part, Jy_part, J_part = MJxy(T0, D, Dx, Dy, t_eval, calc_gradient=True)
            
            predictedx[k, thisrng] = Bx
            predictedy[k, thisrng] = By
            predicted[k, thisrng] = B
            
            Jx_total[k*4:(k+1)*4, thisrng] = Jx_part
            Jy_total[k*4:(k+1)*4, thisrng] = Jy_part
            J_total[k*4:(k+1)*4, thisrng] = J_part
            
    sumpredictedx = np.sum(predictedx, axis=0)
    sumpredictedy = np.sum(predictedy, axis=0)
    sumpredicted = np.sum(predicted, axis=0)
    
    v_pred_matrix = np.column_stack([sumpredictedx, sumpredictedy])
    epsilon = calc_error_metric(vel, tangvel, v_pred_matrix)
    
    sumtrajsq = np.sum(trajectoryx**2) + np.sum(trajectoryy**2) + np.sum(tangvel**2) + 1e-12
    error_x = sumpredictedx - trajectoryx
    error_y = sumpredictedy - trajectoryy
    error_t = sumpredicted - tangvel
    
    grad = (2.0 / sumtrajsq) * (Jx_total.dot(error_x) + Jy_total.dot(error_y) + J_total.dot(error_t))
    return epsilon, grad, v_pred_matrix

def min_jerk_cost_fn(parameters, times, v, tv):
    N_PARAMS_PER_SUBMOVEMENT = 2
    n_movements = len(parameters) // N_PARAMS_PER_SUBMOVEMENT
    params_mat = parameters.reshape(n_movements, N_PARAMS_PER_SUBMOVEMENT)
    
    F = np.zeros((len(times), n_movements))
    for n in range(n_movements):
        t0 = params_mat[n, 0]
        D = params_mat[n, 1]
        mask = (times > t0) & (times < t0 + D)
        time_inds = np.where(mask)[0]
        
        if len(time_inds) > 0:
            Bx, By, B = MJxy(t0, D, 1, 1, times[time_inds], calc_gradient=False)
            F[time_inds, n] = Bx 
            
    ridge = np.eye(n_movements) * 0.5
    try:
        M = np.linalg.solve(F.T @ F + ridge, F.T @ v)
    except np.linalg.LinAlgError:
        M, _, _, _ = np.linalg.lstsq(F.T @ F + ridge, F.T @ v, rcond=None)
        
    full_params = np.zeros(4 * n_movements)
    for k in range(n_movements):
        full_params[k*4] = params_mat[k, 0]     
        full_params[k*4 + 1] = params_mat[k, 1] 
        full_params[k*4 + 2] = M[k, 0]          
        full_params[k*4 + 3] = M[k, 1]          
        
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    cost, full_grad, v_pred = calculateerrorMJxy(full_params, times, v, tv, dt)
    
    grad_inds = []
    for k in range(n_movements):
        grad_inds.extend([4*k, 4*k + 1])
        
    grad = full_grad[grad_inds]
    return cost, grad, v_pred, M

# =====================================================================
# 2. FUNCIONES DE MUESTREO Y SEGMENTACIÓN
# =====================================================================

def residual_sse(v, tv):

    e = v ** 2
    e_tang = tv.flatten() ** 2
    pt_by_pt_cost = np.sum(e, axis=1) + e_tang
    cost = np.sum(pt_by_pt_cost)
    return cost, pt_by_pt_cost

def segment_hand_vel(hand_vel, DT=0.01, still_threshold=0.005, min_ampl=0.02):

    if hand_vel.ndim > 1:
        speed = np.linalg.norm(hand_vel, axis=1)
    else:
        speed = np.abs(hand_vel)

    moving = speed > still_threshold
    moving_padded = np.concatenate(([False], moving, [False]))
    diff = np.diff(moving_padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        if np.max(speed[s:e]) >= min_ampl:
            segments.append([s, e])

    return np.array(segments, dtype=int)

def local_extrema(data, extrema_type='max'):

    lm = np.zeros(len(data), dtype=int)
    is_max = (extrema_type in ['max', 'both'])
    is_min = (extrema_type in ['min', 'both'])

    if len(data) < 2: return lm

    if (data[0] < data[1]) and is_min: lm[0] = 1
    elif (data[0] > data[1]) and is_max: lm[0] = 1

    for k in range(1, len(data) - 1):
        if data[k] > data[k-1] and data[k] > data[k+1] and is_max: lm[k] = 1
        elif data[k] < data[k-1] and data[k] < data[k+1] and is_min: lm[k] = 1

    if (data[-1] < data[-2]) and is_min: lm[-1] = 1
    elif (data[-1] > data[-2]) and is_max: lm[-1] = 1

    return lm

def greedy_onset_sampling(hand_vel, hand_speed, n_samples, DT, no_submovements, greed_factor=3, plot_dist=0):

    hand_speed = np.sqrt(np.sum(hand_vel**2, axis=1))
    cost, pt_by_pt_cost = residual_sse(hand_vel, hand_speed)
    
    pdist = pt_by_pt_cost / np.sum(pt_by_pt_cost)
    pdist = pdist ** greed_factor
    pdist = pdist / np.sum(pdist)
    
    local_mins = local_extrema(pdist, extrema_type='min')
    local_min_inds = np.where(local_mins)[0]
    
    if len(local_min_inds) < 2:
        t0_samples = np.random.uniform(0, len(hand_vel) * DT, n_samples)
        D_samples = np.random.uniform(0.1, 1.0, n_samples)
        return t0_samples, D_samples

    t0_sample_inds = np.random.randint(0, len(local_min_inds) - 1, size=n_samples)
    t0_samples = local_min_inds[t0_sample_inds]
    D_samples = local_min_inds[t0_sample_inds + 1] - t0_samples
    
    t0_samples = t0_samples * DT
    D_samples = D_samples * DT
    return t0_samples, D_samples

def reconstruct_submovements(params, T, DT, fn_type='min_jerk'):
    t = np.arange(T) * DT
    recon = np.zeros((T, 2)) 
    if params is None or len(params) == 0: return recon

    for p in params:
        t0, D = p[0], p[1]
        amplitudes = p[2:] 

        tau = (t - t0) / D
        mask = (tau >= 0) & (tau <= 1)
        base = np.zeros_like(t)

        if 'min_jerk' in fn_type:
            base[mask] = (1.0/D) * (30*tau[mask]**2 - 60*tau[mask]**3 + 30*tau[mask]**4)

        for dim, amp in enumerate(amplitudes):
            if dim < recon.shape[1]:
                recon[:, dim] += base * amp
    return recon

# =====================================================================
# 3. OPTIMIZADOR PRINCIPAL (decompose_lstsq)
# =====================================================================

def decompose_lstsq(times, vel, numsubmovements, **kwargs):
    defaults = {
        'prev_decomp': None, 'D_min': 0.09, 'D_max': 1.0, 'isi': 0.100,
        'term_cond': -np.inf, 'fn_type': 'min_jerk', 'sigma_min': 0.02,
        'sigma_max': 1.0, 'mu_min': -1.0, 'mu_max': 1.0,
        'add_new_submovement': 1, 'sample_randomly': 0,
    }
    defaults.update(kwargs)
    
    prev_decomp = defaults['prev_decomp']
    D_min, D_max = defaults['D_min'], defaults['D_max']
    isi = defaults['isi']
    term_cond = defaults['term_cond']
    fn_type = defaults['fn_type']
    sample_randomly = defaults['sample_randomly']
    add_new_submovement = defaults['add_new_submovement']

    min_jerk = (fn_type == 'min_jerk')
    min_jerk_full = (fn_type == 'min_jerk_full')

    if min_jerk: N_PARAMS_PER_SUBMOVEMENT = 2
    elif min_jerk_full: N_PARAMS_PER_SUBMOVEMENT = 4
    else: raise ValueError(f"Function type {fn_type} no soportada.")

    prev_decomp_size = prev_decomp.shape[0] if prev_decomp is not None else 0
    DT = times[1] - times[0]
    best_cost = np.inf

    t0_max = times[-1] - D_min
    vx_min, vx_max = np.min(vel[:, 0]), np.max(vel[:, 0])
    vy_min, vy_max = np.min(vel[:, 1]), np.max(vel[:, 1])
    T_max = len(times)
    
    lb, ub = [], []
    for k in range(numsubmovements):
        if min_jerk:
            lb.extend([isi * k, D_min])
            ub.extend([t0_max, min(D_max, T_max * DT)])
        elif min_jerk_full:
            lb.extend([isi * k, D_min, vx_min, vy_min])
            ub.extend([t0_max, min(D_max, T_max * DT), vx_max, vy_max])

    bounds = list(zip(lb, ub))
    constraints = []
    if isi > 0 and numsubmovements > 1:
        A = np.zeros((numsubmovements - 1, numsubmovements * N_PARAMS_PER_SUBMOVEMENT))
        b_vec = -isi * np.ones(numsubmovements - 1)
        for k in range(numsubmovements - 1):
            A[k, N_PARAMS_PER_SUBMOVEMENT * k] = 1
            A[k, N_PARAMS_PER_SUBMOVEMENT * (k + 1)] = -1
        constraints.append(LinearConstraint(A, -np.inf, b_vec))

    tv = np.sqrt(np.sum(vel**2, axis=1))
    count = 1
    n_iterations_total = 10
    
    if min_jerk or min_jerk_full: recon = reconstruct_submovements(prev_decomp, T_max, DT, 'min_jerk')
    else: recon = np.zeros_like(vel)

    res_vel = vel - recon
    res_speed = np.sqrt(np.sum(res_vel**2, axis=1))
    no_submovements = np.all(recon == 0) if recon.size > 0 else True

    if not sample_randomly:
        t0_samples, D_samples = greedy_onset_sampling(
            res_vel, res_speed, n_iterations_total, DT, no_submovements, greed_factor=prev_decomp_size + 1
        )
        t0_samples, inds = np.unique(t0_samples, return_index=True)
        D_samples = D_samples[inds]
        if len(t0_samples) < n_iterations_total:
            n_rand_samples = n_iterations_total - len(t0_samples)
            t0_samples = np.concatenate([t0_samples, np.random.uniform(0, t0_max, n_rand_samples)])
            D_samples = np.concatenate([D_samples, np.random.uniform(D_min, D_max, n_rand_samples)])

    n_opt_iterations = 0
    n_func_evals_total = 0
    bestresult = None
    bestfitresult = None

    while count <= n_iterations_total:
        if (numsubmovements == 1) and (count == 1):
            if min_jerk: initialparameters = np.array([0, np.random.uniform(D_min, D_max)])
            elif min_jerk_full: initialparameters = np.array([0, np.random.uniform(D_min, D_max), np.random.uniform(vx_min, vx_max), np.random.uniform(vy_min, vy_max)])
        elif sample_randomly:
            n_new_submovements = numsubmovements - prev_decomp_size
            new_submovement_init_params = []
            for _ in range(n_new_submovements):
                t0 = np.random.uniform(0, t0_max)
                D = np.random.uniform(D_min, D_max)
                if min_jerk: new_submovement_init_params.append([t0, D])
                elif min_jerk_full: new_submovement_init_params.append([t0, D, np.random.uniform(vx_min, vx_max), np.random.uniform(vy_min, vy_max)])
            new_submovement_init_params = np.array(new_submovement_init_params)
            initialparameters = np.vstack([prev_decomp[:, :N_PARAMS_PER_SUBMOVEMENT], new_submovement_init_params]) if prev_decomp_size > 0 else new_submovement_init_params
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()
        elif add_new_submovement:
            n_new_submovements = numsubmovements - prev_decomp_size
            new_submovement_init_params = []
            for _ in range(n_new_submovements):
                t0 = t0_samples[count - 1]
                D = D_samples[count - 1]
                if min_jerk: new_submovement_init_params.append([t0, D])
                elif min_jerk_full: new_submovement_init_params.append([t0, D, vx_max * D, vy_max * D])
            new_submovement_init_params = np.array(new_submovement_init_params)
            initialparameters = np.vstack([prev_decomp[:, :N_PARAMS_PER_SUBMOVEMENT], new_submovement_init_params]) if prev_decomp_size > 0 else new_submovement_init_params
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()
        else:
            initialparameters = prev_decomp[:, :N_PARAMS_PER_SUBMOVEMENT]
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()

        initialparameters = initialparameters.astype(float)

        if min_jerk:
            def obj_fun(params):
                c, g, _, _ = min_jerk_cost_fn(params, times, vel, tv)
                return c, g
            opt_res = minimize(obj_fun, initialparameters, method='SLSQP', jac=True, bounds=bounds, constraints=constraints, options={'maxiter': 5000})
            epsilon, _, fitresult, A_opt = min_jerk_cost_fn(opt_res.x, times, vel, tv)
            result_params = opt_res.x
        elif min_jerk_full:
            def obj_fun(params):
                c, g, _ = calculateerrorMJxy(params, times, vel, tv, DT)
                return c, g
            opt_res = minimize(obj_fun, initialparameters, method='SLSQP', jac=True, bounds=bounds, constraints=constraints, options={'maxiter': 5000})
            epsilon, _, fitresult = calculateerrorMJxy(opt_res.x, times, vel, tv, DT)
            result_params = opt_res.x

        n_opt_iterations += opt_res.nit
        n_func_evals_total += opt_res.nfev
        
        if epsilon < best_cost:
            best_cost = epsilon
            bestresult = result_params
            bestfitresult = fitresult
            if min_jerk: best_A_opt = A_opt
                
        if best_cost < term_cond: break
        else: count += 1

    if bestresult is not None:
        if min_jerk: bestresult = np.hstack([bestresult.reshape(-1, N_PARAMS_PER_SUBMOVEMENT), best_A_opt])
        elif min_jerk_full: bestresult = bestresult.reshape(-1, N_PARAMS_PER_SUBMOVEMENT)

    return best_cost, bestresult, bestfitresult, n_opt_iterations, n_func_evals_total

# =====================================================================
# 4. ORQUESTADOR PRINCIPAL (decompose_submovements_v2)
# =====================================================================

def decompose_submovements_v2(hand_kin, **kwargs):
    defaults = {
        'verbose': True, 'n_submovements': -1, 'method': 'scattershot',
        'prev_decomp': None, 'still_threshold': 0.005, 'min_ampl': 0.02,
        'bin_size_ms': 10.0, 'isi': 0.1, 'mse_perc_thresh': 0.02,
        'use_cost_diff': True, 'fn_type': 'min_jerk', 'vel_inds': [2, 3], 
        'D_min': 0.150, 'proc_idx': 1, 'n_procs': 1, 'zero_greed': False
    }
    defaults.update(kwargs)
    
    verbose = defaults['verbose']
    n_submovements = defaults['n_submovements']
    method = defaults['method']
    min_ampl = defaults['min_ampl']
    still_threshold = defaults['still_threshold']
    isi = defaults['isi']
    bin_size_ms = float(defaults['bin_size_ms'])
    mse_perc_thresh = defaults['mse_perc_thresh']
    use_cost_diff = defaults['use_cost_diff']
    fn_type = defaults['fn_type']
    vel_inds = defaults['vel_inds']
    D_min = defaults['D_min']
    proc_idx = defaults['proc_idx']
    n_procs = defaults['n_procs']
    zero_greed = defaults['zero_greed']

    if method not in ['scattershot', 'greedy']: raise ValueError(f"Method unknown: {method}")

    iterate_n_submovements = (n_submovements == -1)
    DT = bin_size_ms * 1e-3

    hand_vel = hand_kin[:, vel_inds]
    segments = segment_hand_vel(hand_vel, DT=DT, still_threshold=still_threshold, min_ampl=min_ampl)
    segments = np.array(segments)
    
    if len(segments) == 0:
        if verbose: print("No segments found.")
        return {}, np.zeros_like(hand_vel), [], [], {}, {}, 0.0, {}, {}

    minSegLength = math.floor(0.200 / DT)
    segLengths = segments[:, 1] - segments[:, 0]
    segments = segments[segLengths >= minSegLength]
    n_segments = segments.shape[0]
    
    segments_per_proc = math.ceil(n_segments / n_procs)
    split_indices = [list(range(i, min(i + segments_per_proc, n_segments))) for i in range(0, n_segments, segments_per_proc)]
    inds = split_indices[proc_idx - 1] if (proc_idx - 1) < len(split_indices) else []
    
    movements = {}
    submovement_recon = np.zeros_like(hand_vel)
    hand_speed = np.sqrt(np.sum(hand_vel**2, axis=1))

    start_time = time.time()
    n_iterations, n_func_evals, fits, fit_costs = {}, {}, {}, {}
    costs = np.zeros(n_segments)

    for k in inds:
        s = max(0, segments[k, 0] - 1) 
        e = segments[k, 1]
        
        hand_vel_tr = hand_vel[s:e, :]
        hand_speed_tr = hand_speed[s:e]
            
        if not np.isnan(hand_vel_tr).any():
            T = hand_vel_tr.shape[0]
            t = np.arange(T) * DT
            
            if method == 'scattershot':
                costs_tr, bestresult, bestfitresult, n_iterations_, n_func_evals_ = [], {}, {}, [], []
                prev_decomp = None
                
                seg_len = e - s
                max_submovements = max(math.ceil((seg_len * DT - D_min) / max(isi, 0.1)), 1)
                if not iterate_n_submovements: max_submovements = min(max_submovements, n_submovements)
                
                orig_error, _ = residual_sse(hand_vel_tr, hand_speed_tr)
                costs_so_far = np.ones(max_submovements) * np.inf
                
                submov_iter_idx = 0
                while (submov_iter_idx < 1) or ((submov_iter_idx < max_submovements) and not iterate_n_submovements) or ((costs_so_far[submov_iter_idx-1] > mse_perc_thresh) and (submov_iter_idx < max_submovements)):
                    submov_iter_idx += 1
                    m = submov_iter_idx - 1 
                    sample_randomly = 1 if zero_greed else 0
                    if zero_greed: prev_decomp = None
                        
                    c_tr, b_res, b_fit, n_iter, n_evals = decompose_lstsq(
                        t, hand_vel_tr, submov_iter_idx, prev_decomp=prev_decomp, term_cond=orig_error * mse_perc_thresh,
                        fn_type=fn_type, sample_randomly=sample_randomly, isi=isi
                    )
                    
                    costs_tr.append(c_tr)
                    bestresult[m] = b_res
                    bestfitresult[m] = b_fit
                    n_iterations_.append(n_iter)
                    n_func_evals_.append(n_evals)
                    
                    prev_decomp = b_res
                    costs_so_far[m] = c_tr / (orig_error + 1e-12)
                        
                    if (submov_iter_idx > 2) and iterate_n_submovements and ((costs_so_far[m] - costs_so_far[m-1]) > -0.001) and use_cost_diff:
                        break
                
                n_iterations[k] = n_iterations_
                n_func_evals[k] = n_func_evals_
                
                for ii in range(len(bestresult)): bestresult[ii][:, 0] += s * DT
                
                fits[k] = bestresult
                fit_costs[k] = costs_tr
                final_best_result = bestresult[m]
                
                if fn_type == 'min_jerk_full': best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, 'min_jerk')
                else: best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, fn_type)
                    
                costs[k] = costs_tr[m]
                submovement_recon[s:e, :] = best_recon_traj
                movements[k] = final_best_result

    runtime = time.time() - start_time
    return movements, submovement_recon, segments, costs, fits, fit_costs, runtime, n_iterations, n_func_evals
"""

import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint

def segment_hand_vel(hand_vel, DT=0.01, still_threshold=0.005, min_ampl=0.02):
    speed = np.linalg.norm(hand_vel, axis=1) if hand_vel.ndim > 1 else np.abs(hand_vel)
    moving = speed > still_threshold
    moving_padded = np.concatenate(([False], moving, [False]))
    diff = np.diff(moving_padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = []
    for s, e in zip(starts, ends):
        if np.max(speed[s:e]) >= min_ampl:
            segments.append([s, e])
    return np.array(segments, dtype=int)

def residual_sse(v, tv):
    e = v ** 2
    e_tang = tv.flatten() ** 2
    pt_by_pt_cost = np.sum(e, axis=1) + e_tang
    return np.sum(pt_by_pt_cost), pt_by_pt_cost

def local_extrema(data, extrema_type='max'):
    lm = np.zeros(len(data), dtype=int)
    is_max = (extrema_type in ['max', 'both'])
    is_min = (extrema_type in ['min', 'both'])
    if len(data) < 2: return lm
    if (data[0] < data[1]) and is_min: lm[0] = 1
    elif (data[0] > data[1]) and is_max: lm[0] = 1
    for k in range(1, len(data) - 1):
        if data[k] > data[k-1] and data[k] > data[k+1] and is_max: lm[k] = 1
        elif data[k] < data[k-1] and data[k] < data[k+1] and is_min: lm[k] = 1
    if (data[-1] < data[-2]) and is_min: lm[-1] = 1
    elif (data[-1] > data[-2]) and is_max: lm[-1] = 1
    return lm

def greedy_onset_sampling(hand_vel, hand_speed, n_samples, DT, no_submovements, greed_factor=3):
    hand_speed = np.sqrt(np.sum(hand_vel**2, axis=1))
    cost, pt_by_pt_cost = residual_sse(hand_vel, hand_speed)
    pdist = pt_by_pt_cost / np.sum(pt_by_pt_cost)
    pdist = pdist ** greed_factor
    pdist = pdist / np.sum(pdist)
    local_mins = local_extrema(pdist, extrema_type='min')
    local_min_inds = np.where(local_mins)[0]
    if len(local_min_inds) < 2:
        return np.random.uniform(0, len(hand_vel)*DT, n_samples), np.random.uniform(0.1, 1.0, n_samples)
    t0_sample_inds = np.random.randint(0, len(local_min_inds) - 1, size=n_samples)
    t0_samples = local_min_inds[t0_sample_inds] * DT
    D_samples = (local_min_inds[t0_sample_inds + 1] - local_min_inds[t0_sample_inds]) * DT
    return t0_samples, D_samples

def reconstruct_submovements(params, T, DT, fn_type='min_jerk', ndim=2):
    t = np.arange(T) * DT
    recon = np.zeros((T, ndim))
    if params is None or len(params) == 0: return recon
    for p in params:
        t0, D = p[0], p[1]
        amplitudes = p[2:]
        tau = (t - t0) / D
        mask = (tau >= 0) & (tau <= 1)
        base = np.zeros_like(t)
        if 'min_jerk' in fn_type:
            base[mask] = (1.0/D) * (30*tau[mask]**2 - 60*tau[mask]**3 + 30*tau[mask]**4)
        for dim, amp in enumerate(amplitudes):
            if dim < recon.shape[1]:
                recon[:, dim] += base * amp
    return recon

def min_jerk_cost_fn_ND(parameters, times, v, tv):
    """
    Versión generalizada N-Dimensional de la función de coste.
    Acepta cualquier dimensionalidad en 'v' (2D, 3D, etc).
    """
    n_movements = len(parameters) // 2
    params_mat = parameters.reshape(n_movements, 2)
    ndim = v.shape[1]

    F = np.zeros((len(times), n_movements))
    for n in range(n_movements):
        t0, D = params_mat[n, 0], params_mat[n, 1]
        mask = (times > t0) & (times < t0 + D)
        time_inds = np.where(mask)[0]
        if len(time_inds) > 0:
            nt = (times[time_inds] - t0) / D
            shape = (-60 * nt**3 + 30 * nt**4 + 30 * nt**2)
            F[time_inds, n] = (1.0 / D) * shape

    ridge = np.eye(n_movements) * 0.5
    try:
        M = np.linalg.solve(F.T @ F + ridge, F.T @ v)
    except np.linalg.LinAlgError:
        M, _, _, _ = np.linalg.lstsq(F.T @ F + ridge, F.T @ v, rcond=None)

    v_pred = F @ M
    tangpred = np.sqrt(np.sum(v_pred**2, axis=1))

    error_dims = np.sum((v_pred - v)**2)
    error_t = np.sum((tangpred - tv)**2)

    # [!] CORRECCIÓN: Devolvemos el SSE puro. El bucle orquestador se encarga de normalizar.
    cost = (error_dims + error_t) 
    
    return cost, v_pred, M

def decompose_lstsq(times, vel, numsubmovements, **kwargs):
    defaults = {'prev_decomp': None, 'D_min': 0.09, 'D_max': 1.0, 'isi': 0.100, 'term_cond': -np.inf, 'sample_randomly': 0, 'add_new_submovement': 1}
    defaults.update(kwargs)
    prev_decomp = defaults['prev_decomp']
    D_min, D_max = defaults['D_min'], defaults['D_max']
    isi, term_cond = defaults['isi'], defaults['term_cond']
    sample_randomly, add_new_submovement = defaults['sample_randomly'], defaults['add_new_submovement']

    ndim = vel.shape[1]
    prev_decomp_size = prev_decomp.shape[0] if prev_decomp is not None else 0
    DT = times[1] - times[0]
    best_cost = np.inf

    t0_max = times[-1] - D_min
    T_max = len(times)

    lb, ub = [], []
    for k in range(numsubmovements):
        lb.extend([isi * k, D_min])
        ub.extend([t0_max, min(D_max, T_max * DT)])
    bounds = list(zip(lb, ub))

    constraints = []
    if isi > 0 and numsubmovements > 1:
        A = np.zeros((numsubmovements - 1, numsubmovements * 2))
        b_vec = -isi * np.ones(numsubmovements - 1)
        for k in range(numsubmovements - 1):
            A[k, 2 * k] = 1
            A[k, 2 * (k + 1)] = -1
        constraints.append(LinearConstraint(A, -np.inf, b_vec))

    tv = np.sqrt(np.sum(vel**2, axis=1))
    count = 1
    n_iterations_total = 10

    recon = reconstruct_submovements(prev_decomp, T_max, DT, ndim=ndim)
    res_vel = vel - recon
    res_speed = np.sqrt(np.sum(res_vel**2, axis=1))
    no_submovements = np.all(recon == 0) if recon.size > 0 else True

    if not sample_randomly:
        t0_samples, D_samples = greedy_onset_sampling(res_vel, res_speed, n_iterations_total, DT, no_submovements, greed_factor=prev_decomp_size + 1)
        t0_samples, inds = np.unique(t0_samples, return_index=True)
        D_samples = D_samples[inds]
        if len(t0_samples) < n_iterations_total:
            n_rand_samples = n_iterations_total - len(t0_samples)
            t0_samples = np.concatenate([t0_samples, np.random.uniform(0, t0_max, n_rand_samples)])
            D_samples = np.concatenate([D_samples, np.random.uniform(D_min, D_max, n_rand_samples)])

    n_opt_iterations = 0
    n_func_evals_total = 0
    bestresult = None
    bestfitresult = None
    best_A_opt = None

    while count <= n_iterations_total:
        if (numsubmovements == 1) and (count == 1):
            initialparameters = np.array([0, np.random.uniform(D_min, D_max)])
        elif sample_randomly:
            n_new_submovements = numsubmovements - prev_decomp_size
            new_submovement_init_params = []
            for _ in range(n_new_submovements):
                new_submovement_init_params.append([np.random.uniform(0, t0_max), np.random.uniform(D_min, D_max)])
            new_submovement_init_params = np.array(new_submovement_init_params)
            initialparameters = np.vstack([prev_decomp[:, :2], new_submovement_init_params]) if prev_decomp_size > 0 else new_submovement_init_params
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()
        elif add_new_submovement:
            n_new_submovements = numsubmovements - prev_decomp_size
            new_submovement_init_params = []
            for _ in range(n_new_submovements):
                new_submovement_init_params.append([t0_samples[count - 1], D_samples[count - 1]])
            new_submovement_init_params = np.array(new_submovement_init_params)
            initialparameters = np.vstack([prev_decomp[:, :2], new_submovement_init_params]) if prev_decomp_size > 0 else new_submovement_init_params
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()
        else:
            initialparameters = prev_decomp[:, :2]
            initialparameters = initialparameters[initialparameters[:, 0].argsort()].flatten()

        def obj_fun(params):
            c, _, _ = min_jerk_cost_fn_ND(params, times, vel, tv)
            return c

        opt_res = minimize(obj_fun, initialparameters, method='SLSQP', jac=False, bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        epsilon, fitresult, A_opt = min_jerk_cost_fn_ND(opt_res.x, times, vel, tv)
        
        n_opt_iterations += opt_res.nit
        n_func_evals_total += opt_res.nfev
        
        if epsilon < best_cost:
            best_cost = epsilon
            bestresult = opt_res.x
            bestfitresult = fitresult
            best_A_opt = A_opt
                
        if best_cost < term_cond: break
        else: count += 1

    if bestresult is not None:
        bestresult = np.hstack([bestresult.reshape(-1, 2), best_A_opt])
    return best_cost, bestresult, bestfitresult, n_opt_iterations, n_func_evals_total

def decompose_submovements_v2(hand_kin, **kwargs):
    defaults = {'verbose': True, 'n_submovements': -1, 'method': 'scattershot', 'prev_decomp': None, 'still_threshold': 0.005, 'min_ampl': 0.02, 'bin_size_ms': 10.0, 'isi': 0.1, 'mse_perc_thresh': 0.02, 'use_cost_diff': True, 'vel_inds': [2, 3], 'D_min': 0.150, 'proc_idx': 1, 'n_procs': 1, 'zero_greed': False}
    defaults.update(kwargs)
    
    verbose, n_submovements, method = defaults['verbose'], defaults['n_submovements'], defaults['method']
    min_ampl, still_threshold, isi = defaults['min_ampl'], defaults['still_threshold'], defaults['isi']
    bin_size_ms, mse_perc_thresh, use_cost_diff = float(defaults['bin_size_ms']), defaults['mse_perc_thresh'], defaults['use_cost_diff']
    vel_inds, D_min, proc_idx, n_procs, zero_greed = defaults['vel_inds'], defaults['D_min'], defaults['proc_idx'], defaults['n_procs'], defaults['zero_greed']

    iterate_n_submovements = (n_submovements == -1)
    DT = bin_size_ms * 1e-3

    hand_vel = hand_kin[:, vel_inds]
    ndim = hand_vel.shape[1]
    
    segments = segment_hand_vel(hand_vel, DT=DT, still_threshold=still_threshold, min_ampl=min_ampl)
    if len(segments) == 0:
        return {}, np.zeros_like(hand_vel), [], [], {}, {}, 0.0, {}, {}

    minSegLength = math.floor(0.200 / DT)
    segLengths = segments[:, 1] - segments[:, 0]
    segments = segments[segLengths >= minSegLength]
    n_segments = segments.shape[0]
    
    segments_per_proc = math.ceil(n_segments / n_procs)
    split_indices = [list(range(i, min(i + segments_per_proc, n_segments))) for i in range(0, n_segments, segments_per_proc)]
    inds = split_indices[proc_idx - 1] if (proc_idx - 1) < len(split_indices) else []
    
    movements = {}
    submovement_recon = np.zeros_like(hand_vel)
    hand_speed = np.sqrt(np.sum(hand_vel**2, axis=1))

    start_time = time.time()
    n_iterations, n_func_evals, fits, fit_costs = {}, {}, {}, {}
    costs = np.zeros(n_segments)

    for k in inds:
        s = max(0, segments[k, 0] - 1) 
        e = segments[k, 1]
        hand_vel_tr = hand_vel[s:e, :]
        hand_speed_tr = hand_speed[s:e]
            
        if not np.isnan(hand_vel_tr).any():
            T = hand_vel_tr.shape[0]
            t = np.arange(T) * DT
            costs_tr, bestresult, bestfitresult, n_iterations_, n_func_evals_ = [], {}, {}, [], []
            prev_decomp = None
            
            seg_len = e - s
            max_submovements = max(math.ceil((seg_len * DT - D_min) / max(isi, 0.1)), 1)
            if not iterate_n_submovements: max_submovements = min(max_submovements, n_submovements)
            
            orig_error, _ = residual_sse(hand_vel_tr, hand_speed_tr)
            costs_so_far = np.ones(max_submovements) * np.inf
            
            submov_iter_idx = 0
            while (submov_iter_idx < 1) or ((submov_iter_idx < max_submovements) and not iterate_n_submovements) or ((costs_so_far[submov_iter_idx-1] > mse_perc_thresh) and (submov_iter_idx < max_submovements)):
                submov_iter_idx += 1
                m = submov_iter_idx - 1 
                sample_randomly = 1 if zero_greed else 0
                if zero_greed: prev_decomp = None
                    
                c_tr, b_res, b_fit, n_iter, n_evals = decompose_lstsq(
                    t, hand_vel_tr, submov_iter_idx, prev_decomp=prev_decomp, term_cond=orig_error * mse_perc_thresh,
                    sample_randomly=sample_randomly, isi=isi
                )
                
                costs_tr.append(c_tr)
                bestresult[m] = b_res
                bestfitresult[m] = b_fit
                n_iterations_.append(n_iter)
                n_func_evals_.append(n_evals)
                
                prev_decomp = b_res
                costs_so_far[m] = c_tr / (orig_error + 1e-12)
                    
                if (submov_iter_idx > 2) and iterate_n_submovements and ((costs_so_far[m] - costs_so_far[m-1]) > -0.001) and use_cost_diff:
                    break
            
            n_iterations[k] = n_iterations_
            n_func_evals[k] = n_func_evals_
            
            for ii in range(len(bestresult)): bestresult[ii][:, 0] += s * DT
            
            fits[k] = bestresult
            fit_costs[k] = costs_tr
            final_best_result = bestresult[m]
            
            best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, ndim=ndim)
            costs[k] = costs_tr[m]
            submovement_recon[s:e, :] = best_recon_traj
            movements[k] = final_best_result

    runtime = time.time() - start_time
    return movements, submovement_recon, segments, costs, fits, fit_costs, runtime, n_iterations, n_func_evals