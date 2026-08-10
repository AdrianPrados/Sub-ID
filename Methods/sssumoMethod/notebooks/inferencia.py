import os
import numpy as np
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, correlate
from scipy.signal import resample


# --- Importaciones de sssumo ---
try:
    from sssumo.data import SyntheticDataset, OrganicDataset, CombinedSyntheticDataset, ConstantDistribution
    from sssumo.models import TDNNDetector, STEContinuousReconstructor, STEBinarizer
    from sssumo.utils import Config
except ImportError as e:
    print("Error: No se encuentra la librería 'sssumo'.")
    print("Asegúrate de estar en el entorno correcto y de haber instalado el paquete (pip install .).")
    raise e

# --- Configuración Global y Rutas ---

# Ruta raíz especificada
root_dir = '/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/COMPARATIVAS/sssumoMethod/'

fine_tuned_model = True

nn_config_dir = os.path.join(root_dir, 'configs')
nn_checkpoint_dir = os.path.join(root_dir, 'checkpoints')
datasets_dir = os.path.join(root_dir, 'data')

if fine_tuned_model:
    nn_config_name = 'config-0425-tune_ModGauss_wo_writing.yaml'
    nn_checkpoint_number = 9
else:
    nn_config_name = 'config-0423-ModGaussian_ampl.yaml'
    nn_checkpoint_number = 24

nn_checkpoint_name = f'{nn_config_name.split(".")[0]}_{nn_checkpoint_number}.pth'
nn_config_path = os.path.join(root_dir, nn_config_dir, nn_config_name)
nn_checkpoint_path = os.path.join(root_dir, nn_checkpoint_dir, nn_checkpoint_name)

# Mapeo de datasets
dataset2path = {
    'steering': os.path.join(datasets_dir, 'steering_tangential_velocity_data.csv'),
    'crank': os.path.join(datasets_dir, 'crank_tangential_velocity_data.csv'),
    'Fitts': os.path.join(datasets_dir, 'Fitts_tangential_velocity_data.csv'),
    'whacamole': os.path.join(datasets_dir, 'whacamole_tangential_velocity_data.csv'),
    'object_moving': os.path.join(datasets_dir, 'object_moving_tangential_velocity_data.csv'),
    'pointing': os.path.join(datasets_dir, 'pointing_tangential_velocity_data.csv'),
    'tablet_writing': os.path.join(datasets_dir, 'tablet_writing_tangential_velocity_data.csv'),
    'pushT': os.path.join(datasets_dir, 'pusht_converted_correction.csv'),
}

# --- Carga del Modelo y Configuración ---

if not os.path.exists(nn_config_path):
    raise FileNotFoundError(f"Config no encontrada: {nn_config_path}")

config = Config(nn_config_path, root_dir=root_dir)
config.start_with_weights = nn_checkpoint_path

print(f"Cargando experimento: {config.experiment_name}")
device = config.device if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

model = TDNNDetector(
        batchnorm=config.batchnorm,
        dilations=config.dilations,
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device, config.dtype)

model.eval()

if os.path.exists(config.start_with_weights):
    print(f"Cargando pesos desde: {config.start_with_weights}")
    model.load_state_dict(torch.load(config.start_with_weights, map_location=device))
else:
    print(f"ADVERTENCIA: Pesos no encontrados en {config.start_with_weights}")

# Inicializar dataset base para obtener el reconstructor y binarizer
basic_dataset = SyntheticDataset(**config.get_dataset_parameters())
reconstructor = basic_dataset.reconstruction_model.to(device)


# --- Funciones Auxiliares y de Plotting ---
""" def decompose_numpy_velocity(velocity_array, sr=None):
    if model is None:
        print("Error: Modelo SSSUMO no cargado.")
        return None, None, None, None, None

    # --- 1. PREPARACIÓN DE DATOS (Upsampling a 60Hz) ---
    original_len = len(velocity_array)
    TARGET_FREQ = 60.0
    
    # Asumimos una frecuencia base si no se da (ej. 10Hz para PushT, o calculada fuera)
    if sr is None: 
        sr = 10.0 # Default fallback, aunque debería pasarse el correcto
    
    duration = original_len / sr
    
    # Eje de tiempo original y objetivo
    t_orig = np.linspace(0, duration, original_len)
    N_target = int(duration * TARGET_FREQ)
    t_target = np.linspace(0, duration, N_target)
    
    # Interpolación de entrada (Original -> 60Hz)
    if N_target != original_len:
        cs_in = CubicSpline(t_orig, velocity_array)
        vel_input = cs_in(t_target)
    else:
        vel_input = velocity_array

    # --- 2. NORMALIZACIÓN ---
    std_val = np.std(vel_input)
    if std_val < 1e-6: std_val = 1.0
    vel_normalized = vel_input / std_val
    
    # Tensor
    x_input = torch.tensor(vel_normalized, dtype=config.dtype).unsqueeze(0).unsqueeze(0).to(device)
    
    # --- 3. INFERENCIA ---
    with torch.no_grad():
        y_pred = model(x_input)
        x_reconstructed, _ = reconstructor(y_pred)
    
    # --- 4. EXTRACCIÓN DE PARÁMETROS ---
    mask_pred = y_pred[:, 0]
    auc_pred = y_pred[:, 1, :]
    duration_pred = y_pred[:, 2, :]
    
    binarized_mask_pred = reconstructor.binarizer.apply(mask_pred, True, True)
    onsets_idx_60hz = torch.where(binarized_mask_pred[0])[0].cpu().numpy()
    
    # Parámetros en dominio 60Hz
    amps = to_numpy(auc_pred[0])[onsets_idx_60hz] * std_val # Des-normalizar amplitud
    durs_steps = to_numpy(duration_pred[0])[onsets_idx_60hz]
    
    # Convertir tiempos a segundos reales
    t_starts = onsets_idx_60hz / TARGET_FREQ
    t_durations = durs_steps / TARGET_FREQ
    t_ends = t_starts + t_durations
    
    # --- 5. RECONSTRUCCIÓN DE SEÑAL ALINEADA (60Hz -> Original) ---
    vt_recon_60hz = to_numpy(x_reconstructed[0, 0]) * std_val
    
    # Interpolación de salida (60Hz -> Original)
    if len(vt_recon_60hz) != original_len:
        cs_out = CubicSpline(t_target, vt_recon_60hz)
        vt_recon_aligned = cs_out(t_orig)
    else:
        vt_recon_aligned = vt_recon_60hz

    # Asegurar longitudes exactas por errores de redondeo en linspace
    if len(vt_recon_aligned) != original_len:
        vt_recon_aligned = np.interp(np.arange(original_len), np.linspace(0, original_len, len(vt_recon_aligned)), vt_recon_aligned)

    return vt_recon_aligned, t_starts, t_ends, amps, t_durations """
    
def calcular_drift_vs_original(vel_original, fps_original, vel_reconstruida_60hz, max_lag_sec=0.5):
    """
    Calcula el desplazamiento temporal (drift) comparando la señal original (raw)
    directamente con la reconstruida (60Hz), sin resamplear la original primero.
    
    Args:
        vel_original (np.array): La señal de entrada cruda.
        fps_original (float): FPS de la señal cruda (ej. 10.0, 100.0).
        vel_reconstruida_60hz (np.array): La salida del modelo (ya a 60Hz).
        max_lag_sec (float): Rango de búsqueda en segundos (ej. +/- 0.5s).
        
    Returns:
        float: El shift en segundos que se debe SUMAR al tiempo de la reconstrucción 
               para alinearla con la original.
    """
    # 1. Crear ejes de tiempo reales para cada señal
    n_orig = len(vel_original)
    n_recon = len(vel_reconstruida_60hz)
    
    t_orig = np.arange(n_orig) / fps_original
    t_recon = np.arange(n_recon) / 60.0 # Asumimos modelo a 60Hz
    
    # 2. Normalizar ambas señales (RMS) para que el MSE compare FORMA y no amplitud
    # Esto es crítico porque vel_original puede tener escala 100 y la recon escala 1.
    rms_orig = np.sqrt(np.mean(vel_original**2)) or 1.0
    rms_recon = np.sqrt(np.mean(vel_reconstruida_60hz**2)) or 1.0
    
    sig_orig_norm = vel_original / rms_orig
    sig_recon_norm = vel_reconstruida_60hz / rms_recon

    # 3. Búsqueda de Lag (Grid Search)
    # Probamos shifts en pasos de ~1ms o basados en el frame rate más alto (60Hz -> ~0.016s)
    # Usaremos una resolución fina (ej. 0.01s) para precisión
    resolution = 0.01 
    shifts_to_test = np.arange(-max_lag_sec, max_lag_sec, resolution)
    
    best_shift = 0.0
    min_mse = float('inf')
    
    # Para ser eficientes, usaremos interpolación lineal (np.interp)
    # Queremos ver qué shift 's' hace que: Recon(t + s) ≈ Original(t)
    
    for shift in shifts_to_test:
        # Definimos los tiempos donde queremos evaluar la reconstrucción
        # Si Recon(t_recon) debe alinearse a Orig(t_orig), 
        # estamos buscando Recon en los tiempos t_orig - shift.
        # Ejemplo: Si shift es +1s (retraso), lo que pasó en t=0 del original
        # está en t=1 de la recon. Debemos buscar en Recon[1.0].
        # t_query = 0 - (-1) ... espera.
        # Simplificación: t_aligned = t_recon + shift. 
        # Queremos interpolar la Recon sobre la grilla de tiempo Original.
        
        # Mapeamos los tiempos originales al espacio de tiempo de la reconstrucción
        # aplicando el shift inverso para buscar el valor.
        t_query = t_orig - shift
        
        # Filtramos para comparar solo donde hay solapamiento temporal válido
        valid_mask = (t_query >= t_recon[0]) & (t_query <= t_recon[-1])
        
        # Necesitamos un solapamiento mínimo (ej. 20% de la señal)
        if np.sum(valid_mask) < (n_orig * 0.2):
            continue

        # Interpolamos la reconstrucción (60Hz) en los puntos de tiempo de la original
        recon_interpolada = np.interp(t_query[valid_mask], t_recon, sig_recon_norm)
        orig_subset = sig_orig_norm[valid_mask]
        
        # Calcular MSE
        mse = np.mean((orig_subset - recon_interpolada)**2)
        
        if mse < min_mse:
            min_mse = mse
            best_shift = shift

    return best_shift
    
def decompose_numpy_velocity(velocity_array, original_fps=100.0):
    
    # Asegúrate de que estas variables globales estén disponibles o pásalas como argumentos
    # model, reconstructor, device, config
    
    TARGET_FREQ = 60.0
    N_orig = len(velocity_array)
    
    # --- 1. CÁLCULO EXACTO DE MUESTRAS ---
    duration = N_orig / original_fps
    # Calculamos el número exacto de muestras que DEBE tener a 60Hz
    n_new = int(round(duration * TARGET_FREQ))
    
    # Vector de tiempo patrón (Este es tu "reloj" maestro a 60Hz)
    t_new = np.arange(n_new) / TARGET_FREQ
    
    # --- 2. RESAMPLING DE ENTRADA ---
    if abs(original_fps - TARGET_FREQ) > 1.0:
        # Forzamos que la entrada tenga exactamente n_new muestras
        vel_60hz = resample(velocity_array, n_new)
    else:
        # Si ya es similar, cortamos o rellenamos para que coincida exactamente con n_new
        if len(velocity_array) != n_new:
            vel_60hz = resample(velocity_array, n_new)
        else:
            vel_60hz = velocity_array

    # --- 3. NORMALIZACIÓN ---
    rms = np.sqrt(np.mean(vel_60hz**2))
    if rms < 1e-6: rms = 1.0
    vel_normalized = vel_60hz / rms
    
    # --- 4. INFERENCIA ---
    # Convertimos a Tensor
    # Nota: Asegúrate de que 'device' esté definido en el scope
    x_input = torch.tensor(vel_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_pred = model(x_input)
        x_reconstructed, _ = reconstructor(y_pred)
    
    # Salida cruda de la red
    vt_recon_raw = x_reconstructed[0, 0].cpu().numpy()

    # --- 5. CANDADO DE SEGURIDAD (LA GARANTÍA DEL 100%) ---
    # Verificamos si la red cambió el tamaño (común en CNNs con tamaños impares)
    if len(vt_recon_raw) != n_new:
        # Si difieren, interpolamos la salida para que coincida EXACTAMENTE con t_new
        # Usamos interpolación lineal para ajustar esa pequeña diferencia de muestras
        t_current_output = np.linspace(0, duration, len(vt_recon_raw))
        vt_recon_raw = np.interp(t_new, t_current_output, vt_recon_raw)

    # Reconstrucción final des-normalizada
    vt_recon_final = vt_recon_raw * rms

    # --- 6. EXTRACCIÓN DE PRIMITIVAS ---
    mask_pred = y_pred[:, 0]
    duration_pred = y_pred[:, 2, :]
    auc_pred = y_pred[:, 1, :] # Asumiendo que existe auc_pred
    
    auc_duration_pred = torch.stack([auc_pred, duration_pred], dim=1)
    pred_primitives_tensor = reconstructor.primitive(auc_duration_pred)
    
    bin_mask = reconstructor.binarizer.apply(mask_pred, True, True)
    onsets_idx_60hz = torch.where(bin_mask[0])[0].cpu().numpy()
    
    extracted_primitives = []
    
    for idx in onsets_idx_60hz:
        dur_val = duration_pred[0, idx].item()
        if dur_val < 1: continue
        
        end_idx = math.ceil(dur_val)
        # Protección de índices por si la primitiva se sale del tensor
        if end_idx > pred_primitives_tensor.shape[2]:
            end_idx = pred_primitives_tensor.shape[2]
            
        raw_curve = pred_primitives_tensor[0, idx, :end_idx].cpu().numpy()
        real_curve = raw_curve * rms* 1.2 
        
        # El tiempo de inicio está bloqueado a la rejilla de 60Hz
        t_start = idx / TARGET_FREQ
        
        # El eje de tiempo de la primitiva se construye a 60Hz
        t_curve = (np.arange(len(real_curve)) / TARGET_FREQ) + t_start
        
        extracted_primitives.append((t_curve, real_curve))

    # Ahora sí: len(t_new) == len(vt_recon_final) y el paso es exactamente 1/60
    return (t_new, vt_recon_final), extracted_primitives




def to_numpy(tensor):
    """Convierte tensor a numpy array de forma segura."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def plot_submovement_decomposition_matplotlib(x_noisy, x_clean, x_reconstructed, y_true, y_pred, binarized_mask_pred, true_primitives, pred_primitives, sr=60, include=(True, True, True, True, True)):
    """
    Visualiza la descomposición utilizando Matplotlib.
    """
    show_signals, show_submovs, show_mask, show_ampl, show_dur = include

    if not any(include):
        print("No plots requested.")
        return

    num_rows = sum(include)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows), sharex=True, dpi=100)
    if num_rows == 1:
        axes = [axes]

    current_ax_idx = 0
    T = x_noisy.shape[-1]
    time_axis = np.arange(T) / sr

    # 1. Signals
    if show_signals:
        ax = axes[current_ax_idx]
        ax.plot(time_axis, to_numpy(x_noisy[0, 0]), label="Noisy Input", color='gray', alpha=0.5, linewidth=1)
        ax.plot(time_axis, to_numpy(x_clean[0, 0]), label="Clean Target", color='blue', linewidth=1.5, alpha=0.7)
        ax.plot(time_axis, to_numpy(x_reconstructed[0, 0]), label="Reconstructed", color='red', linestyle='--', linewidth=2)
        ax.set_title("Noisy Signal, Clean Ground-Truth, and Reconstructed Signals")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        current_ax_idx += 1

    # 2. Submovements
    if show_submovs:
        ax = axes[current_ax_idx]
        
        # True Submovements (Solo si existen primitivas reales)
        if true_primitives is not None:
            true_submovement_onsets = torch.where(y_true[0, 0])[0]
            for i, submovement_onset in enumerate(true_submovement_onsets):
                submovement_onset = submovement_onset.item()
                duration_value = y_true[0, 2, submovement_onset].item()
                # Safety check for duration
                if duration_value > 0:
                    primitive_slice = true_primitives[0, submovement_onset][:math.ceil(duration_value)]
                    velocity_profile = torch.cat([torch.zeros(1).to(primitive_slice.device), primitive_slice])
                    velocity_profile_time_points = (torch.arange(0, len(velocity_profile), 1) + submovement_onset) / sr
                    ax.plot(velocity_profile_time_points.cpu().numpy(), velocity_profile.detach().cpu().numpy(), color='blue', alpha=0.5, label="True Submovement" if i == 0 else "")

        # Predicted Submovements
        predicted_submovement_onsets = torch.where(binarized_mask_pred[0])[0]
        duration_pred_tensor = y_pred[:, 2, :]
        
        for i, submovement_onset in enumerate(predicted_submovement_onsets):
            submovement_onset = submovement_onset.item()
            duration_value = duration_pred_tensor[0, submovement_onset].item()
            
            if pred_primitives is not None and duration_value > 0:
                primitive_slice = pred_primitives[0, submovement_onset][:math.ceil(duration_value)]
                velocity_profile = torch.cat([torch.zeros(1).to(primitive_slice.device), primitive_slice])
                velocity_profile_time_points = (torch.arange(0, len(velocity_profile), 1) + submovement_onset - 0.5) / sr
                ax.plot(velocity_profile_time_points.cpu().numpy(), velocity_profile.detach().cpu().numpy(), color='red', linestyle='--', label="Pred Submovement" if i == 0 else "")

        ax.set_title("Ground-truth and Reconstructed Submovements Decomposition")
        # Solo mostrar leyenda si se graficó algo
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        current_ax_idx += 1

    # Prepare indices for marker plots
    if show_mask or show_ampl or show_dur:
        true_mask_np = to_numpy(y_true[0, 0])
        true_indices = np.where(true_mask_np == 1)[0]
        pred_mask_np = to_numpy(binarized_mask_pred[0])
        pred_indices = np.where(pred_mask_np == 1)[0]

    # 3. Mask
    if show_mask:
        ax = axes[current_ax_idx]
        ax.plot(time_axis, to_numpy(y_pred[0, 0]), color='red', label="Predicted Onset Probability")
        ax.scatter(time_axis[true_indices], to_numpy(y_true[0, 0])[true_indices], color='blue', marker='o', s=50, label="True Onsets")
        ax.scatter(time_axis[pred_indices], to_numpy(y_pred[0, 0])[pred_indices], color='red', marker='o', facecolors='none', s=50, label="Predicted Onsets")
        ax.set_title("Ground-Truth Onsets and Onset Detection Signal")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        current_ax_idx += 1

    # 4. Amplitude
    if show_ampl:
        ax = axes[current_ax_idx]
        ax.plot(time_axis, to_numpy(y_pred[0, 1]), color='red', label="Predicted Amplitude Signal")
        ax.scatter(time_axis[true_indices], to_numpy(y_true[0, 1])[true_indices], color='blue', marker='o', s=50, label="True Amplitude")
        ax.scatter(time_axis[pred_indices], to_numpy(y_pred[0, 1])[pred_indices], color='red', marker='o', facecolors='none', s=50, label="Predicted Amplitude")
        ax.set_title("Ground-Truth and Predicted Amplitude")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        current_ax_idx += 1

    # 5. Duration
    if show_dur:
        ax = axes[current_ax_idx]
        ax.plot(time_axis, to_numpy(y_pred[0, 2]), color='red', label="Predicted Duration Signal")
        ax.scatter(time_axis[true_indices], to_numpy(y_true[0, 2])[true_indices], color='blue', marker='o', s=50, label="True Duration")
        ax.scatter(time_axis[pred_indices], to_numpy(y_pred[0, 2])[pred_indices], color='red', marker='o', facecolors='none', s=50, label="Predicted Duration")
        ax.set_title("Ground-Truth and Predicted Duration")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        current_ax_idx += 1

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# --- Funciones de Inferencia ---

# Variables globales de caché para datos orgánicos
_cached_dataset = None
_cached_dataset_name = None

def submovement_decomposition_on_organic(dataset_name='object_moving', sample_id=0, segment_time=(0, 900), snr=50):
    global _cached_dataset, _cached_dataset_name

    # Ensure dataset params are available
    organic_dataset_kwargs = {'snr_distribution': snr, 'noise_mode': 'gaussian', 'quadratic_mean': 1, 'low_pass_filter': np.inf, 'purpose': 'test'}

    # Load Dataset only if needed
    if _cached_dataset_name != dataset_name or _cached_dataset is None:
        print(f"Loading dataset: {dataset_name}...")
        if dataset_name not in dataset2path:
            print(f"Dataset {dataset_name} no encontrado en las rutas definidas.")
            return
        data_path = dataset2path[dataset_name]
        if not os.path.exists(data_path):
            print(f"Archivo no existe: {data_path}")
            return
        _cached_dataset = OrganicDataset(data_path, **organic_dataset_kwargs)
        _cached_dataset_name = dataset_name

    # Update SNR on the cached dataset
    if _cached_dataset is not None:
        _cached_dataset.snr_distribution = ConstantDistribution(snr)

    dataset = _cached_dataset

    # Validate Sample ID
    if sample_id < 0 or sample_id >= len(dataset):
        print(f"Sample ID {sample_id} is out of range (0-{len(dataset)-1}).")
        return

    # Get Data
    x_noisy, x_clean, _ = dataset[sample_id]

    # Validate Segment
    start, end = segment_time
    T_total = x_noisy.shape[-1]
    if start >= end or start < 0 or end > T_total:
        segment_length = end - start
        end_new = min(end, T_total)
        start_new = max(0, end_new - segment_length)
        print(f"Invalid segment {segment_time} for signal length {T_total}.")
        print(f"Setting segment to {start_new, end_new}.")
        start, end = start_new, end_new


    # Slice and Batch
    x_noisy = x_noisy[:, start:end].unsqueeze(0).to(config.device)
    x_clean = x_clean[:, start:end].unsqueeze(0).to(config.device)

    # Model Inference
    y_pred = model(x_noisy).detach()
    x_reconstructed, _ = reconstructor(y_pred)

    # Parse Predictions
    mask_pred = y_pred[:, 0]
    auc_pred = y_pred[:, 1, :]
    duration_pred = y_pred[:, 2, :]
    auc_duration_pred = torch.stack([auc_pred, duration_pred], dim=1)

    binarized_mask_pred = reconstructor.binarizer.apply(mask_pred, True, True)
    pred_primitives = reconstructor.primitive(auc_duration_pred)

    # Create Dummy Ground Truth (as organic data has no ground truth submovements)
    # This allows reusing the plotting function without errors.
    T_segment = x_noisy.shape[-1]
    y_true_dummy = torch.zeros((1, 3, T_segment)).to(y_pred.device)

    print(f"Visualizando descomposición orgánica para: {dataset_name} (Sample {sample_id})")
    
    # Visualization
    plot_submovement_decomposition_matplotlib(
        x_noisy,
        x_clean,
        x_reconstructed,
        y_true_dummy, # Pasamos el dummy
        y_pred,
        binarized_mask_pred,
        true_primitives=None, # No hay primitivas reales
        pred_primitives=pred_primitives,
        sr=60,
        include=(True, True, False, False, False) # Configuración solicitada
    )

def submovement_decomposition_on_synthetic(total_duration=600, seed=42, snr=50):
    """
    Versión para datos sintéticos utilizando la misma función de plot.
    """
    config.num_samples = 1
    config.batch_size = 1
    config.total_duration_distribution = total_duration
    config.seed = seed
    config.snr_distribution = snr
    
    # Dataset temporal para generar una muestra
    dataset = SyntheticDataset(**config.get_dataset_parameters())
    x_noisy, x_clean, y_true = dataset[0]
    
    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)
    y_true = y_true.unsqueeze(0).to(device)

    # Inferencia
    y_pred = model(x_noisy).detach()
    x_reconstructed, _ = reconstructor(y_pred)

    mask_pred = y_pred[:, 0]
    auc_pred = y_pred[:, 1, :]
    duration_pred = y_pred[:, 2, :]
    auc_duration_pred = torch.stack([auc_pred, duration_pred], dim=1)
    
    binarized_mask_pred = reconstructor.binarizer.apply(mask_pred, True, True)
    
    # Obtener primitivas (reales y predichas)
    pred_primitives = reconstructor.primitive(auc_duration_pred)
    
    auc_true = y_true[:, 1, :]
    duration_true = y_true[:, 2, :]
    auc_duration_true = torch.stack([auc_true, duration_true], dim=1)
    true_primitives = reconstructor.primitive(auc_duration_true)

    print(f"Visualizando descomposición SINTÉTICA (Seed {seed})")

    plot_submovement_decomposition_matplotlib(
        x_noisy,
        x_clean,
        x_reconstructed,
        y_true,
        y_pred,
        binarized_mask_pred,
        true_primitives,
        pred_primitives,
        sr=60,
        include=(True, True, True, True, True) # Mostramos todo en sintético
    )

# --- Bloque Main ---

if __name__ == "__main__":
    print("\n=== SCRIPT DE INFERENCIA SSSUMO ===")
    
    # 1. Ejecutar prueba con datos Sintéticos
    # Esto validará que el modelo y el ploteo completo funcionen
    """ try:
        submovement_decomposition_on_synthetic(total_duration=600, seed=42)
    except Exception as e:
        print(f"Error en sintético: {e}") """

    # 2. Ejecutar prueba con datos Orgánicos
    # Cambia 'object_moving' por el dataset que tengas descargado
    target_dataset = 'pushT'
    
    if os.path.exists(dataset2path.get(target_dataset, "")):
        try:
            submovement_decomposition_on_organic(
                dataset_name=target_dataset,
                sample_id=0,
                segment_time=(0, 2000)
            )
        except Exception as e:
            print(f"Error en orgánico: {e}")
    else:
        print(f"\nSaltando inferencia orgánica: Dataset '{target_dataset}' no encontrado.")