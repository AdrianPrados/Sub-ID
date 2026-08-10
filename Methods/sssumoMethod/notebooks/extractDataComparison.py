#! Here I am assuming a position within the sssumo repository structure but the SSSUMO can not reconstruct that,
# it is just an estimation script to extract data for comparison with other methods.

import os
import numpy as np
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import time 

# --- Importaciones de sssumo ---
try:
    from sssumo.data import SyntheticDataset, OrganicDataset, CombinedSyntheticDataset, ConstantDistribution
    from sssumo.models import TDNNDetector, STEContinuousReconstructor, STEBinarizer
    from sssumo.utils import Config
except ImportError as e:
    print("Error: No se encuentra la librería 'sssumo'.")
    raise e

# --- Configuración Global y Rutas ---
root_dir = '/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/COMPARATIVAS/sssumo/'
OUTPUT_DIR = 'DatosComparacion'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

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

dataset2path = {
    'steering': os.path.join(datasets_dir, 'steering_tangential_velocity_data.csv'),
    'pushT': os.path.join(datasets_dir, 'pusht_converted_correction.csv'),
    # ... otros datasets ...
}

# --- Carga del Modelo ---
config = Config(nn_config_path, root_dir=root_dir)
config.start_with_weights = nn_checkpoint_path
device = config.device if torch.cuda.is_available() else 'cpu'

model = TDNNDetector(
        batchnorm=config.batchnorm,
        dilations=config.dilations,
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device, config.dtype)
model.eval()
model.load_state_dict(torch.load(config.start_with_weights, map_location=device))

basic_dataset = SyntheticDataset(**config.get_dataset_parameters())
reconstructor = basic_dataset.reconstruction_model.to(device)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor): return tensor.detach().cpu().numpy()
    return tensor

def cumtrapz_numpy(y, dx=1.0, initial=0.0):
    """Integración acumulativa simple para numpy arrays."""
    return np.concatenate([[initial], initial + np.cumsum(y)[:-1] * dx])

# --- Función Principal Modificada ---

def submovement_decomposition_on_organic(dataset_name='object_moving', sample_id=0, segment_time=(0, 900), snr=50):
    global _cached_dataset, _cached_dataset_name
    
    start_time = time.time() 

    # Carga de Dataset (Original de SSSUMO)
    organic_dataset_kwargs = {'snr_distribution': snr, 'noise_mode': 'gaussian', 'quadratic_mean': 1, 'low_pass_filter': np.inf, 'purpose': 'test'}
    
    # HACK: Para reconstruir posición, necesitamos los datos RAW (VX, VY) que SSSUMO normalmente oculta
    # Vamos a cargar el CSV manualmente para tener VX/VY originales
    raw_df = None
    if dataset_name in dataset2path:
        csv_path = dataset2path[dataset_name]
        if os.path.exists(csv_path):
            try:
                # Intentamos cargar columnas de velocidad si existen (depende del formato del CSV de sssumo)
                # SSSUMO suele guardar solo Vt en los CSV procesados. 
                # Si es el caso, NO PODEMOS recuperar la dirección real.
                # Asumiremos que el CSV tiene estructura estándar o cargaremos solo Vt.
                raw_df = pd.read_csv(csv_path)
            except: pass

    # Carga normal de SSSUMO
    data_path = dataset2path[dataset_name]
    dataset = OrganicDataset(data_path, **organic_dataset_kwargs)
    dataset.snr_distribution = ConstantDistribution(snr)

    # Obtener Vt ruidosa
    x_noisy, x_clean, _ = dataset[sample_id]
    
    # Validar Segmento
    start, end = segment_time
    T_total = x_noisy.shape[-1]
    if end > T_total: end = T_total
    
    # Slice
    x_input = x_noisy[:, start:end].unsqueeze(0).to(config.device)
    
    # Inferencia
    y_pred = model(x_input).detach()
    x_reconstructed, _ = reconstructor(y_pred) # Esto es Vt reconstruida
    
    end_time = time.time()
    exec_time = end_time - start_time

    # Parsear resultados
    mask_pred = y_pred[:, 0]
    auc_pred = y_pred[:, 1, :]
    duration_pred = y_pred[:, 2, :]
    binarized_mask_pred = reconstructor.binarizer.apply(mask_pred, True, True)
    
    pred_onsets = torch.where(binarized_mask_pred[0])[0].cpu().numpy()
    pred_durs = to_numpy(duration_pred[0])[pred_onsets]
    pred_amps = to_numpy(auc_pred[0])[pred_onsets]

    # --- RECONSTRUCCIÓN DE POSICIÓN (ESTIMACIÓN) ---
    vt_ref = to_numpy(x_input[0, 0])
    vt_fit = to_numpy(x_reconstructed[0, 0])
    
    # Intentamos recuperar la dirección original si es posible
    # Como SSSUMO OrganicDataset solo devuelve Vt, no tenemos VX/VY en 'x_noisy'.
    # Opción A: Si el CSV tiene vx, vy, usarlos.
    # Opción B (Fallback): Asumir movimiento 1D o dirección constante.
    
    # Para comparar justamente con PushT/MinJerk que usan 2D, necesitamos 2D.
    # Si 'pusht_converted_correction.csv' tiene vx/vy, úsalos.
    # Si solo tiene Vt, generaremos "Posición Ficticia" (integrando Vt) que será una línea recta.
    
    # Intento de cargar VX/VY del dataframe original alineado
    vx_ref_arr, vy_ref_arr = np.zeros_like(vt_ref), np.zeros_like(vt_ref)
    
    if raw_df is not None and 'vx' in raw_df.columns and 'vy' in raw_df.columns:
        # Asumiendo que sample_id corresponde a filas secuenciales o estructura conocida
        # Esto es complejo sin saber la estructura exacta del CSV de SSSUMO.
        # Por seguridad, usaremos la integración 1D (distancia recorrida).
        print("Aviso: No se pudo extraer dirección 2D exacta del dataset de SSSUMO.")
        print("Se guardará la trayectoria como distancia acumulada (1D) para referencia.")
        
        px_ref = cumtrapz_numpy(vt_ref)
        py_ref = np.zeros_like(px_ref)
        
        px_fit = cumtrapz_numpy(vt_fit)
        py_fit = np.zeros_like(px_fit)
        
    else:
        # Fallback 1D
        px_ref = cumtrapz_numpy(vt_ref)
        py_ref = np.zeros_like(px_ref)
        px_fit = cumtrapz_numpy(vt_fit)
        py_fit = np.zeros_like(px_fit)

    # --- GUARDAR DATOS ---
    save_path = os.path.join(OUTPUT_DIR, "decomposition_data_sssumo.npz")
    print(f"Saving SSSUMO data to: {save_path}")
    
    def calc_r2(y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    r2_val = calc_r2(vt_ref, vt_fit)
    rmse_val = np.sqrt(np.mean((vt_ref - vt_fit)**2))
    num_k = len(pred_onsets)
    
    np.savez(
        save_path,
        t=np.arange(len(vt_ref)),
        vx_ref=vx_ref_arr, vy_ref=vy_ref_arr, vt_ref=vt_ref,
        px_ref=px_ref, py_ref=py_ref,
        vx_fit=np.zeros_like(vt_fit), vy_fit=np.zeros_like(vt_fit), vt_fit=vt_fit,
        px_fit=px_fit, py_fit=py_fit,
        subm_starts=pred_onsets,
        subm_ends=pred_onsets + pred_durs,
        subm_scales_x=pred_amps,
        subm_scales_y=np.zeros_like(pred_amps),
        metrics=np.array([r2_val, rmse_val, num_k, exec_time])
    )
    print("Done.")

# --- Bloque Main ---
if __name__ == "__main__":
    print("\n=== SCRIPT DE INFERENCIA SSSUMO ===")
    target_dataset = 'pushT'
    if os.path.exists(dataset2path.get(target_dataset, "")):
        try:
            submovement_decomposition_on_organic(target_dataset, sample_id=0, segment_time=(0, 2000))
        except Exception as e: print(f"Error: {e}")
    else:
        print("Dataset no encontrado.")