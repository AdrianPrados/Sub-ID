import os
import zarr
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# --- CONFIGURACIÓN ---
INPUT_ZARR_PATH = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/COMPARATIVAS/sssumo/notebooks/pusht_real/real_pusht_20230105/replay_buffer.zarr"
OUTPUT_CSV_NAME = "pusht_converted_correction.csv" # Nombre nuevo

# Frecuencias
ORIGINAL_FREQ = 10.0
TARGET_FREQ = 60.0 

def convert_pusht_force_test():
    print(f"--- Procesando para TEST PURO: {INPUT_ZARR_PATH} ---")
    
    if not os.path.exists(INPUT_ZARR_PATH):
        print("❌ ERROR: No encuentro el archivo Zarr")
        return

    root = zarr.open(INPUT_ZARR_PATH, mode='r')
    
    if 'data/robot_eef_pose' in root: pose_array = root['data/robot_eef_pose']
    elif 'data/state' in root: pose_array = root['data/state']
    else: return

    episode_ends = root['meta/episode_ends'][:]
    data_values = pose_array[:] 
    
    data_rows = []
    start_idx = 3
    
    for trial_idx, end_idx in enumerate(episode_ends):
        if start_idx >= len(data_values): break
        
        # 1. Extraer y Suavizar (SavGol) - IGUAL QUE TU SCRIPT
        raw_pos = data_values[start_idx:end_idx, :2]
        N = len(raw_pos)
        t_orig = np.arange(N) / ORIGINAL_FREQ
        
        if N > 31:
            xs_smooth = savgol_filter(raw_pos[:, 0], 31, 3)
            ys_smooth = savgol_filter(raw_pos[:, 1], 31, 3)
        else:
            xs_smooth, ys_smooth = raw_pos[:, 0], raw_pos[:, 1]
            
        # 2. Upsampling a 60 Hz (Interpolación)
        duration = t_orig[-1]
        n_new = int(duration * TARGET_FREQ)
        
        if n_new < 2:
            start_idx = end_idx; continue

        t_new = np.linspace(0, duration, n_new)
        cs_x = CubicSpline(t_orig, xs_smooth)
        cs_y = CubicSpline(t_orig, ys_smooth)
        xs_new = cs_x(t_new)
        ys_new = cs_y(t_new)
        
        # 3. Calcular Velocidad
        vx = np.gradient(xs_new, 1.0/TARGET_FREQ)
        vy = np.gradient(ys_new, 1.0/TARGET_FREQ)
        vt = np.sqrt(vx**2 + vy**2)
        
        # --- EL TRUCO MAESTRO ---
        # Ponemos 'participant': 9 para TODOS los trials.
        # Como 9 es un número alto, SSSUMO lo mandará al set de 'test' casi seguro.
        # Si queremos estar 100% seguros de que entre en test, mejor variamos:
        # Pero si ponemos todos en 9, y pedimos 'test', SSSUMO verá que solo hay part 9
        # y quizás lo meta en train? 
        # MEJOR ESTRATEGIA: 
        # Participante 0 = Train
        # Participante 1 = Test
        # Asignamos al Trial 0 -> Participante 1 (Test)
        # Asignamos al resto -> Participante 0 (Train)
        
        if trial_idx == 0:
            forced_participant = 9 # Test
        else:
            forced_participant = 0 # Train (Relleno)

        for i in range(n_new):
            data_rows.append({
                'tangential_velocity': vt[i],
                'x': xs_new[i],
                'y': ys_new[i],
                'z': 0.0,
                'time': t_new[i],
                'trial_n': trial_idx,       # Mantenemos el ID original
                'repetition': 0,
                'participant': forced_participant, 
                'experiment': 0
            })
            
        start_idx = end_idx

    df = pd.DataFrame(data_rows)
    cols = ['tangential_velocity', 'x', 'y', 'z', 'time', 'trial_n', 'repetition', 'participant', 'experiment']
    df = df[cols]
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f"✅ CSV Generado: {OUTPUT_CSV_NAME}")
    print("   -> Trial 0 asignado a Part 9 (Test)")
    print("   -> Resto asignados a Part 0 (Train)")

if __name__ == "__main__":
    convert_pusht_force_test()