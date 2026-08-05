import zarr
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def index_range_for_times(t_sensor, t0, t1):
    """
    Return a numpy array of indices in t_sensor that fall between the
    closest samples to t0 and t1 (inclusive).
    """
    i0 = int(np.argmin(np.abs(t_sensor - t0)))
    i1 = int(np.argmin(np.abs(t_sensor - t1)))

    # ensure correct order
    if i0 > i1:
        i0, i1 = i1, i0

    # return index array
    return np.arange(i0, i1 + 1)

def get_speed(states, times):
    """
    Compute absolute speed (scalar velocity magnitude) of a particle
    from position states and timestamps.

    states: (N, 3) or (N, 7) array
    times: (N,) array (not required to be uniform)
    """
    # Extract 3D position
    pos = states[:, :3]

    # Compute velocity (central difference)
    v = np.zeros_like(pos)
    dt = np.diff(times)

    # Forward difference for first
    v[0] = (pos[1] - pos[0]) / dt[0]

    # Central differences for interior
    for i in range(1, len(pos) - 1):
        v[i] = (pos[i+1] - pos[i-1]) / (times[i+1] - times[i-1])

    # Backward difference for last
    v[-1] = (pos[-1] - pos[-2]) / dt[-1]

    # Speed = magnitude of velocity
    speed = np.linalg.norm(v, axis=1)

    return speed, v

def import_data(zarr_root:str):
    zarr_data =  zarr.open(zarr_root, mode='r')
    
    # Visualize zarr structure
    print(zarr_data.tree())
    
    episode_ends_raw = zarr_data['data/episode_ends'][:]      # shape: (N_trials,)
    
    states_raw = zarr_data['data/state'][:]                   # shape: (N_mocap, 7)
    times_raw = zarr_data['data/time'][:]                     # shape: (N_mocap, 1)

    wrench_raw = zarr_data['data/wrench'][:]                  # shape: (N_mocap, 6)
    wrench_times_raw = zarr_data['data/wrench_time'][:]       # shape: (N_mocap, 1)

    if 'data/image' in zarr_data:
        images_raw = zarr_data['data/image']                      # shape: (N_camera, H, W, 3)
        images_times_raw = zarr_data['data/image_times']          # shape: (N_camear, H, W, 3)
    else:
        images_raw = None
        images_times_raw = None
        print("data/image does not exist")

    if 'data/dynamixel' in zarr_data:
        dynamixel_raw = zarr_data['data/dynamixel'][:,1]          # shape: (N_dynamixel, 1) angle in deg
        dynamixel_times_raw = zarr_data['data/dynamixel_time'][:] # shape: (N_dynamixel, 1)
    else:
        print("data/dynamixel does not exist")

    # Ensure if time is (N,) not (N,1) to preven error with interp1d and slerp
    times_raw = np.asarray(times_raw).ravel()
    wrench_times_raw = np.asarray(wrench_times_raw).ravel()
    if 'data/dynamixel' in zarr_data:
        dynamixel_times_raw = np.asarray(dynamixel_times_raw).ravel()

    # # Check sampling frequency 
    # plt.figure()
    # plt.scatter(times_raw[:-1]- times_raw[0],1/np.diff(times_raw),label = "state")
    # plt.scatter(dynamixel_times_raw[:-1]- dynamixel_times_raw[0],1/np.diff(dynamixel_times_raw),label = "dynamixel")
    # plt.scatter(wrench_times_raw[:-1]- wrench_times_raw[0],1/np.diff(wrench_times_raw), label = 'wrench')
    # plt.xlabel("Sample Number")
    # plt.ylabel("Sampling Frequency (Hz)")
    # plt.legend()
    # plt.show()

    # Interpolate paramters
    target_hz = 100
    dt = 1.0 / float(target_hz)

    target_times_list = []
    states_list = []
    wrenches_list = []
    dynamixel_orientationes_list = []
    episode_ends_list = []
    n_list = 0

    # This loop interpolates force, pose and dynamixel data to the same frequency.

    for trial in range(episode_ends_raw.shape[0]):
        if trial == 0:
            dex_start = 0
        else:
            dex_start = episode_ends_raw[trial-1]

        # (FIX) Indexing error some were need to fix later
        dex_end = episode_ends_raw[trial] - 1
        
        # Time window for this trial in mocap time base
        t0 = times_raw[dex_start]
        t1 = times_raw[dex_end]

        # Select dynamixel samples in this time window
        dex_states = np.arange(dex_start, dex_end)
        dex_wrench = index_range_for_times(wrench_times_raw, t0, t1)
        if 'data/dynamixel' in zarr_data:
            dex_dynamixel = index_range_for_times(dynamixel_times_raw, t0, t1)
        
        # Interpolate to a fixed sampling rate for pose 
        # Hard code chop 1 samples to avoid slerp problems
        target_times = np.arange(times_raw[dex_start]+dt, times_raw[dex_end]-dt, dt)

        # Create empty states vector for the trial
        states = np.zeros((len(target_times),7))
        wrenches = np.zeros((len(target_times),6))
        if 'data/dynamixel' in zarr_data:
            dynamixel_orientationes = np.zeros((len(target_times)))

        # Interpolate translation
        for i in range(3):
            # Build interpolators once
            f_interp = interp1d(times_raw[dex_states],
                                states_raw[dex_states, i],
                                kind='cubic') 
            states[:,i] = f_interp(target_times) # Apply interpolation 

        # Interpolation rotation with slerp
        rot_in = Rotation.from_quat(states_raw[dex_states,3:7])
        slerp = Slerp(times_raw[dex_states], rot_in)
        states[:,3:7] = slerp(target_times).as_quat()

        # Interpolate wrenches
        for i in range(6):
            # Build interpolator
            f_interp = interp1d(wrench_times_raw[dex_wrench],
                                wrench_raw[dex_wrench, i],
                                kind='cubic',
                                bounds_error=False,
                                fill_value=(wrench_raw[dex_wrench[0],i], wrench_raw[dex_wrench[-1],i]))
            wrenches[:,i] = f_interp(target_times) # Apply interpolation 

        if 'data/dynamixel' in zarr_data:
            f_interp = interp1d(dynamixel_times_raw[dex_dynamixel],
                                dynamixel_raw[dex_dynamixel],
                                kind='nearest',
                                bounds_error=False,
                                fill_value=(dynamixel_raw[dex_dynamixel[0]], dynamixel_raw[dex_dynamixel[-1]]))
        
            dynamixel_orientationes = f_interp(target_times)

        # 4th order butterworth filter
        order = 4
        target_hz = 100
        cf = 20
        b, a = butter(order, cf / (target_hz / 2), btype="low", analog=False)
        states[:,0] = filtfilt(b, a, states[:,0], axis=0)
        states[:,1] = filtfilt(b, a, states[:,1], axis=0)
        states[:,2] = filtfilt(b, a, states[:,2], axis=0)

        # # Stanity Check
        # fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # # States
        # axes[0].plot(times_raw[dex_states], states_raw[dex_states,0],"o", mfc='none',label = "raw")
        # axes[0].plot(target_times, states[:,0], ".-", label="interp")
        # axes[0].set_ylabel("Position (m)")
        # axes[0].set_xlabel("Time (s)")
        # axes[0].legend()

        # # Wrenches
        # axes[1].plot(wrench_times_raw[dex_wrench], wrench_raw[dex_wrench,0],"o", mfc='none',label = "raw")
        # axes[1].plot(target_times, wrenches[:,0], ".-", label="interp")
        # axes[1].set_ylabel("Force (N)")
        # axes[1].set_xlabel("Time (s)")
        # axes[1].legend()

        # # Dynamixel
        # axes[2].plot(dynamixel_times_raw[dex_dynamixel], dynamixel_raw[dex_dynamixel],"o", mfc='none',label = "raw")
        # axes[2].plot(target_times, dynamixel_orientationes[:], ".-", label="interp")
        # axes[2].set_ylabel("Angle (degrees)")
        # axes[2].set_xlabel("Time (s)")
        # axes[2].legend()
        # plt.show()

        # # Stanity Check
        # plt.figure()
        # plt.plot(times_raw[dex_start:dex_end], states_raw[dex_start:dex_end,0],"o", mfc='none',label = "raw")
        # plt.plot(target_times, states[:,0], ".-", label="interp")
        # plt.legend()
        # plt.show()

        # Add to interpolated state list
        target_times_list.append(target_times)
        states_list.append(states)
        wrenches_list.append(wrenches)
        if 'data/dynamixel' in zarr_data:
            dynamixel_orientationes_list.append(dynamixel_orientationes)

        # Find new episode ends list
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_all = np.concatenate(states_list, axis=0)
    wrenches_all = np.concatenate(wrenches_list, axis=0)
    if 'data/dynamixel' in zarr_data:
        dynamixel_orientationes_all = np.concatenate(dynamixel_orientationes_list, axis=0)
    else:
        dynamixel_orientationes_all = None

    episode_ends_all = np.array(episode_ends_list)

    return states_all, wrenches_all, dynamixel_orientationes_all, episode_ends_all, target_times_all, images_raw, images_times_raw


def main() -> None:
    zarr_root = "/home/adrian/Escritorio/ImitationLearning/LfDSynergies/MutualSynergies/subirGit/data/adrian_data/spatula_pose_raw.zarr"

    states, wrenches, dynamixel, episode_ends, times, images, images_times = import_data(zarr_root)

    speed, v = get_speed(states, times)

    # Select episode 0 or 1
    episode = 0
    if episode == 0:
        dex = range(0,episode_ends[episode])
    else: 
        dex = range(episode_ends[episode]+1,episode_ends[episode+1])

    # Speed
    fig, ax = plt.subplots()
    ax.plot(times[dex], speed[dex])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")

    # Extra plot for conditions of 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    episode = 0
    ax.plot(states[dex,0],states[dex,1], states[dex,2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect((1,1,1))
    plt.show()


if __name__ == "__main__":
    main()
