# **Sub-ID: Identifiable Decomposition of Submovements in Human Hand Trajectories**
This README file provides an overview of the project "Sub-ID: Identifiable Decomposition of Submovements in Human Hand Trajectories." The project focuses on analyzing and decomposing human hand movement trajectories into identifiable submovements.

The GitHub is under construction 👷, and the code will be made available soon 😊.

---

## 🚀 Features and How It Works

The `SubID.py` script is a robust, dimension-agnostic pipeline designed to decompose 2D and 3D kinematic trajectories into overlapping submovements. It operates through a multi-step optimization process:

*   **Multiple Primitive Methods:** Choose between Minimum Jerk (`minjerk`) and Log-Normal Bases (`lgnb`) to model the velocity profiles.
*   **Optimization Pipeline:** Combines a greedy Non-Negative Least Squares (NNLS) initialization with a global L-BFGS-B optimization to refine parameters like start times, durations, and asymmetry ($\mu$).
*   **Dynamic Regularization:** Implements a dynamic Ridge Regression ($\alpha$) penalty.
*   **Collinearity Handling:** The algorithm calculates the true temporal overlap using the velocity kernels to scale the penalty dynamically, preventing basis cancellation in position space.
*   **Automated Merging:** Includes a proximity-based merging step to dynamically combine excessively close submovements and refine the fit.
*   **Dataset Versatility:** Features built-in data loaders for specific `.zarr` and `.csv` datasets, automatically handling resampling, filtering, and 2D/3D spatial dimensions.

## 💻 How to Run

The script is executed via the command line and includes a fully integrated argument parser for quick testing and evaluation.

### Command Line Arguments

You can customize the execution of the algorithm by passing the following flags:

*   `--dataset`
    *   **Description:** Defines the specific data source or trajectory you want to load and analyze[cite: 12]. The script automatically adapts to the spatial dimensions (2D or 3D) and applies the corresponding filtering and preprocessing required for the selected dataset[cite: 12].
    *   **Available Options:** `'SYNTHETIC'`, `'PUSHT'`, `'PUSHTReal2d'`, `'SUBJECT'`, `'LETTERS'`, `'PUSHTReal3d'`, `'SPATULA'`, `'MOVING3D'`[cite: 12].
    *   **Default:** `'LETTERS'`[cite: 12].

*   `--method`
    *   **Description:** Specifies the mathematical primitive model used to decompose the velocity profile of the submovements[cite: 12]. 
    *   **Available Options:** 
        *   `minjerk`: Uses Minimum Jerk models (symmetric, polynomial, bell-shaped profiles)[cite: 12].
        *   `lgnb`: Uses Log-Normal Bases (asymmetric profiles, dynamically optimizing the skewness parameter $\mu$)[cite: 12].
    *   **Default:** `'minjerk'`[cite: 12].

*   `--alpha`
    *   **Description:** Controls the Ridge Regression ($\alpha$) penalty strategy applied during the spatial reconstruction step[cite: 12].
    *   **Available Options:**
        *   `dynamic`: The algorithm automatically calculates the optimal penalty at each step based on the temporal overlap (collinearity) of the velocity bases[cite: 12].
        *   `<float>` (e.g., `0.05` or `0.0000`): Forces a fixed, global penalty across the entire optimization process[cite: 12]. *Note: When using the `lgnb` method on real data, a very low penalty (e.g., `0.0000`) is often recommended to prevent aggressive coefficient cancellation caused by the collinearity of the asymmetric tails.*
    *   **Default:** `'dynamic'`[cite: 12].

*   `-h` or `--help`
    *   **Description:** Displays a help message in the console summarizing all available commands and exits[cite: 12].

### 📖 Usage Examples

**1. Run the default dataset with Minimum Jerk and dynamic Ridge:**
```bash
python SubID.py --dataset PUSHT 
```

**2. Analyze a 3D dataset using the Log-Normal (LGNB) method:**
```bash
python SubID.py --dataset SPATULA --method lgnb
```

**3. Run a synthetic dataset with Minimum Jerk and a fixed Ridge alpha of 0.05::**
```bash
python SubID.py --dataset SPATULA --method minjerk --alpha 0.05
```
