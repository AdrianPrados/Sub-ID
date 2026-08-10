# **Sub-ID: Identifiable Decomposition of Submovements in Human Hand Trajectories**
This README file provides an overview of the project "Sub-ID: Identifiable Decomposition of Submovements in Human Hand Trajectories." The project focuses on analyzing and decomposing human hand movement trajectories into identifiable submovements.

The GitHub is under construction 👷, and the code will be made available soon 😊.

---
## 🛠️ Installation

To ensure all dependencies are correctly installed, we use [Conda](https://docs.conda.io/en/latest/) as our environment manager. An `environment.yml` file is provided in the repository.

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/AdrianPrados/Sub-ID
cd /Sub-ID
```

2. Create the Conda environment using the provided file:
```bash
conda env create -f environment.yml
```

3. Activate the newly created environment:
```bash
conda activate SubID
```
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
    *   **Description:** Defines the specific data source or trajectory you want to load and analyze. The script automatically adapts to the spatial dimensions (2D or 3D) and applies the corresponding filtering and preprocessing required for the selected dataset.
    *   **Available Options:** `'SYNTHETIC'`, `'PUSHT'`, `'PUSHTReal2d'`, `'SUBJECT'`, `'LETTERS'`, `'PUSHTReal3d'`, `'SPATULA'`, `'MOVING3D'`.
    *   **Default:** `'LETTERS'`.

*   `--method`
    *   **Description:** Specifies the mathematical primitive model used to decompose the velocity profile of the submovements. 
    *   **Available Options:** 
        *   `minjerk`: Uses Minimum Jerk models (symmetric, polynomial, bell-shaped profiles).
        *   `lgnb`: Uses Log-Normal Bases (asymmetric profiles, dynamically optimizing the skewness parameter $\mu$).
    *   **Default:** `'minjerk'`.

*   `--alpha`
    *   **Description:** Controls the Ridge Regression ($\alpha$) penalty strategy applied during the spatial reconstruction step.
    *   **Available Options:**
        *   `dynamic`: The algorithm automatically calculates the optimal penalty at each step based on the temporal overlap (collinearity) of the velocity bases.
        *   `<float>` (e.g., `0.05` or `0.0000`): Forces a fixed, global penalty across the entire optimization process. *Note: When using the `lgnb` method on real data, a very low penalty (e.g., `0.0000`) is often recommended to prevent aggressive coefficient cancellation caused by the collinearity of the asymmetric tails.*
    *   **Default:** `'dynamic'`.

*   `-h` or `--help`
    *   **Description:** Displays a help message in the console summarizing all available commands and exits.

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
## 📊 Comparison with State-of-the-Art Methods

To benchmark our approach, this repository includes `Comparison_SubID.py`, a separate pipeline that evaluates the unified Sub-ID algorithm against three well-established submovement decomposition methods:

*   **[SSSUMO](https://arxiv.org/abs/2507.08028):** A modern deep learning approach utilizing a Time Delay Neural Network (TDNN) architecture to directly predict submovement onset frames and durations from kinematic data without requiring manual iteration.
*   **[Gowda Method](https://pubmed.ncbi.nlm.nih.gov/26011861/):** A greedy, iterative optimization technique designed to explicitly extract minimum jerk velocity profiles by successively minimizing the residual velocities across the trajectory.
*   **[Scattershot](https://link.springer.com/article/10.1007/s00422-006-0055-y) ([Jason Implementation](https://github.com/JasonFriedman/submovements)):** A stochastic global optimization strategy built to decompose complex 2D planar trajectories into overlapping bell-shaped velocity profiles, ensuring robust convergence while avoiding local minima.

### Running the Comparison Script

Unlike the main `SubID.py` script, `Comparison_SubID.py` is configured by directly modifying the global variables inside the code rather than using command-line arguments.

1. Open `Comparison_SubID.py` in your preferred code editor.
2. Scroll to the `if __name__ == '__main__':` block at the bottom of the file.
3. Modify the core configuration variables to fit your testing needs:
    *   `METHOD`: Set to `'minjerk'` or `'lgnb'`.
    *   `ALPHA_STRATEGY`: Set to `'dynamic'` or a specific float value (e.g., `0.05`).
    *   `DATA_SOURCE`: Choose one of the predefined dataset keys (e.g., `'PUSHTReal3d'`).
4. Run the script from your terminal:
    ```bash
    python Comparison_SubID.py
    ```

Once completed, the script will output 2D/3D multi-axis trajectory and tangential velocity plots into the `Plots_ALL_COMPARISONS` directory, visually contrasting the reconstruction quality of all algorithms against the ground truth.