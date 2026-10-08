# root

*Community 0 | 8 files | cohesion 1.00*

## Definition

This community groups 8 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AdaptiveLambdaScheduler`, `AnnealingScheduler`, `Application`, `BatchCrystallographyAnalyzer`, `BatchSizeProspector`, `BerryPhaseCalculator`, `BerryPhaseResult`, `CheckpointAnalyzer`. Core file: `schrodinger_crystal_fixed.py` (177 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 0 | yes |
| `berry_phase_calculator.py` | py | utility | 17 | yes |
| `crystallographer.py` | py | utility | 114 | yes |
| `experiment2.py` | py | utility | 83 | no |
| `install.sh` | sh | utility | 0 | no |
| `main.py` | py | utility | 120 | yes |
| `orbital_visualizer2.py` | py | utility | 17 | yes |
| `schrodinger_crystal_fixed.py` | py | utility | 177 | yes |

## Key Symbols

- `BerryPhaseResult` (class, `berry_phase_calculator.py:22`) `class BerryPhaseResult` - Results from Berry phase calculation.
- `BerryPhaseCalculator` (class, `berry_phase_calculator.py:37`) `class BerryPhaseCalculator` - Calculates Berry phase from training checkpoint trajectory.
- `__init__` (method, `berry_phase_calculator.py:45`) `def __init__(self, device)`
- `load_checkpoints` (method, `berry_phase_calculator.py:48`) `def load_checkpoints(self, checkpoint_dir)` - Load all checkpoints from directory in chronological order.
- `_extract_epoch` (method, `berry_phase_calculator.py:69`) `def _extract_epoch(self, filepath)` - Extract epoch number from checkpoint filename.
- `extract_spectral_kernels` (method, `berry_phase_calculator.py:74`) `def extract_spectral_kernels(self, state_dict)` - Extract spectral kernels from model state dict.
- `flatten_kernel_params` (method, `berry_phase_calculator.py:107`) `def flatten_kernel_params(self, state_dict)` - Flatten all kernel parameters into a single complex vector.
- `compute_spectral_density` (method, `berry_phase_calculator.py:119`) `def compute_spectral_density(self, kernel)` - Compute spectral density \|W(k)\|².
- `compute_center_of_mass` (method, `berry_phase_calculator.py:123`) `def compute_center_of_mass(self, kernel)` - Compute center of mass in 2D Fourier space.
- `compute_berry_connection_discrete` (method, `berry_phase_calculator.py:157`) `def compute_berry_connection_discrete(self, theta_prev, theta_curr)` - Compute discrete Berry connection between two parameter states.
- `compute_eigenvalue_spectrum` (method, `berry_phase_calculator.py:188`) `def compute_eigenvalue_spectrum(self, kernel)` - Compute eigenvalue spectrum of kernel Gram matrix.
- `compute_eigenvalue_gap` (method, `berry_phase_calculator.py:209`) `def compute_eigenvalue_gap(self, eigenvalues)` - Compute gap between two largest eigenvalues.
- `compute_trajectory_metrics` (method, `berry_phase_calculator.py:215`) `def compute_trajectory_metrics(self, kernels)` - Compute trajectory metrics in parameter space.
- `calculate_berry_phase` (method, `berry_phase_calculator.py:248`) `def calculate_berry_phase(self, checkpoint_dir)` - Main method to calculate Berry phase from checkpoint directory.
- `calculate_from_final_checkpoint` (method, `berry_phase_calculator.py:334`) `def calculate_from_final_checkpoint(self, checkpoint_path)` - Calculate Berry phase estimates from final checkpoint metrics history.
- `visualize_results` (method, `berry_phase_calculator.py:388`) `def visualize_results(result, output_path)` - Create visualization of Berry phase results.
- `main` (method, `berry_phase_calculator.py:514`) `def main()`
- `SchrodingerCrystallographyConfig` (class, `crystallographer.py:61`) `class SchrodingerCrystallographyConfig` - Comprehensive configuration for Schrodinger crystallographic analysis.
- `LoggerFactory` (class, `crystallographer.py:152`) `class LoggerFactory` - Factory for creating configured logger instances.
- `create_logger` (method, `crystallographer.py:159`) `def create_logger(name, level)` - Create and configure a logger with standardized formatting.
- `IMetricCalculator` (class, `crystallographer.py:182`) `class IMetricCalculator(ABC)` - Interface for metric calculation strategies.
- `compute` (method, `crystallographer.py:189`) `def compute(self, model)` - Compute metrics for the given model.
- `HamiltonianOperator` (class, `crystallographer.py:203`) `class HamiltonianOperator` - Analytical Hamiltonian operator for fallback computation.
- `__init__` (method, `crystallographer.py:209`) `def __init__(self, config)` - Initialize the Hamiltonian operator with precomputed spectral operators.
- `_precompute_spectral_operators` (method, `crystallographer.py:220`) `def _precompute_spectral_operators(self)` - Precompute the Laplacian spectrum in Fourier space for efficient application.
- `apply` (method, `crystallographer.py:229`) `def apply(self, field)` - Apply the Hamiltonian operator to a field using spectral methods.
- `time_evolution` (method, `crystallographer.py:243`) `def time_evolution(self, field, dt)` - Perform time evolution of the field under the Hamiltonian.
- `SpectralLayer` (class, `crystallographer.py:262`) `class SpectralLayer(Module)` - Spectral convolution layer operating in Fourier space with complex kernels.
- `__init__` (method, `crystallographer.py:268`) `def __init__(self, channels, grid_size, config)` - Initialize spectral layer with complex-valued kernels.
- `forward` (method, `crystallographer.py:287`) `def forward(self, x)` - Apply spectral convolution in Fourier domain.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 2 file(s) lack file-level docs (e.g. `experiment2.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `berry_phase_calculator.py`
- `crystallographer.py`
- `experiment2.py`
- `install.sh`
- `main.py`
- `orbital_visualizer2.py`
- `schrodinger_crystal_fixed.py`
