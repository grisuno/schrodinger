# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 8 | **Total Symbols Extracted:** 528 | **Total Imports:** 104

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    crystallographer_py["crystallographer.py (py)"]
    class crystallographer_py mod;
    crystallographer_py_SchrodingerCrystallographyConfig["SchrodingerCrystallographyConfig"]
    class crystallographer_py_SchrodingerCrystallographyConfig cls;
    crystallographer_py --> crystallographer_py_SchrodingerCrystallographyConfig
    crystallographer_py_LoggerFactory["LoggerFactory"]
    class crystallographer_py_LoggerFactory cls;
    crystallographer_py --> crystallographer_py_LoggerFactory
    crystallographer_py_IMetricCalculator["IMetricCalculator"]
    class crystallographer_py_IMetricCalculator cls;
    crystallographer_py --> crystallographer_py_IMetricCalculator
    crystallographer_py_HamiltonianOperator["HamiltonianOperator"]
    class crystallographer_py_HamiltonianOperator cls;
    crystallographer_py --> crystallographer_py_HamiltonianOperator
    crystallographer_py_SpectralLayer["SpectralLayer"]
    class crystallographer_py_SpectralLayer cls;
    crystallographer_py --> crystallographer_py_SpectralLayer
    schrodinger_crystal_fixed_py["schrodinger_crystal_fixed.py (py)"]
    class schrodinger_crystal_fixed_py mod;
    schrodinger_crystal_fixed_py_Config["Config"]
    class schrodinger_crystal_fixed_py_Config cls;
    schrodinger_crystal_fixed_py --> schrodinger_crystal_fixed_py_Config
    schrodinger_crystal_fixed_py_IPhaseDetector["IPhaseDetector"]
    class schrodinger_crystal_fixed_py_IPhaseDetector cls;
    schrodinger_crystal_fixed_py --> schrodinger_crystal_fixed_py_IPhaseDetector
    schrodinger_crystal_fixed_py_IMetricCalculator["IMetricCalculator"]
    class schrodinger_crystal_fixed_py_IMetricCalculator cls;
    schrodinger_crystal_fixed_py --> schrodinger_crystal_fixed_py_IMetricCalculator
    schrodinger_crystal_fixed_py_SeedManager["SeedManager"]
    class schrodinger_crystal_fixed_py_SeedManager cls;
    schrodinger_crystal_fixed_py --> schrodinger_crystal_fixed_py_SeedManager
    schrodinger_crystal_fixed_py_LoggerFactory["LoggerFactory"]
    class schrodinger_crystal_fixed_py_LoggerFactory cls;
    schrodinger_crystal_fixed_py --> schrodinger_crystal_fixed_py_LoggerFactory
    main_py["main.py (py)"]
    class main_py mod;
    main_py_Config["Config"]
    class main_py_Config cls;
    main_py --> main_py_Config
    main_py_SeedManager["SeedManager"]
    class main_py_SeedManager cls;
    main_py --> main_py_SeedManager
    main_py_LoggerFactory["LoggerFactory"]
    class main_py_LoggerFactory cls;
    main_py --> main_py_LoggerFactory
    main_py_IAnalysisStrategy["IAnalysisStrategy"]
    class main_py_IAnalysisStrategy cls;
    main_py --> main_py_IAnalysisStrategy
    main_py_IMetricsCalculator["IMetricsCalculator"]
    class main_py_IMetricsCalculator cls;
    main_py --> main_py_IMetricsCalculator
    experiment2_py["experiment2.py (py)"]
    class experiment2_py mod;
    experiment2_py_Config["Config"]
    class experiment2_py_Config cls;
    experiment2_py --> experiment2_py_Config
    experiment2_py_SeedManager["SeedManager"]
    class experiment2_py_SeedManager cls;
    experiment2_py --> experiment2_py_SeedManager
    experiment2_py_LoggerFactory["LoggerFactory"]
    class experiment2_py_LoggerFactory cls;
    experiment2_py --> experiment2_py_LoggerFactory
    experiment2_py_IAnalysisStrategy["IAnalysisStrategy"]
    class experiment2_py_IAnalysisStrategy cls;
    experiment2_py --> experiment2_py_IAnalysisStrategy
    experiment2_py_IMetricsCalculator["IMetricsCalculator"]
    class experiment2_py_IMetricsCalculator cls;
    experiment2_py --> experiment2_py_IMetricsCalculator
    orbital_visualizer2_py["orbital_visualizer2.py (py)"]
    class orbital_visualizer2_py mod;
    orbital_visualizer2_py_Config["Config"]
    class orbital_visualizer2_py_Config cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_Config
    orbital_visualizer2_py_WavefunctionCalculator["WavefunctionCalculator"]
    class orbital_visualizer2_py_WavefunctionCalculator cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_WavefunctionCalculator
    orbital_visualizer2_py_HamiltonianNNProcessor["HamiltonianNNProcessor"]
    class orbital_visualizer2_py_HamiltonianNNProcessor cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_HamiltonianNNProcessor
    orbital_visualizer2_py_MonteCarloSampler["MonteCarloSampler"]
    class orbital_visualizer2_py_MonteCarloSampler cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_MonteCarloSampler
    orbital_visualizer2_py_OrbitalVisualizer["OrbitalVisualizer"]
    class orbital_visualizer2_py_OrbitalVisualizer cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_OrbitalVisualizer
    berry_phase_calculator_py["berry_phase_calculator.py (py)"]
    class berry_phase_calculator_py mod;
    berry_phase_calculator_py_BerryPhaseResult["BerryPhaseResult"]
    class berry_phase_calculator_py_BerryPhaseResult cls;
    berry_phase_calculator_py --> berry_phase_calculator_py_BerryPhaseResult
    berry_phase_calculator_py_BerryPhaseCalculator["BerryPhaseCalculator"]
    class berry_phase_calculator_py_BerryPhaseCalculator cls;
    berry_phase_calculator_py --> berry_phase_calculator_py_BerryPhaseCalculator
    berry_phase_calculator_py_visualize_results["visualize_results"]
    class berry_phase_calculator_py_visualize_results fn;
    berry_phase_calculator_py --> berry_phase_calculator_py_visualize_results
    berry_phase_calculator_py_main["main"]
    class berry_phase_calculator_py_main fn;
    berry_phase_calculator_py --> berry_phase_calculator_py_main
    berry_phase_calculator_py___init__["__init__"]
    class berry_phase_calculator_py___init__ fn;
    berry_phase_calculator_py --> berry_phase_calculator_py___init__
    app_py["app.py (py)"]
    class app_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_torch["torch"]
    class ext_torch ext;
    berry_phase_calculator_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    berry_phase_calculator_py -.->|imports| ext_torch_nn
    ext_numpy["numpy"]
    class ext_numpy ext;
    berry_phase_calculator_py -.->|imports| ext_numpy
    ext_os["os"]
    class ext_os ext;
    berry_phase_calculator_py -.->|imports| ext_os
    ext_glob["glob"]
    class ext_glob ext;
    berry_phase_calculator_py -.->|imports| ext_glob
    ext_re["re"]
    class ext_re ext;
    berry_phase_calculator_py -.->|imports| ext_re
    ext_typing["typing"]
    class ext_typing ext;
    berry_phase_calculator_py -.->|imports| ext_typing
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    berry_phase_calculator_py -.->|imports| ext_dataclasses
    ext_json["json"]
    class ext_json ext;
    berry_phase_calculator_py -.->|imports| ext_json
    ext_argparse["argparse"]
    class ext_argparse ext;
    berry_phase_calculator_py -.->|imports| ext_argparse
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    berry_phase_calculator_py -.->|imports| ext_matplotlib_pyplot
    ext_logging["logging"]
    class ext_logging ext;
    crystallographer_py -.->|imports| ext_logging
    crystallographer_py -.->|imports| ext_argparse
    crystallographer_py -.->|imports| ext_torch
    crystallographer_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    crystallographer_py -.->|imports| ext_torch_nn_functional
    crystallographer_py -.->|imports| ext_numpy
    crystallographer_py -.->|imports| ext_os
    crystallographer_py -.->|imports| ext_json
    ext_math["math"]
    class ext_math ext;
    crystallographer_py -.->|imports| ext_math
    ext_copy["copy"]
    class ext_copy ext;
    crystallographer_py -.->|imports| ext_copy
    ext_warnings["warnings"]
    class ext_warnings ext;
    crystallographer_py -.->|imports| ext_warnings
    ext_datetime["datetime"]
    class ext_datetime ext;
    crystallographer_py -.->|imports| ext_datetime
    crystallographer_py -.->|imports| ext_typing
    ext_abc["abc"]
    class ext_abc ext;
    crystallographer_py -.->|imports| ext_abc
    crystallographer_py -.->|imports| ext_dataclasses
    ext_collections["collections"]
    class ext_collections ext;
    crystallographer_py -.->|imports| ext_collections
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    crystallographer_py -.->|imports| ext_pathlib
    crystallographer_py -.->|imports| ext_dataclasses
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    crystallographer_py -.->|imports| ext_matplotlib
    crystallographer_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_gridspec["matplotlib.gridspec"]
    class ext_matplotlib_gridspec ext;
    crystallographer_py -.->|imports| ext_matplotlib_gridspec
    ext_seaborn["seaborn"]
    class ext_seaborn ext;
    crystallographer_py -.->|imports| ext_seaborn
    ext_scipy["scipy"]
    class ext_scipy ext;
    crystallographer_py -.->|imports| ext_scipy
    ext_scipy_stats["scipy.stats"]
    class ext_scipy_stats ext;
    crystallographer_py -.->|imports| ext_scipy_stats
    ext_scipy_linalg["scipy.linalg"]
    class ext_scipy_linalg ext;
    crystallographer_py -.->|imports| ext_scipy_linalg
    ext_scipy_ndimage["scipy.ndimage"]
    class ext_scipy_ndimage ext;
    crystallographer_py -.->|imports| ext_scipy_ndimage
    ext_sklearn_decomposition["sklearn.decomposition"]
    class ext_sklearn_decomposition ext;
    crystallographer_py -.->|imports| ext_sklearn_decomposition
    experiment2_py -.->|imports| ext_argparse
    experiment2_py -.->|imports| ext_torch
    experiment2_py -.->|imports| ext_torch_nn
    experiment2_py -.->|imports| ext_torch_nn_functional
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    experiment2_py -.->|imports| ext_torch_optim
    ext_torch_utils_data["torch.utils.data"]
    class ext_torch_utils_data ext;
    experiment2_py -.->|imports| ext_torch_utils_data
    experiment2_py -.->|imports| ext_numpy
    experiment2_py -.->|imports| ext_os
    ext_time["time"]
    class ext_time ext;
    experiment2_py -.->|imports| ext_time
    experiment2_py -.->|imports| ext_json
    experiment2_py -.->|imports| ext_datetime
    experiment2_py -.->|imports| ext_typing
    experiment2_py -.->|imports| ext_abc
    experiment2_py -.->|imports| ext_dataclasses
    experiment2_py -.->|imports| ext_collections
    experiment2_py -.->|imports| ext_logging
    ext_traceback["traceback"]
    class ext_traceback ext;
    experiment2_py -.->|imports| ext_traceback
    main_py -.->|imports| ext_argparse
    main_py -.->|imports| ext_torch
    main_py -.->|imports| ext_torch_nn
    main_py -.->|imports| ext_torch_nn_functional
    main_py -.->|imports| ext_torch_optim
    main_py -.->|imports| ext_torch_utils_data
    main_py -.->|imports| ext_numpy
    main_py -.->|imports| ext_os
    main_py -.->|imports| ext_time
    main_py -.->|imports| ext_json
    main_py -.->|imports| ext_datetime
    main_py -.->|imports| ext_typing
    main_py -.->|imports| ext_abc
    main_py -.->|imports| ext_dataclasses
    main_py -.->|imports| ext_collections
    main_py -.->|imports| ext_logging
    main_py -.->|imports| ext_math
    main_py -.->|imports| ext_copy
    orbital_visualizer2_py -.->|imports| ext_numpy
    ext_scipy_special["scipy.special"]
    class ext_scipy_special ext;
    orbital_visualizer2_py -.->|imports| ext_scipy_special
    orbital_visualizer2_py -.->|imports| ext_matplotlib_pyplot
    orbital_visualizer2_py -.->|imports| ext_matplotlib
    orbital_visualizer2_py -.->|imports| ext_torch
    orbital_visualizer2_py -.->|imports| ext_os
    ext_sys["sys"]
    class ext_sys ext;
    orbital_visualizer2_py -.->|imports| ext_sys
    orbital_visualizer2_py -.->|imports| ext_warnings
    orbital_visualizer2_py -.->|imports| ext_typing
    ext_schrodinger_crystal_fixed2["schrodinger_crystal_fixed2"]
    class ext_schrodinger_crystal_fixed2 ext;
    orbital_visualizer2_py -.->|imports| ext_schrodinger_crystal_fixed2
    ext_plotly_graph_objects["plotly.graph_objects"]
    class ext_plotly_graph_objects ext;
    orbital_visualizer2_py -.->|imports| ext_plotly_graph_objects
    orbital_visualizer2_py -.->|imports| ext_traceback
    schrodinger_crystal_fixed_py -.->|imports| ext_argparse
    schrodinger_crystal_fixed_py -.->|imports| ext_torch
    schrodinger_crystal_fixed_py -.->|imports| ext_torch_nn
    schrodinger_crystal_fixed_py -.->|imports| ext_torch_nn_functional
    schrodinger_crystal_fixed_py -.->|imports| ext_torch_optim
    schrodinger_crystal_fixed_py -.->|imports| ext_torch_utils_data
    schrodinger_crystal_fixed_py -.->|imports| ext_numpy
    schrodinger_crystal_fixed_py -.->|imports| ext_os
    schrodinger_crystal_fixed_py -.->|imports| ext_time
    schrodinger_crystal_fixed_py -.->|imports| ext_json
    schrodinger_crystal_fixed_py -.->|imports| ext_datetime
    schrodinger_crystal_fixed_py -.->|imports| ext_typing
    schrodinger_crystal_fixed_py -.->|imports| ext_abc
    schrodinger_crystal_fixed_py -.->|imports| ext_dataclasses
    schrodinger_crystal_fixed_py -.->|imports| ext_collections
    schrodinger_crystal_fixed_py -.->|imports| ext_logging
    schrodinger_crystal_fixed_py -.->|imports| ext_math
    schrodinger_crystal_fixed_py -.->|imports| ext_copy
    schrodinger_crystal_fixed_py -.->|imports| ext_warnings
```

---

## Architecture Reference

### PY (7 files)

#### `app.py`
**Path:** `app.py`

*No symbols extracted*

#### `berry_phase_calculator.py`
**Path:** `berry_phase_calculator.py`

**Classes:**
- `BerryPhaseResult` (line 22) `class BerryPhaseResult` - *Results from Berry phase calculation.*
- `BerryPhaseCalculator` (line 37) `class BerryPhaseCalculator` - *Calculates Berry phase from training checkpoint trajectory.

Uses the spectral kernel parameters as the parameter space θ,
and computes the geometric phase accumulated during training.*

**Functions:**
- `visualize_results` (line 388) `def visualize_results(result, output_path)` - *Create visualization of Berry phase results.*
- `main` (line 514) `def main()`
- `__init__` (line 45) `def __init__(self, device)`
- `load_checkpoints` (line 48) `def load_checkpoints(self, checkpoint_dir)` - *Load all checkpoints from directory in chronological order.*
- `_extract_epoch` (line 69) `def _extract_epoch(self, filepath)` - *Extract epoch number from checkpoint filename.*
- `extract_spectral_kernels` (line 74) `def extract_spectral_kernels(self, state_dict)` - *Extract spectral kernels from model state dict.
Returns dict with 'real' and 'imag' kernels per layer.*
- `flatten_kernel_params` (line 107) `def flatten_kernel_params(self, state_dict)` - *Flatten all kernel parameters into a single complex vector.*
- `compute_spectral_density` (line 119) `def compute_spectral_density(self, kernel)` - *Compute spectral density |W(k)|².*
- `compute_center_of_mass` (line 123) `def compute_center_of_mass(self, kernel)` - *Compute center of mass in 2D Fourier space.

For each kernel layer [C_out, C_in, freq_h, freq_w]:
- Map frequency indices to angular coordinates
- Compute weighted CM*
- `compute_berry_connection_discrete` (line 157) `def compute_berry_connection_discrete(self, theta_prev, theta_curr)` - *Compute discrete Berry connection between two parameter states.

Uses the formula for discrete Berry phase:
A = Im[log(⟨ψ(θ_{n-1})|ψ(θ_n)⟩)]

For parameter vectors, this is:
A = Im[log(θ_{n-1}^* · θ_n)]*
- `compute_eigenvalue_spectrum` (line 188) `def compute_eigenvalue_spectrum(self, kernel)` - *Compute eigenvalue spectrum of kernel Gram matrix.*
- `compute_eigenvalue_gap` (line 209) `def compute_eigenvalue_gap(self, eigenvalues)` - *Compute gap between two largest eigenvalues.*
- `compute_trajectory_metrics` (line 215) `def compute_trajectory_metrics(self, kernels)` - *Compute trajectory metrics in parameter space.*
- `calculate_berry_phase` (line 248) `def calculate_berry_phase(self, checkpoint_dir)` - *Main method to calculate Berry phase from checkpoint directory.*
- `calculate_from_final_checkpoint` (line 334) `def calculate_from_final_checkpoint(self, checkpoint_path)` - *Calculate Berry phase estimates from final checkpoint metrics history.*

#### `crystallographer.py`
**Path:** `crystallographer.py`

**Classes:**
- `SchrodingerCrystallographyConfig` (line 61) `class SchrodingerCrystallographyConfig` - *Comprehensive configuration for Schrodinger crystallographic analysis.
All parameters are centralized here following the Single Responsibility Principle.
No magic numbers or hardcoded values appear elsewhere in the codebase.*
- `LoggerFactory` (line 152) `class LoggerFactory` - *Factory for creating configured logger instances.
Follows the Single Responsibility Principle for logging configuration.*
- `IMetricCalculator` (line 182) `class IMetricCalculator(ABC)` - *Interface for metric calculation strategies.
Follows the Interface Segregation Principle by defining a minimal contract.*
- `HamiltonianOperator` (line 203) `class HamiltonianOperator` - *Analytical Hamiltonian operator for fallback computation.
Implements spectral operators for Laplacian computation in Fourier space.*
- `SpectralLayer` (line 262) `class SpectralLayer` - *Spectral convolution layer operating in Fourier space with complex kernels.
Implements learnable frequency-domain transformations.*
- `HamiltonianBackbone` (line 316) `class HamiltonianBackbone` - *Neural network backbone for Hamiltonian inference.
Learns to approximate Hamiltonian operations from data.*
- `SchrodingerSpectralNetwork` (line 358) `class SchrodingerSpectralNetwork` - *Schrodinger equation neural network with spectral layers.
Implements expansion-contraction architecture with Fourier convolutions.*
- `HamiltonianInferenceEngine` (line 404) `class HamiltonianInferenceEngine` - *Engine for Hamiltonian inference using either a pretrained backbone
or analytical operators as fallback.*
- `SchrodingerPotentialGenerator` (line 490) `class SchrodingerPotentialGenerator` - *Generator for various potential energy landscapes used in Schrodinger equation.
Supports harmonic, double-well, Coulomb-like, and periodic lattice potentials.*
- `SyntheticDataGenerator` (line 584) `class SyntheticDataGenerator` - *Generator for synthetic Schrodinger equation training data.
Creates initial and target wavefunction pairs for various potentials.*
- `WeightIntegrityCalculator` (line 709) `class WeightIntegrityCalculator(IMetricCalculator)` - *Calculator for weight integrity metrics including NaN and Inf detection.*
- `DiscretizationCalculator` (line 772) `class DiscretizationCalculator(IMetricCalculator)` - *Calculator for discretization margin and alpha purity metrics.
These metrics quantify how close weights are to integer values.*
- `LocalComplexityCalculator` (line 852) `class LocalComplexityCalculator(IMetricCalculator)` - *Calculator for local complexity metrics measuring weight diversity.*
- `SuperpositionCalculator` (line 911) `class SuperpositionCalculator(IMetricCalculator)` - *Calculator for superposition metrics measuring weight correlations.*
- `GradientDynamicsCalculator` (line 974) `class GradientDynamicsCalculator(IMetricCalculator)` - *Calculator for gradient-based metrics including condition number and effective temperature.*
- `SpectralGeometryCalculator` (line 1092) `class SpectralGeometryCalculator(IMetricCalculator)` - *Calculator for spectral geometry metrics including MBL level spacing.*
- `RicciCurvatureCalculator` (line 1182) `class RicciCurvatureCalculator(IMetricCalculator)` - *Calculator for Ricci curvature estimation in weight space.*
- `ThermodynamicCalculator` (line 1266) `class ThermodynamicCalculator(IMetricCalculator)` - *Calculator for thermodynamic potentials including Gibbs free energy.*
- `KappaQuantumCalculator` (line 1346) `class KappaQuantumCalculator(IMetricCalculator)` - *Calculator for quantum condition number of the weight covariance.*
- `PoyntingVectorCalculator` (line 1397) `class PoyntingVectorCalculator(IMetricCalculator)` - *Calculator for Poynting vector magnitude representing energy flow.*
- `HbarEffectiveCalculator` (line 1490) `class HbarEffectiveCalculator(IMetricCalculator)` - *Calculator for effective Planck constant under lambda pressure.*
- `WeightDiffractionCalculator` (line 1529) `class WeightDiffractionCalculator(IMetricCalculator)` - *Calculator for weight diffraction analysis with advanced Bragg peak detection.*
- `PhaseStructureCalculator` (line 1620) `class PhaseStructureCalculator(IMetricCalculator)` - *Calculator for phase structure analysis using histogram-based methods.*
- `ComplexKernelHolomorphyCalculator` (line 1679) `class ComplexKernelHolomorphyCalculator(IMetricCalculator)` - *Calculator for analyzing holomorphy of complex spectral kernels using Cauchy-Riemann equations.*
- `NormConservationCalculator` (line 1762) `class NormConservationCalculator(IMetricCalculator)` - *Calculator for norm conservation error in Schrodinger dynamics.*
- `CrystallographicGrader` (line 1800) `class CrystallographicGrader` - *Advanced grader for crystallographic quality assessment.
Implements refined threshold-based grading system.*
- `PhaseClassifier` (line 1873) `class PhaseClassifier` - *Classifier for thermodynamic phase identification.*
- `CheckpointLoader` (line 1943) `class CheckpointLoader` - *Loader for neural network checkpoints with architecture reconstruction.*
- `DefinitiveCrystallographySuite` (line 2073) `class DefinitiveCrystallographySuite` - *Comprehensive crystallographic analysis suite combining all metric calculators.
Follows the Single Responsibility Principle for orchestration.*
- `BatchCrystallographyAnalyzer` (line 2354) `class BatchCrystallographyAnalyzer` - *Batch analysis orchestrator with visualization generation.*

**Functions:**
- `build_argument_parser` (line 2551) `def build_argument_parser()` - *Build the command-line argument parser.

Returns:
    Configured ArgumentParser instance.*
- `main` (line 2611) `def main()` - *Main entry point for the definitive crystallographer.*
- `create_logger` (line 159) `def create_logger(name, level)` - *Create and configure a logger with standardized formatting.

Args:
    name: Identifier for the logger instance.
    level: Logging level string (DEBUG, INFO, WARNING, ERROR).

Returns:
    Configured Logger instance ready for use.*
- `compute` (line 189) `def compute(self, model)` - *Compute metrics for the given model.

Args:
    model: Neural network model to analyze.
    **kwargs: Additional parameters required for computation.

Returns:
    Dictionary containing computed metric values.*
- `__init__` (line 209) `def __init__(self, config)` - *Initialize the Hamiltonian operator with precomputed spectral operators.

Args:
    config: Configuration containing grid size and other parameters.*
- `_precompute_spectral_operators` (line 220) `def _precompute_spectral_operators(self)` - *Precompute the Laplacian spectrum in Fourier space for efficient application.*
- `apply` (line 229) `def apply(self, field)` - *Apply the Hamiltonian operator to a field using spectral methods.

Args:
    field: Input field tensor (2D or batched).

Returns:
    Transformed field after Hamiltonian application.*
- `time_evolution` (line 243) `def time_evolution(self, field, dt)` - *Perform time evolution of the field under the Hamiltonian.

Args:
    field: Input field tensor.
    dt: Time step size.

Returns:
    Time-evolved field with preserved norm.*
- `__init__` (line 268) `def __init__(self, channels, grid_size, config)` - *Initialize spectral layer with complex-valued kernels.

Args:
    channels: Number of input/output channels.
    grid_size: Spatial dimension of the input grid.
    config: Configuration object (optional for parameter access).*
- `forward` (line 287) `def forward(self, x)` - *Apply spectral convolution in Fourier domain.

Args:
    x: Input tensor of shape (batch, channels, height, width).

Returns:
    Transformed tensor after spectral convolution.*
- `__init__` (line 322) `def __init__(self, config)` - *Initialize the Hamiltonian backbone network.

Args:
    config: Configuration containing architecture parameters.*
- `forward` (line 338) `def forward(self, x)` - *Forward pass through the Hamiltonian backbone.

Args:
    x: Input tensor of shape (batch, height, width) or (batch, 1, height, width).

Returns:
    Hamiltonian-transformed output tensor.*
- `__init__` (line 364) `def __init__(self, config)` - *Initialize the Schrodinger spectral network.

Args:
    config: Configuration containing all architecture parameters.*
- `forward` (line 384) `def forward(self, x)` - *Forward pass through the Schrodinger network.

Args:
    x: Input tensor of shape (batch, channels, height, width).

Returns:
    Output tensor after expansion, spectral processing, and contraction.*
- `__init__` (line 410) `def __init__(self, config)` - *Initialize the Hamiltonian inference engine.

Args:
    config: Configuration containing backbone path and device settings.*
- `_try_load_backbone` (line 423) `def _try_load_backbone(self)` - *Attempt to load a pretrained backbone for Hamiltonian inference.
Falls back to analytical operator if backbone is unavailable.*
- `apply_hamiltonian` (line 454) `def apply_hamiltonian(self, field)` - *Apply the Hamiltonian to a field using backbone or analytical operator.

Args:
    field: Input field tensor.

Returns:
    Hamiltonian-transformed field.*
- `time_evolve` (line 469) `def time_evolve(self, field, dt)` - *Perform time evolution using backbone or analytical operator.

Args:
    field: Input field tensor.
    dt: Time step size.

Returns:
    Time-evolved field with preserved norm.*
- `__init__` (line 496) `def __init__(self, config)` - *Initialize the potential generator.

Args:
    config: Configuration containing potential parameters.*
- `harmonic_potential` (line 506) `def harmonic_potential(self)` - *Generate a harmonic oscillator potential.

Returns:
    2D tensor with harmonic potential centered at grid center.*
- `double_well_potential` (line 520) `def double_well_potential(self)` - *Generate a double-well potential.

Returns:
    2D tensor with double-well potential along x-axis.*
- `coulomb_like_potential` (line 534) `def coulomb_like_potential(self)` - *Generate a Coulomb-like central potential.

Returns:
    2D tensor with Coulomb potential centered at grid center.*
- `periodic_lattice_potential` (line 548) `def periodic_lattice_potential(self)` - *Generate a periodic lattice potential.

Returns:
    2D tensor with periodic cosine potential.*
- `generate_mixed_potential` (line 560) `def generate_mixed_potential(self, seed)` - *Generate a mixed potential combining multiple potential types.

Args:
    seed: Random seed for determining mixture weights.

Returns:
    2D tensor with weighted combination of potential types.*
- `__init__` (line 590) `def __init__(self, config, hamiltonian_engine)` - *Initialize the synthetic data generator.

Args:
    config: Configuration containing data generation parameters.
    hamiltonian_engine: Engine for Hamiltonian operations.*
- `generate_batch` (line 603) `def generate_batch(self, seed)` - *Generate a batch of validation data.

Args:
    seed: Random seed for reproducibility.

Returns:
    Tuple of (initial_states, target_states) tensors.*
- `_solve_schrodinger_sample` (line 633) `def _solve_schrodinger_sample(self, potential, sample_seed)` - *Solve the Schrodinger equation for a single sample.

Args:
    potential: Potential energy landscape.
    sample_seed: Random seed for eigenstate selection.

Returns:
    Tuple of (real_part, imaginary_part, energy) for the wavefunction.*
- `_time_evolve_wavefunction` (line 673) `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential)` - *Time evolve a wavefunction under the given potential.

Args:
    psi_real: Real part of the wavefunction.
    psi_imag: Imaginary part of the wavefunction.
    potential: Potential energy landscape.

Returns:
    Tuple of (evolved_real, evolved_imag) wavefunction components.*
- `__init__` (line 714) `def __init__(self, config)` - *Initialize the weight integrity calculator.

Args:
    config: Configuration containing tolerance parameters.*
- `compute` (line 723) `def compute(self, model)` - *Compute weight integrity metrics for the model.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing integrity metrics:
        - is_valid: Boolean indicating no NaN or Inf values
        - has_nan: Boolean indicating presence of NaN values
        - has_inf: Boolean indicating presence of Inf values
        - total_params: Total number of parameters
        - nan_count: Number of NaN values
        - inf_count: Number of Inf values
        - corruption_ratio: Ratio of corrupted to total parameters*
- `__init__` (line 778) `def __init__(self, config)` - *Initialize the discretization calculator.

Args:
    config: Configuration containing discretization thresholds.*
- `compute` (line 787) `def compute(self, model)` - *Compute discretization metrics for the model.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - delta: Maximum discretization margin
        - alpha: Purity index (-log(delta))
        - spectral_entropy: Entropy of weight power spectrum
        - is_discrete: Boolean indicating discretization below threshold
        - layer_deltas: Per-layer discretization margins*
- `_compute_spectral_entropy` (line 828) `def _compute_spectral_entropy(self, weights)` - *Compute the spectral entropy of the weight distribution.

Args:
    weights: Flattened weight tensor.

Returns:
    Spectral entropy value.*
- `__init__` (line 857) `def __init__(self, config)` - *Initialize the local complexity calculator.

Args:
    config: Configuration containing dimension limits.*
- `compute` (line 866) `def compute(self, model)` - *Compute local complexity for the model weights.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing local_complexity value.*
- `_compute_local_complexity` (line 887) `def _compute_local_complexity(self, weights)` - *Compute local complexity for a weight matrix.

Args:
    weights: 2D weight tensor.

Returns:
    Local complexity value between 0 and 1.*
- `__init__` (line 916) `def __init__(self, config)` - *Initialize the superposition calculator.

Args:
    config: Configuration containing computation parameters.*
- `compute` (line 925) `def compute(self, model)` - *Compute superposition metrics for the model.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing superposition value.*
- `_compute_superposition` (line 946) `def _compute_superposition(self, weights)` - *Compute superposition metric for a weight matrix.

Args:
    weights: 2D weight tensor.

Returns:
    Superposition value indicating average correlation.*
- `__init__` (line 979) `def __init__(self, config)` - *Initialize the gradient dynamics calculator.

Args:
    config: Configuration containing gradient computation parameters.*
- `compute` (line 988) `def compute(self, model)` - *Compute gradient dynamics metrics.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain 'val_x' and 'val_y' tensors.

Returns:
    Dictionary containing:
        - kappa: Condition number of gradient covariance
        - effective_temperature: Temperature derived from gradient variance
        - gradient_variance: Variance of gradient samples*
- `__init__` (line 1097) `def __init__(self, config)` - *Initialize the spectral geometry calculator.

Args:
    config: Configuration containing spectral analysis parameters.*
- `compute` (line 1106) `def compute(self, model)` - *Compute spectral geometry metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - spectral_gap: Gap between largest eigenvalues
        - effective_dimension: Number of significant eigenvalues
        - participation_ratio: Measure of eigenvalue distribution
        - level_spacing_ratio: MBL indicator
        - largest_eigenvalue: Maximum eigenvalue
        - smallest_eigenvalue: Minimum eigenvalue*
- `_compute_level_spacing_ratio` (line 1161) `def _compute_level_spacing_ratio(self, spacings)` - *Compute the level spacing ratio for MBL analysis.

Args:
    spacings: Array of eigenvalue level spacings.

Returns:
    Average level spacing ratio.*
- `__init__` (line 1187) `def __init__(self, config)` - *Initialize the Ricci curvature calculator.

Args:
    config: Configuration containing curvature estimation parameters.*
- `compute` (line 1196) `def compute(self, model)` - *Compute Ricci curvature metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - ricci_scalar: Estimated Ricci scalar curvature
        - mean_sectional_curvature: Average sectional curvature
        - curvature_variance: Variance of sectional curvatures*
- `_compute_ricci_scalar` (line 1225) `def _compute_ricci_scalar(self, metric)` - *Compute the Ricci scalar from the metric tensor.

Args:
    metric: Metric tensor as numpy array.

Returns:
    Estimated Ricci scalar.*
- `_estimate_sectional_curvatures` (line 1243) `def _estimate_sectional_curvatures(self, metric, samples)` - *Estimate sectional curvatures by sampling 2D sections.

Args:
    metric: Metric tensor as numpy array.
    samples: Number of sectional curvature samples.

Returns:
    Array of estimated sectional curvatures.*
- `__init__` (line 1271) `def __init__(self, config)` - *Initialize the thermodynamic calculator.

Args:
    config: Configuration containing thermodynamic parameters.*
- `compute` (line 1280) `def compute(self, model)` - *Compute thermodynamic metrics.

Args:
    model: Neural network model (unused but required by interface).
    **kwargs: Must contain delta, alpha, kappa, effective_temperature.

Returns:
    Dictionary containing:
        - gibbs_free_energy: Gibbs free energy estimate
        - entropy_proxy: Entropy approximation
        - critical_temperature_estimate: Predicted critical temperature
        - phase_stability: Stability classification
        - phase_type: Phase classification string*
- `_classify_phase` (line 1320) `def _classify_phase(self, delta, kappa, temp, alpha)` - *Classify the thermodynamic phase based on metrics.

Args:
    delta: Discretization margin.
    kappa: Condition number.
    temp: Effective temperature.
    alpha: Purity index.

Returns:
    Phase classification string.*
- `__init__` (line 1351) `def __init__(self, config)` - *Initialize the kappa quantum calculator.

Args:
    config: Configuration containing quantum parameters.*
- `compute` (line 1360) `def compute(self, model)` - *Compute quantum condition number.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing kappa_quantum value.*
- `__init__` (line 1402) `def __init__(self, config)` - *Initialize the Poynting vector calculator.

Args:
    config: Configuration containing energy flow parameters.*
- `compute` (line 1411) `def compute(self, model)` - *Compute Poynting vector metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - poynting_magnitude: Magnitude of energy flow
        - is_radiating: Boolean indicating significant energy flow
        - field_orthogonality: Measure of field orthogonality
        - energy_distribution: Dictionary of energy distribution metrics*
- `__init__` (line 1495) `def __init__(self, config)` - *Initialize the hbar effective calculator.

Args:
    config: Configuration containing physical constants.*
- `compute` (line 1504) `def compute(self, model)` - *Compute effective hbar.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain delta and lambda_pressure.

Returns:
    Dictionary containing hbar_effective value.*
- `__init__` (line 1534) `def __init__(self, config)` - *Initialize the weight diffraction calculator.

Args:
    config: Configuration containing spectral analysis parameters.*
- `compute` (line 1543) `def compute(self, model)` - *Compute weight diffraction metrics with Bragg peak detection.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - bragg_peaks: List of detected Bragg peaks
        - is_crystalline_structure: Boolean indicating crystalline pattern
        - spectral_entropy: Entropy of power spectrum
        - num_peaks: Number of detected peaks*
- `_compute_spectral_entropy` (line 1602) `def _compute_spectral_entropy(self, power_spectrum)` - *Compute spectral entropy of the power spectrum.

Args:
    power_spectrum: Power spectrum tensor.

Returns:
    Spectral entropy value.*
- `__init__` (line 1625) `def __init__(self, config)` - *Initialize the phase structure calculator.

Args:
    config: Configuration containing histogram parameters.*
- `compute` (line 1634) `def compute(self, model)` - *Compute phase structure metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - per_layer: Per-layer phase classifications
        - distribution: Count of each phase type
        - dominant_phase: Most common phase type*
- `__init__` (line 1684) `def __init__(self, config)` - *Initialize the holomorphy calculator.

Args:
    config: Configuration containing holomorphy threshold.*
- `compute` (line 1693) `def compute(self, model)` - *Compute holomorphy metrics for complex kernels.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - per_layer: Per-layer holomorphy analysis
        - holomorphic_fraction: Fraction of holomorphic layers
        - average_cr_error: Average Cauchy-Riemann error*
- `__init__` (line 1767) `def __init__(self, config)` - *Initialize the norm conservation calculator.

Args:
    config: Configuration containing normalization parameters.*
- `compute` (line 1776) `def compute(self, model)` - *Compute norm conservation error.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain val_x tensor.

Returns:
    Dictionary containing norm_conservation_error value.*
- `__init__` (line 1806) `def __init__(self, config)` - *Initialize the crystallographic grader.

Args:
    config: Configuration containing grading thresholds.*
- `assign_grade` (line 1815) `def assign_grade(self, delta, alpha, kappa, num_bragg_peaks)` - *Assign a crystallographic grade based on multiple metrics.

Args:
    delta: Discretization margin.
    alpha: Purity index.
    kappa: Condition number.
    num_bragg_peaks: Number of detected Bragg peaks.

Returns:
    Dictionary containing:
        - grade: Grade classification string
        - description: Human-readable description
        - quality_score: Numerical quality score (0-1)
        - crystalline_features: Count of crystalline indicators
        - is_crystalline: Boolean indicating crystalline classification*
- `classify` (line 1879) `def classify(metrics, config)` - *Classify the thermodynamic phase based on computed metrics.

Args:
    metrics: Dictionary of computed metrics.
    config: Configuration containing phase thresholds.

Returns:
    Dictionary containing:
        - phase: Phase classification string
        - confidence: Classification confidence (0-1)
        - is_crystal: Boolean indicating crystalline phase*
- `__init__` (line 1948) `def __init__(self, config)` - *Initialize the checkpoint loader.

Args:
    config: Configuration containing architecture parameters.*
- `load` (line 1958) `def load(self, checkpoint_path)` - *Load a model from checkpoint file.

Args:
    checkpoint_path: Path to checkpoint file.

Returns:
    Loaded model or None if loading fails.*
- `extract_metadata` (line 1995) `def extract_metadata(self, checkpoint_path)` - *Extract metadata from checkpoint file.

Args:
    checkpoint_path: Path to checkpoint file.

Returns:
    Dictionary containing checkpoint metadata.*
- `categorize_weights` (line 2028) `def categorize_weights(self, state_dict)` - *Categorize weights by layer type for detailed analysis.

Args:
    state_dict: Model state dictionary.

Returns:
    Dictionary of categorized weight tensors.*
- `__init__` (line 2079) `def __init__(self, config)` - *Initialize the crystallography suite with all calculators.

Args:
    config: Configuration containing all analysis parameters.*
- `scan_directory` (line 2112) `def scan_directory(self, directory)` - *Scan directory for checkpoint files.

Args:
    directory: Path to checkpoint directory.

Returns:
    List of checkpoint file paths sorted by modification time.*
- `analyze_checkpoint` (line 2131) `def analyze_checkpoint(self, checkpoint_path, seed)` - *Perform comprehensive analysis on a single checkpoint.

Args:
    checkpoint_path: Path to checkpoint file.
    seed: Random seed for reproducible analysis.

Returns:
    Dictionary containing all computed metrics.*
- `run_full_analysis` (line 2215) `def run_full_analysis(self, directory, seed)` - *Run comprehensive analysis on all checkpoints in a directory.

Args:
    directory: Path to checkpoint directory.
    seed: Random seed for reproducible analysis.

Returns:
    List of analysis results for each checkpoint.*
- `generate_report` (line 2236) `def generate_report(self, results, output_path)` - *Generate JSON report from analysis results.

Args:
    results: List of analysis results.
    output_path: Path for output JSON file.*
- `generate_summary` (line 2249) `def generate_summary(self, results)` - *Generate aggregate summary from analysis results.

Args:
    results: List of analysis results.

Returns:
    Dictionary containing aggregate statistics.*
- `print_summary` (line 2322) `def print_summary(self, results)` - *Print formatted summary table to console.

Args:
    results: List of analysis results.*
- `__init__` (line 2359) `def __init__(self, config)` - *Initialize the batch analyzer.

Args:
    config: Configuration containing all analysis parameters.*
- `analyze_directory` (line 2372) `def analyze_directory(self, directory, seed)` - *Analyze all checkpoints in a directory with visualization.

Args:
    directory: Path to checkpoint directory.
    seed: Random seed for reproducible analysis.

Returns:
    Dictionary containing summary and individual results.*
- `_save_summary` (line 2404) `def _save_summary(self, summary, results)` - *Save analysis summary and individual reports.

Args:
    summary: Aggregate summary dictionary.
    results: List of individual analysis results.*
- `_generate_visualization` (line 2426) `def _generate_visualization(self, results)` - *Generate comprehensive visualization of analysis results.

Args:
    results: List of analysis results.*

#### `experiment2.py`
**Path:** `experiment2.py`

**Classes:**
- `Config` (line 22) `class Config`
- `SeedManager` (line 74) `class SeedManager`
- `LoggerFactory` (line 84) `class LoggerFactory`
- `IAnalysisStrategy` (line 99) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 105) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 111) `class HamiltonianOperator`
- `HamiltonianDataset` (line 133) `class HamiltonianDataset(Dataset)`
- `SpectralLayer` (line 183) `class SpectralLayer`
- `HamiltonianNeuralNetwork` (line 227) `class HamiltonianNeuralNetwork`
- `LocalComplexityAnalyzer` (line 257) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 273) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 302) `class CrystallographyMetricsCalculator(IMetricsCalculator)`
- `ThermodynamicMetricsCalculator` (line 737) `class ThermodynamicMetricsCalculator(IMetricsCalculator)`
- `SpectroscopyMetricsCalculator` (line 770) `class SpectroscopyMetricsCalculator(IMetricsCalculator)`
- `CheckpointManager` (line 804) `class CheckpointManager`
- `TrainingMetricsMonitor` (line 874) `class TrainingMetricsMonitor`
- `GlassStateDetector` (line 912) `class GlassStateDetector`
- `TrainingEngine` (line 973) `class TrainingEngine`
- `SeedMiningSystem` (line 1113) `class SeedMiningSystem`
- `SingleExperimentRunner` (line 1164) `class SingleExperimentRunner`
- `CheckpointAnalyzer` (line 1223) `class CheckpointAnalyzer`
- `Application` (line 1275) `class Application`

**Functions:**
- `main` (line 1334) `def main()`
- `set_seed` (line 76) `def set_seed(seed)`
- `create_logger` (line 86) `def create_logger(name, level)`
- `analyze` (line 101) `def analyze(self, model)`
- `compute` (line 107) `def compute(self, model)`
- `__init__` (line 112) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 116) `def _precompute_spectral_operators(self)`
- `apply` (line 122) `def apply(self, field)`
- `time_evolution` (line 127) `def time_evolution(self, field, dt)`
- `__init__` (line 134) `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- `__len__` (line 173) `def __len__(self)`
- `__getitem__` (line 176) `def __getitem__(self, idx)`
- `get_validation_batch` (line 179) `def get_validation_batch(self)`
- `__init__` (line 184) `def __init__(self, channels, grid_size)`
- `forward` (line 195) `def forward(self, x)`
- `__init__` (line 228) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 243) `def forward(self, x)`
- `compute_local_complexity` (line 259) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 275) `def compute_superposition(weights)`
- `compute` (line 303) `def compute(self, model, val_x, val_y)` - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311) `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin_from_state_dict` (line 348) `def compute_discretization_margin_from_state_dict(model)` - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) `def compute_discretization_margin(coeffs)` - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) `def compute_alpha_purity_from_model(model)` - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) `def compute_alpha_purity(coeffs)` - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) `def compute_kappa(model, val_x, val_y, num_batches)` - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) `def compute_kappa_quantum(model, hbar)` - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) `def _compute_crystallography_metrics(self, model, val_x, val_y)` - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) `def _check_weight_integrity(self, model)` - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) `def compute_poynting_vector(model)` - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) `def compute_all_metrics(model, val_x, val_y)` - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738) `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- `compute_effective_temperature` (line 747) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 760) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `compute` (line 771) `def compute(self, model)`
- `compute_weight_diffraction` (line 776) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 795) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 805) `def __init__(self, interval_minutes, max_checkpoints)`
- `should_save_checkpoint` (line 813) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 818) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `__init__` (line 875) `def __init__(self)`
- `update_metrics` (line 895) `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- `__init__` (line 913) `def __init__(self, patience_epochs)`
- `should_stop` (line 918) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 963) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `__init__` (line 974) `def __init__(self, model, optimizer, device, logger)`
- `train_epoch` (line 997) `def train_epoch(self, dataloader, epoch)`
- `validate` (line 1027) `def validate(self, val_x, val_y)`
- `compute_weight_metrics` (line 1040) `def compute_weight_metrics(self)`
- `execute_training` (line 1056) `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- `__init__` (line 1114) `def __init__(self, max_attempts)`
- `mine` (line 1118) `def mine(self)`
- `__init__` (line 1165) `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- `run` (line 1175) `def run(self)`
- `__init__` (line 1224) `def __init__(self, checkpoint_path, results_dir)`
- `analyze` (line 1230) `def analyze(self)`
- `__init__` (line 1276) `def __init__(self)`
- `_create_argument_parser` (line 1280) `def _create_argument_parser(self)`
- `run` (line 1294) `def run(self)`
- `safe_compute` (line 694) `def safe_compute(func)`

#### `main.py`
**Path:** `main.py`

**Classes:**
- `Config` (line 44) `class Config`
- `SeedManager` (line 153) `class SeedManager`
- `LoggerFactory` (line 165) `class LoggerFactory`
- `IAnalysisStrategy` (line 180) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 186) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 192) `class HamiltonianOperator`
- `SpectralLayer` (line 216) `class SpectralLayer`
- `HamiltonianBackbone` (line 254) `class HamiltonianBackbone`
- `HamiltonianInferenceEngine` (line 281) `class HamiltonianInferenceEngine`
- `SchrodingerPotentialGenerator` (line 339) `class SchrodingerPotentialGenerator`
- `SchrodingerDataset` (line 389) `class SchrodingerDataset(Dataset)`
- `SchrodingerSpectralNetwork` (line 508) `class SchrodingerSpectralNetwork`
- `LocalComplexityAnalyzer` (line 542) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 559) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 580) `class CrystallographyMetricsCalculator(IMetricsCalculator)`
- `ThermodynamicMetricsCalculator` (line 790) `class ThermodynamicMetricsCalculator(IMetricsCalculator)`
- `SpectroscopyMetricsCalculator` (line 846) `class SpectroscopyMetricsCalculator(IMetricsCalculator)`
- `LambdaPressureScheduler` (line 883) `class LambdaPressureScheduler`
- `AnnealingScheduler` (line 917) `class AnnealingScheduler`
- `TrainingMetricsMonitor` (line 947) `class TrainingMetricsMonitor`
- `CheckpointManager` (line 1050) `class CheckpointManager`
- `GlassStateDetector` (line 1102) `class GlassStateDetector`
- `WeightIntegrityChecker` (line 1158) `class WeightIntegrityChecker`
- `TrainingEngine` (line 1190) `class TrainingEngine`
- `BatchSizeProspector` (line 1335) `class BatchSizeProspector`
- `SeedMiner` (line 1395) `class SeedMiner`
- `FullTrainingOrchestrator` (line 1491) `class FullTrainingOrchestrator`
- `RefinementOrchestrator` (line 1619) `class RefinementOrchestrator`
- `ExperimentOrchestrator` (line 1737) `class ExperimentOrchestrator`

**Functions:**
- `build_argument_parser` (line 1847) `def build_argument_parser()`
- `main` (line 1914) `def main()`
- `set_seed` (line 155) `def set_seed(seed, device)`
- `create_logger` (line 167) `def create_logger(name, level)`
- `analyze` (line 182) `def analyze(self, model)`
- `compute` (line 188) `def compute(self, model)`
- `__init__` (line 193) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 197) `def _precompute_spectral_operators(self)`
- `apply` (line 203) `def apply(self, field)`
- `time_evolution` (line 208) `def time_evolution(self, field, dt)`
- `__init__` (line 217) `def __init__(self, channels, grid_size)`
- `forward` (line 228) `def forward(self, x)`
- `__init__` (line 255) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 270) `def forward(self, x)`
- `__init__` (line 282) `def __init__(self, config)`
- `_try_load_backbone` (line 289) `def _try_load_backbone(self)`
- `apply_hamiltonian` (line 323) `def apply_hamiltonian(self, field)`
- `time_evolve` (line 329) `def time_evolve(self, field, dt)`
- `__init__` (line 340) `def __init__(self, config)`
- `harmonic_potential` (line 344) `def harmonic_potential(self)`
- `double_well_potential` (line 352) `def double_well_potential(self)`
- `coulomb_like_potential` (line 360) `def coulomb_like_potential(self)`
- `periodic_lattice_potential` (line 368) `def periodic_lattice_potential(self)`
- `generate_mixed_potential` (line 374) `def generate_mixed_potential(self, seed)`
- `__init__` (line 390) `def __init__(self, config, hamiltonian_engine, seed)`
- `_solve_schrodinger_sample` (line 435) `def _solve_schrodinger_sample(self, potential, sample_seed)`
- `_time_evolve_wavefunction` (line 465) `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential, energy)`
- `__len__` (line 498) `def __len__(self)`
- `__getitem__` (line 501) `def __getitem__(self, idx)`
- `get_validation_batch` (line 504) `def get_validation_batch(self)`
- `__init__` (line 509) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, input_channels, output_channels)`
- `forward` (line 531) `def forward(self, x)`
- `compute_local_complexity` (line 544) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 561) `def compute_superposition(weights)`
- `__init__` (line 581) `def __init__(self, config)`
- `compute` (line 585) `def compute(self, model)`
- `compute_kappa` (line 590) `def compute_kappa(self, model, val_x, val_y, num_batches)`
- `compute_discretization_margin` (line 643) `def compute_discretization_margin(self, model)`
- `compute_alpha_purity` (line 651) `def compute_alpha_purity(self, model)`
- `compute_kappa_quantum` (line 657) `def compute_kappa_quantum(self, model)`
- `compute_poynting_vector` (line 678) `def compute_poynting_vector(self, model)`
- `compute_hbar_effective` (line 734) `def compute_hbar_effective(self, model, lambda_pressure)`
- `compute_all_metrics` (line 743) `def compute_all_metrics(self, model, val_x, val_y)`
- `__init__` (line 791) `def __init__(self, config)`
- `compute` (line 794) `def compute(self, model)`
- `compute_effective_temperature` (line 807) `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 831) `def compute_specific_heat(self, loss_history, temp_history)`
- `__init__` (line 847) `def __init__(self, config)`
- `compute` (line 850) `def compute(self, model)`
- `compute_weight_diffraction` (line 854) `def compute_weight_diffraction(self, coeffs)`
- `_compute_spectral_entropy` (line 874) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 884) `def __init__(self, config)`
- `current_lambda` (line 893) `def current_lambda(self)`
- `step` (line 896) `def step(self, epoch)`
- `compute_regularization_loss` (line 905) `def compute_regularization_loss(self, model)`
- `set_lambda` (line 913) `def set_lambda(self, value)`
- `__init__` (line 918) `def __init__(self, config)`
- `temperature` (line 926) `def temperature(self)`
- `step` (line 929) `def step(self)`
- `accept_perturbation` (line 935) `def accept_perturbation(self, delta_loss)`
- `should_restart` (line 943) `def should_restart(self, current_delta, best_delta)`
- `__init__` (line 948) `def __init__(self, config)`
- `update_metrics` (line 966) `def update_metrics(self)`
- `compute_delta_slope` (line 977) `def compute_delta_slope(self)`
- `format_progress_bar` (line 990) `def format_progress_bar(self, epoch, total_epochs, phase)`
- `__init__` (line 1051) `def __init__(self, config, checkpoint_dir)`
- `should_save_checkpoint` (line 1060) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 1065) `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- `__init__` (line 1103) `def __init__(self, config)`
- `should_stop` (line 1109) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 1144) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `check` (line 1160) `def check(model)`
- `__init__` (line 1191) `def __init__(self, config)`
- `compute_weight_metrics` (line 1201) `def compute_weight_metrics(self, model)`
- `compute_norm_conservation_error` (line 1216) `def compute_norm_conservation_error(self, model, val_x)`
- `train_single_epoch` (line 1229) `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler)`
- `validate` (line 1273) `def validate(self, model, val_x, val_y)`
- `collect_all_metrics` (line 1286) `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr)`
- `__init__` (line 1336) `def __init__(self, config, hamiltonian_engine)`
- `prospect` (line 1341) `def prospect(self)`
- `__init__` (line 1396) `def __init__(self, config, hamiltonian_engine, batch_size)`
- `mine` (line 1407) `def mine(self)`
- `__init__` (line 1492) `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- `run_phase3_training` (line 1505) `def run_phase3_training(self)`
- `__init__` (line 1620) `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- `run_phase4_refinement` (line 1639) `def run_phase4_refinement(self)`
- `__init__` (line 1738) `def __init__(self, config)`
- `run` (line 1742) `def run(self)`
- `_save_final_results` (line 1787) `def _save_final_results(self, model, monitor, seed, batch_size)`
- `safe_compute` (line 756) `def safe_compute(func)`
- `safe_get` (line 996) `def safe_get(key)`

#### `orbital_visualizer2.py`
**Path:** `orbital_visualizer2.py`

**Classes:**
- `Config` (line 44) `class Config`
- `WavefunctionCalculator` (line 83) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions.*
- `HamiltonianNNProcessor` (line 126) `class HamiltonianNNProcessor` - *Uses YOUR TRAINED MODEL for calculations.*
- `MonteCarloSampler` (line 160) `class MonteCarloSampler` - *Monte Carlo sampling for orbital visualization.*
- `OrbitalVisualizer` (line 265) `class OrbitalVisualizer` - *HIGH RESOLUTION visualization - NOT 16x16!*

**Functions:**
- `main` (line 433) `def main()`
- `radial_wavefunction` (line 87) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 97) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_on_grid` (line 107) `def psi_on_grid(n, l, m, grid_size)`
- `__init__` (line 129) `def __init__(self, engine)`
- `is_model_loaded` (line 133) `def is_model_loaded(self)`
- `compute_expected_energy` (line 136) `def compute_expected_energy(self, n, l, m)`
- `__init__` (line 163) `def __init__(self, hamiltonian_processor)`
- `find_max_probability` (line 166) `def find_max_probability(self, n, l, m)`
- `sample` (line 195) `def sample(self, n, l, m, num_samples)`
- `visualize` (line 268) `def visualize(self, data, save_path, hamiltonian_processor)`
- `_plotly` (line 396) `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`

#### `schrodinger_crystal_fixed.py`
**Path:** `schrodinger_crystal_fixed.py`

**Classes:**
- `Config` (line 53) `class Config`
- `IPhaseDetector` (line 212) `class IPhaseDetector(ABC)`
- `IMetricCalculator` (line 218) `class IMetricCalculator(ABC)`
- `SeedManager` (line 224) `class SeedManager`
- `LoggerFactory` (line 236) `class LoggerFactory`
- `HamiltonianOperator` (line 251) `class HamiltonianOperator`
- `SpectralLayer` (line 275) `class SpectralLayer`
- `HamiltonianBackbone` (line 313) `class HamiltonianBackbone`
- `HamiltonianInferenceEngine` (line 340) `class HamiltonianInferenceEngine`
- `SchrodingerPotentialGenerator` (line 398) `class SchrodingerPotentialGenerator`
- `SchrodingerDataset` (line 448) `class SchrodingerDataset(Dataset)`
- `SchrodingerSpectralNetwork` (line 567) `class SchrodingerSpectralNetwork`
- `FullFourierAnalyzer` (line 601) `class FullFourierAnalyzer` - *Complete 2D Fourier Transform analysis for resonance detection.
Implements full spectral analysis including power spectrum density,
phase coherence, and harmonic ratio detection for crystalline structure.*
- `FourierMassCenterAnalyzer` (line 793) `class FourierMassCenterAnalyzer` - *Analyzes center of mass in the 2D torus Fourier space.
Detects spatial alignments indicating liquid -> crystal transition.
Enhanced with full 2D FFT integration.*
- `TopologicalPhaseDetector` (line 873) `class TopologicalPhaseDetector(IPhaseDetector)` - *Detects topological phase transition (liquid -> crystal) using
Fourier mass center analysis with hysteresis and full spectral integration.*
- `SpectralFieldExtractor` (line 946) `class SpectralFieldExtractor`
- `TopologicalCrystallizationLoss` (line 968) `class TopologicalCrystallizationLoss`
- `CrystallizationPressureApplicator` (line 1006) `class CrystallizationPressureApplicator`
- `TopologicalMetricsCalculator` (line 1021) `class TopologicalMetricsCalculator(IMetricCalculator)`
- `LocalComplexityAnalyzer` (line 1084) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 1101) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 1122) `class CrystallographyMetricsCalculator(IMetricCalculator)`
- `ThermodynamicMetricsCalculator` (line 1332) `class ThermodynamicMetricsCalculator(IMetricCalculator)`
- `SpectralGeometryCalculator` (line 1412) `class SpectralGeometryCalculator(IMetricCalculator)`
- `RicciCurvatureCalculator` (line 1465) `class RicciCurvatureCalculator(IMetricCalculator)`
- `SpectroscopyMetricsCalculator` (line 1509) `class SpectroscopyMetricsCalculator(IMetricCalculator)`
- `LambdaPressureScheduler` (line 1546) `class LambdaPressureScheduler`
- `AdaptiveLambdaScheduler` (line 1580) `class AdaptiveLambdaScheduler(LambdaPressureScheduler)`
- `QuadruplePrecisionLambdaScheduler` (line 1601) `class QuadruplePrecisionLambdaScheduler` - *Lambda scheduler using quadruple precision (float128) for Phase 5.
Provides extreme precision for crystallization pressure.*
- `AnnealingScheduler` (line 1639) `class AnnealingScheduler`
- `TopologicalAnnealingScheduler` (line 1669) `class TopologicalAnnealingScheduler(AnnealingScheduler)`
- `TrainingMetricsMonitor` (line 1688) `class TrainingMetricsMonitor`
- `CheckpointManager` (line 1844) `class CheckpointManager`
- `Phase5CheckpointManager` (line 1902) `class Phase5CheckpointManager` - *Specialized checkpoint manager for Phase 5 with quadruple precision.
Only overwrites latest.pth when new checkpoint is better (higher accuracy).*
- `GlassStateDetector` (line 2011) `class GlassStateDetector`
- `WeightIntegrityChecker` (line 2067) `class WeightIntegrityChecker`
- `TrainingEngine` (line 2099) `class TrainingEngine`
- `BatchSizeProspector` (line 2284) `class BatchSizeProspector`
- `SeedMiner` (line 2355) `class SeedMiner`
- `FullTrainingOrchestrator` (line 2496) `class FullTrainingOrchestrator`
- `RefinementOrchestrator` (line 2628) `class RefinementOrchestrator`
- `Phase5Orchestrator` (line 2757) `class Phase5Orchestrator` - *Phase 5: Quadruple precision (float128) high-pressure crystallization.
Uses extreme lambda pressure with thermal injection for final crystallization.*
- `ExperimentOrchestrator` (line 2910) `class ExperimentOrchestrator`

**Functions:**
- `build_argument_parser` (line 3065) `def build_argument_parser()`
- `main` (line 3160) `def main()`
- `detect` (line 214) `def detect(self, spectral_field)`
- `compute` (line 220) `def compute(self, model)`
- `set_seed` (line 226) `def set_seed(seed, device)`
- `create_logger` (line 238) `def create_logger(name, level)`
- `__init__` (line 252) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 256) `def _precompute_spectral_operators(self)`
- `apply` (line 262) `def apply(self, field)`
- `time_evolution` (line 267) `def time_evolution(self, field, dt)`
- `__init__` (line 276) `def __init__(self, channels, grid_size)`
- `forward` (line 287) `def forward(self, x)`
- `__init__` (line 314) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 329) `def forward(self, x)`
- `__init__` (line 341) `def __init__(self, config)`
- `_try_load_backbone` (line 348) `def _try_load_backbone(self)`
- `apply_hamiltonian` (line 382) `def apply_hamiltonian(self, field)`
- `time_evolve` (line 388) `def time_evolve(self, field, dt)`
- `__init__` (line 399) `def __init__(self, config)`
- `harmonic_potential` (line 403) `def harmonic_potential(self)`
- `double_well_potential` (line 411) `def double_well_potential(self)`
- `coulomb_like_potential` (line 419) `def coulomb_like_potential(self)`
- `periodic_lattice_potential` (line 427) `def periodic_lattice_potential(self)`
- `generate_mixed_potential` (line 433) `def generate_mixed_potential(self, seed)`
- `__init__` (line 449) `def __init__(self, config, hamiltonian_engine, seed)`
- `_solve_schrodinger_sample` (line 494) `def _solve_schrodinger_sample(self, potential, sample_seed)`
- `_time_evolve_wavefunction` (line 524) `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential, energy)`
- `__len__` (line 557) `def __len__(self)`
- `__getitem__` (line 560) `def __getitem__(self, idx)`
- `get_validation_batch` (line 563) `def get_validation_batch(self)`
- `__init__` (line 568) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, input_channels, output_channels)`
- `forward` (line 590) `def forward(self, x)`
- `__init__` (line 607) `def __init__(self, config)`
- `compute_full_spectrum` (line 615) `def compute_full_spectrum(self, spectral_field)` - *Compute complete 2D Fourier spectrum with phase and magnitude analysis.*
- `detect_bragg_peaks` (line 692) `def detect_bragg_peaks(self, power_spectrum, threshold_sigma)` - *Detect Bragg peaks in power spectrum for crystalline structure identification.*
- `compute_resonance_metrics` (line 749) `def compute_resonance_metrics(self, spectral_field)` - *Compute resonance metrics for crystallization detection.*
- `__init__` (line 799) `def __init__(self, config)`
- `compute_mass_center` (line 807) `def compute_mass_center(self, spectral_field)` - *Compute center of mass of weight spectrum on the torus.
Integrates with full Fourier analysis for comprehensive detection.*
- `__init__` (line 878) `def __init__(self, config)`
- `detect` (line 885) `def detect(self, spectral_field)`
- `extract` (line 948) `def extract(model, grid_size)`
- `__init__` (line 969) `def __init__(self, config)`
- `forward` (line 974) `def forward(self, phase_info, epoch)`
- `__init__` (line 1007) `def __init__(self, config)`
- `apply` (line 1011) `def apply(self, model, phase_info)`
- `__init__` (line 1022) `def __init__(self, config)`
- `compute` (line 1029) `def compute(self, model)`
- `apply_crystallization_pressure` (line 1062) `def apply_crystallization_pressure(self, model, topo_metrics)`
- `_empty_metrics` (line 1068) `def _empty_metrics()`
- `compute_local_complexity` (line 1086) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 1103) `def compute_superposition(weights)`
- `__init__` (line 1123) `def __init__(self, config)`
- `compute` (line 1127) `def compute(self, model)`
- `compute_kappa` (line 1132) `def compute_kappa(self, model, val_x, val_y, num_batches)`
- `compute_discretization_margin` (line 1185) `def compute_discretization_margin(self, model)`
- `compute_alpha_purity` (line 1193) `def compute_alpha_purity(self, model)`
- `compute_kappa_quantum` (line 1199) `def compute_kappa_quantum(self, model)`
- `compute_poynting_vector` (line 1220) `def compute_poynting_vector(self, model)`
- `compute_hbar_effective` (line 1276) `def compute_hbar_effective(self, model, lambda_pressure)`
- `compute_all_metrics` (line 1285) `def compute_all_metrics(self, model, val_x, val_y)`
- `__init__` (line 1333) `def __init__(self, config)`
- `compute` (line 1336) `def compute(self, model)`
- `compute_effective_temperature` (line 1363) `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 1387) `def compute_specific_heat(self, loss_history, temp_history)`
- `compute_gibbs_free_energy` (line 1401) `def compute_gibbs_free_energy(self, delta, alpha, temperature)`
- `compute_critical_temperature` (line 1408) `def compute_critical_temperature(self, alpha)`
- `__init__` (line 1413) `def __init__(self, config)`
- `compute` (line 1416) `def compute(self, model)`
- `_compute_level_spacing_ratio` (line 1453) `def _compute_level_spacing_ratio(self, spacings)`
- `__init__` (line 1466) `def __init__(self, config)`
- `compute` (line 1469) `def compute(self, model)`
- `_compute_ricci_scalar` (line 1488) `def _compute_ricci_scalar(self, metric)`
- `_estimate_sectional_curvatures` (line 1496) `def _estimate_sectional_curvatures(self, metric)`
- `__init__` (line 1510) `def __init__(self, config)`
- `compute` (line 1513) `def compute(self, model)`
- `compute_weight_diffraction` (line 1517) `def compute_weight_diffraction(self, coeffs)`
- `_compute_spectral_entropy` (line 1537) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 1547) `def __init__(self, config)`
- `current_lambda` (line 1556) `def current_lambda(self)`
- `step` (line 1559) `def step(self, epoch)`
- `compute_regularization_loss` (line 1568) `def compute_regularization_loss(self, model)`
- `set_lambda` (line 1576) `def set_lambda(self, value)`
- `__init__` (line 1581) `def __init__(self, config)`
- `step_adaptive` (line 1586) `def step_adaptive(self, epoch, topo_phase_state)`
- `__init__` (line 1606) `def __init__(self, config)`
- `current_lambda` (line 1615) `def current_lambda(self)`
- `step` (line 1618) `def step(self, epoch, improvement)`
- `compute_regularization_loss` (line 1627) `def compute_regularization_loss(self, model)`
- `set_lambda` (line 1635) `def set_lambda(self, value)`
- `__init__` (line 1640) `def __init__(self, config)`
- `temperature` (line 1648) `def temperature(self)`
- `step` (line 1651) `def step(self)`
- `accept_perturbation` (line 1657) `def accept_perturbation(self, delta_loss)`
- `should_restart` (line 1665) `def should_restart(self, current_delta, best_delta)`
- `__init__` (line 1670) `def __init__(self, config)`
- `step_adaptive` (line 1674) `def step_adaptive(self, alignment_trend, resonance_score)`
- `__init__` (line 1689) `def __init__(self, config)`
- `update_metrics` (line 1719) `def update_metrics(self)`
- `compute_delta_slope` (line 1730) `def compute_delta_slope(self)`
- `format_progress_bar` (line 1743) `def format_progress_bar(self, epoch, total_epochs, phase)`
- `__init__` (line 1845) `def __init__(self, config, checkpoint_dir)`
- `should_save_checkpoint` (line 1854) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 1859) `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- `load_latest_checkpoint` (line 1895) `def load_latest_checkpoint(self)`
- `__init__` (line 1907) `def __init__(self, config)`
- `_load_best_metrics` (line 1918) `def _load_best_metrics(self)`
- `should_save` (line 1937) `def should_save(self, current_delta, current_alpha, current_acc)`
- `save_checkpoint` (line 1948) `def save_checkpoint(self, model, optimizer, epoch, metrics, lambda_value)`
- `load_checkpoint` (line 1992) `def load_checkpoint(self, model, optimizer)`
- `__init__` (line 2012) `def __init__(self, config)`
- `should_stop` (line 2018) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 2053) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `check` (line 2069) `def check(model)`
- `__init__` (line 2100) `def __init__(self, config)`
- `compute_weight_metrics` (line 2113) `def compute_weight_metrics(self, model)`
- `compute_norm_conservation_error` (line 2128) `def compute_norm_conservation_error(self, model, val_x)`
- `train_single_epoch` (line 2141) `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler)`
- `validate` (line 2185) `def validate(self, model, val_x, val_y)`
- `collect_all_metrics` (line 2198) `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr, epoch)`
- `__init__` (line 2285) `def __init__(self, config, hamiltonian_engine)`
- `prospect` (line 2290) `def prospect(self)`
- `__init__` (line 2356) `def __init__(self, config, hamiltonian_engine, batch_size)`
- `mine` (line 2367) `def mine(self)`
- `__init__` (line 2497) `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- `run_phase3_training` (line 2510) `def run_phase3_training(self)`
- `__init__` (line 2629) `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- `run_phase4_refinement` (line 2648) `def run_phase4_refinement(self)`
- `__init__` (line 2762) `def __init__(self, config, hamiltonian_engine, model, monitor, seed, batch_size)`
- `run_phase5_crystallization` (line 2780) `def run_phase5_crystallization(self)`
- `__init__` (line 2911) `def __init__(self, config)`
- `run` (line 2915) `def run(self)`
- `_save_final_results` (line 2990) `def _save_final_results(self, model, monitor, seed, batch_size)`
- `safe_compute` (line 1298) `def safe_compute(func)`
- `safe_get` (line 1747) `def safe_get(key)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
