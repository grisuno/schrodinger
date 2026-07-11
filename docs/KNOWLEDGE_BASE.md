# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 8 | **Total Symbols Extracted:** 528 | **Total Imports:** 104

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
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

**Classs:**
- `BerryPhaseResult` (line 22) - *Results from Berry phase calculation.*
- `BerryPhaseCalculator` (line 37) - *Calculates Berry phase from training checkpoint trajectory.

Uses the spectral kernel parameters as the parameter space θ,
and computes the geometric phase accumulated during training.*

**Functions:**
- `visualize_results` (line 388) - *Create visualization of Berry phase results.*
- `main` (line 514)
- `__init__` (line 45)
- `load_checkpoints` (line 48) - *Load all checkpoints from directory in chronological order.*
- `_extract_epoch` (line 69) - *Extract epoch number from checkpoint filename.*
- `extract_spectral_kernels` (line 74) - *Extract spectral kernels from model state dict.
Returns dict with 'real' and 'imag' kernels per layer.*
- `flatten_kernel_params` (line 107) - *Flatten all kernel parameters into a single complex vector.*
- `compute_spectral_density` (line 119) - *Compute spectral density |W(k)|².*
- `compute_center_of_mass` (line 123) - *Compute center of mass in 2D Fourier space.

For each kernel layer [C_out, C_in, freq_h, freq_w]:
- Map frequency indices to angular coordinates
- Compute weighted CM*
- `compute_berry_connection_discrete` (line 157) - *Compute discrete Berry connection between two parameter states.

Uses the formula for discrete Berry phase:
A = Im[log(⟨ψ(θ_{n-1})|ψ(θ_n)⟩)]

For parameter vectors, this is:
A = Im[log(θ_{n-1}^* · θ_n)]*
- `compute_eigenvalue_spectrum` (line 188) - *Compute eigenvalue spectrum of kernel Gram matrix.*
- `compute_eigenvalue_gap` (line 209) - *Compute gap between two largest eigenvalues.*
- `compute_trajectory_metrics` (line 215) - *Compute trajectory metrics in parameter space.*
- `calculate_berry_phase` (line 248) - *Main method to calculate Berry phase from checkpoint directory.*
- `calculate_from_final_checkpoint` (line 334) - *Calculate Berry phase estimates from final checkpoint metrics history.*

#### `crystallographer.py`
**Path:** `crystallographer.py`

**Classs:**
- `SchrodingerCrystallographyConfig` (line 61) - *Comprehensive configuration for Schrodinger crystallographic analysis.
All parameters are centralized here following the Single Responsibility Principle.
No magic numbers or hardcoded values appear elsewhere in the codebase.*
- `LoggerFactory` (line 152) - *Factory for creating configured logger instances.
Follows the Single Responsibility Principle for logging configuration.*
- `IMetricCalculator` (line 182) - *Interface for metric calculation strategies.
Follows the Interface Segregation Principle by defining a minimal contract.*
- `HamiltonianOperator` (line 203) - *Analytical Hamiltonian operator for fallback computation.
Implements spectral operators for Laplacian computation in Fourier space.*
- `SpectralLayer` (line 262) - *Spectral convolution layer operating in Fourier space with complex kernels.
Implements learnable frequency-domain transformations.*
- `HamiltonianBackbone` (line 316) - *Neural network backbone for Hamiltonian inference.
Learns to approximate Hamiltonian operations from data.*
- `SchrodingerSpectralNetwork` (line 358) - *Schrodinger equation neural network with spectral layers.
Implements expansion-contraction architecture with Fourier convolutions.*
- `HamiltonianInferenceEngine` (line 404) - *Engine for Hamiltonian inference using either a pretrained backbone
or analytical operators as fallback.*
- `SchrodingerPotentialGenerator` (line 490) - *Generator for various potential energy landscapes used in Schrodinger equation.
Supports harmonic, double-well, Coulomb-like, and periodic lattice potentials.*
- `SyntheticDataGenerator` (line 584) - *Generator for synthetic Schrodinger equation training data.
Creates initial and target wavefunction pairs for various potentials.*
- `WeightIntegrityCalculator` (line 709) - *Calculator for weight integrity metrics including NaN and Inf detection.*
- `DiscretizationCalculator` (line 772) - *Calculator for discretization margin and alpha purity metrics.
These metrics quantify how close weights are to integer values.*
- `LocalComplexityCalculator` (line 852) - *Calculator for local complexity metrics measuring weight diversity.*
- `SuperpositionCalculator` (line 911) - *Calculator for superposition metrics measuring weight correlations.*
- `GradientDynamicsCalculator` (line 974) - *Calculator for gradient-based metrics including condition number and effective temperature.*
- `SpectralGeometryCalculator` (line 1092) - *Calculator for spectral geometry metrics including MBL level spacing.*
- `RicciCurvatureCalculator` (line 1182) - *Calculator for Ricci curvature estimation in weight space.*
- `ThermodynamicCalculator` (line 1266) - *Calculator for thermodynamic potentials including Gibbs free energy.*
- `KappaQuantumCalculator` (line 1346) - *Calculator for quantum condition number of the weight covariance.*
- `PoyntingVectorCalculator` (line 1397) - *Calculator for Poynting vector magnitude representing energy flow.*
- `HbarEffectiveCalculator` (line 1490) - *Calculator for effective Planck constant under lambda pressure.*
- `WeightDiffractionCalculator` (line 1529) - *Calculator for weight diffraction analysis with advanced Bragg peak detection.*
- `PhaseStructureCalculator` (line 1620) - *Calculator for phase structure analysis using histogram-based methods.*
- `ComplexKernelHolomorphyCalculator` (line 1679) - *Calculator for analyzing holomorphy of complex spectral kernels using Cauchy-Riemann equations.*
- `NormConservationCalculator` (line 1762) - *Calculator for norm conservation error in Schrodinger dynamics.*
- `CrystallographicGrader` (line 1800) - *Advanced grader for crystallographic quality assessment.
Implements refined threshold-based grading system.*
- `PhaseClassifier` (line 1873) - *Classifier for thermodynamic phase identification.*
- `CheckpointLoader` (line 1943) - *Loader for neural network checkpoints with architecture reconstruction.*
- `DefinitiveCrystallographySuite` (line 2073) - *Comprehensive crystallographic analysis suite combining all metric calculators.
Follows the Single Responsibility Principle for orchestration.*
- `BatchCrystallographyAnalyzer` (line 2354) - *Batch analysis orchestrator with visualization generation.*

**Functions:**
- `build_argument_parser` (line 2551) - *Build the command-line argument parser.

Returns:
    Configured ArgumentParser instance.*
- `main` (line 2611) - *Main entry point for the definitive crystallographer.*
- `create_logger` (line 159) - *Create and configure a logger with standardized formatting.

Args:
    name: Identifier for the logger instance.
    level: Logging level string (DEBUG, INFO, WARNING, ERROR).

Returns:
    Configured Logger instance ready for use.*
- `compute` (line 189) - *Compute metrics for the given model.

Args:
    model: Neural network model to analyze.
    **kwargs: Additional parameters required for computation.

Returns:
    Dictionary containing computed metric values.*
- `__init__` (line 209) - *Initialize the Hamiltonian operator with precomputed spectral operators.

Args:
    config: Configuration containing grid size and other parameters.*
- `_precompute_spectral_operators` (line 220) - *Precompute the Laplacian spectrum in Fourier space for efficient application.*
- `apply` (line 229) - *Apply the Hamiltonian operator to a field using spectral methods.

Args:
    field: Input field tensor (2D or batched).

Returns:
    Transformed field after Hamiltonian application.*
- `time_evolution` (line 243) - *Perform time evolution of the field under the Hamiltonian.

Args:
    field: Input field tensor.
    dt: Time step size.

Returns:
    Time-evolved field with preserved norm.*
- `__init__` (line 268) - *Initialize spectral layer with complex-valued kernels.

Args:
    channels: Number of input/output channels.
    grid_size: Spatial dimension of the input grid.
    config: Configuration object (optional for parameter access).*
- `forward` (line 287) - *Apply spectral convolution in Fourier domain.

Args:
    x: Input tensor of shape (batch, channels, height, width).

Returns:
    Transformed tensor after spectral convolution.*
- `__init__` (line 322) - *Initialize the Hamiltonian backbone network.

Args:
    config: Configuration containing architecture parameters.*
- `forward` (line 338) - *Forward pass through the Hamiltonian backbone.

Args:
    x: Input tensor of shape (batch, height, width) or (batch, 1, height, width).

Returns:
    Hamiltonian-transformed output tensor.*
- `__init__` (line 364) - *Initialize the Schrodinger spectral network.

Args:
    config: Configuration containing all architecture parameters.*
- `forward` (line 384) - *Forward pass through the Schrodinger network.

Args:
    x: Input tensor of shape (batch, channels, height, width).

Returns:
    Output tensor after expansion, spectral processing, and contraction.*
- `__init__` (line 410) - *Initialize the Hamiltonian inference engine.

Args:
    config: Configuration containing backbone path and device settings.*
- `_try_load_backbone` (line 423) - *Attempt to load a pretrained backbone for Hamiltonian inference.
Falls back to analytical operator if backbone is unavailable.*
- `apply_hamiltonian` (line 454) - *Apply the Hamiltonian to a field using backbone or analytical operator.

Args:
    field: Input field tensor.

Returns:
    Hamiltonian-transformed field.*
- `time_evolve` (line 469) - *Perform time evolution using backbone or analytical operator.

Args:
    field: Input field tensor.
    dt: Time step size.

Returns:
    Time-evolved field with preserved norm.*
- `__init__` (line 496) - *Initialize the potential generator.

Args:
    config: Configuration containing potential parameters.*
- `harmonic_potential` (line 506) - *Generate a harmonic oscillator potential.

Returns:
    2D tensor with harmonic potential centered at grid center.*
- `double_well_potential` (line 520) - *Generate a double-well potential.

Returns:
    2D tensor with double-well potential along x-axis.*
- `coulomb_like_potential` (line 534) - *Generate a Coulomb-like central potential.

Returns:
    2D tensor with Coulomb potential centered at grid center.*
- `periodic_lattice_potential` (line 548) - *Generate a periodic lattice potential.

Returns:
    2D tensor with periodic cosine potential.*
- `generate_mixed_potential` (line 560) - *Generate a mixed potential combining multiple potential types.

Args:
    seed: Random seed for determining mixture weights.

Returns:
    2D tensor with weighted combination of potential types.*
- `__init__` (line 590) - *Initialize the synthetic data generator.

Args:
    config: Configuration containing data generation parameters.
    hamiltonian_engine: Engine for Hamiltonian operations.*
- `generate_batch` (line 603) - *Generate a batch of validation data.

Args:
    seed: Random seed for reproducibility.

Returns:
    Tuple of (initial_states, target_states) tensors.*
- `_solve_schrodinger_sample` (line 633) - *Solve the Schrodinger equation for a single sample.

Args:
    potential: Potential energy landscape.
    sample_seed: Random seed for eigenstate selection.

Returns:
    Tuple of (real_part, imaginary_part, energy) for the wavefunction.*
- `_time_evolve_wavefunction` (line 673) - *Time evolve a wavefunction under the given potential.

Args:
    psi_real: Real part of the wavefunction.
    psi_imag: Imaginary part of the wavefunction.
    potential: Potential energy landscape.

Returns:
    Tuple of (evolved_real, evolved_imag) wavefunction components.*
- `__init__` (line 714) - *Initialize the weight integrity calculator.

Args:
    config: Configuration containing tolerance parameters.*
- `compute` (line 723) - *Compute weight integrity metrics for the model.

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
- `__init__` (line 778) - *Initialize the discretization calculator.

Args:
    config: Configuration containing discretization thresholds.*
- `compute` (line 787) - *Compute discretization metrics for the model.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - delta: Maximum discretization margin
        - alpha: Purity index (-log(delta))
        - spectral_entropy: Entropy of weight power spectrum
        - is_discrete: Boolean indicating discretization below threshold
        - layer_deltas: Per-layer discretization margins*
- `_compute_spectral_entropy` (line 828) - *Compute the spectral entropy of the weight distribution.

Args:
    weights: Flattened weight tensor.

Returns:
    Spectral entropy value.*
- `__init__` (line 857) - *Initialize the local complexity calculator.

Args:
    config: Configuration containing dimension limits.*
- `compute` (line 866) - *Compute local complexity for the model weights.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing local_complexity value.*
- `_compute_local_complexity` (line 887) - *Compute local complexity for a weight matrix.

Args:
    weights: 2D weight tensor.

Returns:
    Local complexity value between 0 and 1.*
- `__init__` (line 916) - *Initialize the superposition calculator.

Args:
    config: Configuration containing computation parameters.*
- `compute` (line 925) - *Compute superposition metrics for the model.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing superposition value.*
- `_compute_superposition` (line 946) - *Compute superposition metric for a weight matrix.

Args:
    weights: 2D weight tensor.

Returns:
    Superposition value indicating average correlation.*
- `__init__` (line 979) - *Initialize the gradient dynamics calculator.

Args:
    config: Configuration containing gradient computation parameters.*
- `compute` (line 988) - *Compute gradient dynamics metrics.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain 'val_x' and 'val_y' tensors.

Returns:
    Dictionary containing:
        - kappa: Condition number of gradient covariance
        - effective_temperature: Temperature derived from gradient variance
        - gradient_variance: Variance of gradient samples*
- `__init__` (line 1097) - *Initialize the spectral geometry calculator.

Args:
    config: Configuration containing spectral analysis parameters.*
- `compute` (line 1106) - *Compute spectral geometry metrics.

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
- `_compute_level_spacing_ratio` (line 1161) - *Compute the level spacing ratio for MBL analysis.

Args:
    spacings: Array of eigenvalue level spacings.

Returns:
    Average level spacing ratio.*
- `__init__` (line 1187) - *Initialize the Ricci curvature calculator.

Args:
    config: Configuration containing curvature estimation parameters.*
- `compute` (line 1196) - *Compute Ricci curvature metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - ricci_scalar: Estimated Ricci scalar curvature
        - mean_sectional_curvature: Average sectional curvature
        - curvature_variance: Variance of sectional curvatures*
- `_compute_ricci_scalar` (line 1225) - *Compute the Ricci scalar from the metric tensor.

Args:
    metric: Metric tensor as numpy array.

Returns:
    Estimated Ricci scalar.*
- `_estimate_sectional_curvatures` (line 1243) - *Estimate sectional curvatures by sampling 2D sections.

Args:
    metric: Metric tensor as numpy array.
    samples: Number of sectional curvature samples.

Returns:
    Array of estimated sectional curvatures.*
- `__init__` (line 1271) - *Initialize the thermodynamic calculator.

Args:
    config: Configuration containing thermodynamic parameters.*
- `compute` (line 1280) - *Compute thermodynamic metrics.

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
- `_classify_phase` (line 1320) - *Classify the thermodynamic phase based on metrics.

Args:
    delta: Discretization margin.
    kappa: Condition number.
    temp: Effective temperature.
    alpha: Purity index.

Returns:
    Phase classification string.*
- `__init__` (line 1351) - *Initialize the kappa quantum calculator.

Args:
    config: Configuration containing quantum parameters.*
- `compute` (line 1360) - *Compute quantum condition number.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing kappa_quantum value.*
- `__init__` (line 1402) - *Initialize the Poynting vector calculator.

Args:
    config: Configuration containing energy flow parameters.*
- `compute` (line 1411) - *Compute Poynting vector metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - poynting_magnitude: Magnitude of energy flow
        - is_radiating: Boolean indicating significant energy flow
        - field_orthogonality: Measure of field orthogonality
        - energy_distribution: Dictionary of energy distribution metrics*
- `__init__` (line 1495) - *Initialize the hbar effective calculator.

Args:
    config: Configuration containing physical constants.*
- `compute` (line 1504) - *Compute effective hbar.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain delta and lambda_pressure.

Returns:
    Dictionary containing hbar_effective value.*
- `__init__` (line 1534) - *Initialize the weight diffraction calculator.

Args:
    config: Configuration containing spectral analysis parameters.*
- `compute` (line 1543) - *Compute weight diffraction metrics with Bragg peak detection.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - bragg_peaks: List of detected Bragg peaks
        - is_crystalline_structure: Boolean indicating crystalline pattern
        - spectral_entropy: Entropy of power spectrum
        - num_peaks: Number of detected peaks*
- `_compute_spectral_entropy` (line 1602) - *Compute spectral entropy of the power spectrum.

Args:
    power_spectrum: Power spectrum tensor.

Returns:
    Spectral entropy value.*
- `__init__` (line 1625) - *Initialize the phase structure calculator.

Args:
    config: Configuration containing histogram parameters.*
- `compute` (line 1634) - *Compute phase structure metrics.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - per_layer: Per-layer phase classifications
        - distribution: Count of each phase type
        - dominant_phase: Most common phase type*
- `__init__` (line 1684) - *Initialize the holomorphy calculator.

Args:
    config: Configuration containing holomorphy threshold.*
- `compute` (line 1693) - *Compute holomorphy metrics for complex kernels.

Args:
    model: Neural network model to analyze.

Returns:
    Dictionary containing:
        - per_layer: Per-layer holomorphy analysis
        - holomorphic_fraction: Fraction of holomorphic layers
        - average_cr_error: Average Cauchy-Riemann error*
- `__init__` (line 1767) - *Initialize the norm conservation calculator.

Args:
    config: Configuration containing normalization parameters.*
- `compute` (line 1776) - *Compute norm conservation error.

Args:
    model: Neural network model to analyze.
    **kwargs: Must contain val_x tensor.

Returns:
    Dictionary containing norm_conservation_error value.*
- `__init__` (line 1806) - *Initialize the crystallographic grader.

Args:
    config: Configuration containing grading thresholds.*
- `assign_grade` (line 1815) - *Assign a crystallographic grade based on multiple metrics.

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
- `classify` (line 1879) - *Classify the thermodynamic phase based on computed metrics.

Args:
    metrics: Dictionary of computed metrics.
    config: Configuration containing phase thresholds.

Returns:
    Dictionary containing:
        - phase: Phase classification string
        - confidence: Classification confidence (0-1)
        - is_crystal: Boolean indicating crystalline phase*
- `__init__` (line 1948) - *Initialize the checkpoint loader.

Args:
    config: Configuration containing architecture parameters.*
- `load` (line 1958) - *Load a model from checkpoint file.

Args:
    checkpoint_path: Path to checkpoint file.

Returns:
    Loaded model or None if loading fails.*
- `extract_metadata` (line 1995) - *Extract metadata from checkpoint file.

Args:
    checkpoint_path: Path to checkpoint file.

Returns:
    Dictionary containing checkpoint metadata.*
- `categorize_weights` (line 2028) - *Categorize weights by layer type for detailed analysis.

Args:
    state_dict: Model state dictionary.

Returns:
    Dictionary of categorized weight tensors.*
- `__init__` (line 2079) - *Initialize the crystallography suite with all calculators.

Args:
    config: Configuration containing all analysis parameters.*
- `scan_directory` (line 2112) - *Scan directory for checkpoint files.

Args:
    directory: Path to checkpoint directory.

Returns:
    List of checkpoint file paths sorted by modification time.*
- `analyze_checkpoint` (line 2131) - *Perform comprehensive analysis on a single checkpoint.

Args:
    checkpoint_path: Path to checkpoint file.
    seed: Random seed for reproducible analysis.

Returns:
    Dictionary containing all computed metrics.*
- `run_full_analysis` (line 2215) - *Run comprehensive analysis on all checkpoints in a directory.

Args:
    directory: Path to checkpoint directory.
    seed: Random seed for reproducible analysis.

Returns:
    List of analysis results for each checkpoint.*
- `generate_report` (line 2236) - *Generate JSON report from analysis results.

Args:
    results: List of analysis results.
    output_path: Path for output JSON file.*
- `generate_summary` (line 2249) - *Generate aggregate summary from analysis results.

Args:
    results: List of analysis results.

Returns:
    Dictionary containing aggregate statistics.*
- `print_summary` (line 2322) - *Print formatted summary table to console.

Args:
    results: List of analysis results.*
- `__init__` (line 2359) - *Initialize the batch analyzer.

Args:
    config: Configuration containing all analysis parameters.*
- `analyze_directory` (line 2372) - *Analyze all checkpoints in a directory with visualization.

Args:
    directory: Path to checkpoint directory.
    seed: Random seed for reproducible analysis.

Returns:
    Dictionary containing summary and individual results.*
- `_save_summary` (line 2404) - *Save analysis summary and individual reports.

Args:
    summary: Aggregate summary dictionary.
    results: List of individual analysis results.*
- `_generate_visualization` (line 2426) - *Generate comprehensive visualization of analysis results.

Args:
    results: List of analysis results.*

#### `experiment2.py`
**Path:** `experiment2.py`

**Classs:**
- `Config` (line 22)
- `SeedManager` (line 74)
- `LoggerFactory` (line 84)
- `IAnalysisStrategy` (line 99)
- `IMetricsCalculator` (line 105)
- `HamiltonianOperator` (line 111)
- `HamiltonianDataset` (line 133)
- `SpectralLayer` (line 183)
- `HamiltonianNeuralNetwork` (line 227)
- `LocalComplexityAnalyzer` (line 257)
- `SuperpositionAnalyzer` (line 273)
- `CrystallographyMetricsCalculator` (line 302)
- `ThermodynamicMetricsCalculator` (line 737)
- `SpectroscopyMetricsCalculator` (line 770)
- `CheckpointManager` (line 804)
- `TrainingMetricsMonitor` (line 874)
- `GlassStateDetector` (line 912)
- `TrainingEngine` (line 973)
- `SeedMiningSystem` (line 1113)
- `SingleExperimentRunner` (line 1164)
- `CheckpointAnalyzer` (line 1223)
- `Application` (line 1275)

**Functions:**
- `main` (line 1334)
- `set_seed` (line 76)
- `create_logger` (line 86)
- `analyze` (line 101)
- `compute` (line 107)
- `__init__` (line 112)
- `_precompute_spectral_operators` (line 116)
- `apply` (line 122)
- `time_evolution` (line 127)
- `__init__` (line 134)
- `__len__` (line 173)
- `__getitem__` (line 176)
- `get_validation_batch` (line 179)
- `__init__` (line 184)
- `forward` (line 195)
- `__init__` (line 228)
- `forward` (line 243)
- `compute_local_complexity` (line 259)
- `compute_superposition` (line 275)
- `compute` (line 303) - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311)
- `compute_discretization_margin_from_state_dict` (line 348) - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738)
- `compute_effective_temperature` (line 747)
- `compute_specific_heat` (line 760)
- `compute` (line 771)
- `compute_weight_diffraction` (line 776)
- `_compute_spectral_entropy` (line 795)
- `__init__` (line 805)
- `should_save_checkpoint` (line 813)
- `save_checkpoint` (line 818)
- `__init__` (line 875)
- `update_metrics` (line 895)
- `__init__` (line 913)
- `should_stop` (line 918)
- `is_crystal_formed` (line 963)
- `__init__` (line 974)
- `train_epoch` (line 997)
- `validate` (line 1027)
- `compute_weight_metrics` (line 1040)
- `execute_training` (line 1056)
- `__init__` (line 1114)
- `mine` (line 1118)
- `__init__` (line 1165)
- `run` (line 1175)
- `__init__` (line 1224)
- `analyze` (line 1230)
- `__init__` (line 1276)
- `_create_argument_parser` (line 1280)
- `run` (line 1294)
- `safe_compute` (line 694)

#### `main.py`
**Path:** `main.py`

**Classs:**
- `Config` (line 44)
- `SeedManager` (line 153)
- `LoggerFactory` (line 165)
- `IAnalysisStrategy` (line 180)
- `IMetricsCalculator` (line 186)
- `HamiltonianOperator` (line 192)
- `SpectralLayer` (line 216)
- `HamiltonianBackbone` (line 254)
- `HamiltonianInferenceEngine` (line 281)
- `SchrodingerPotentialGenerator` (line 339)
- `SchrodingerDataset` (line 389)
- `SchrodingerSpectralNetwork` (line 508)
- `LocalComplexityAnalyzer` (line 542)
- `SuperpositionAnalyzer` (line 559)
- `CrystallographyMetricsCalculator` (line 580)
- `ThermodynamicMetricsCalculator` (line 790)
- `SpectroscopyMetricsCalculator` (line 846)
- `LambdaPressureScheduler` (line 883)
- `AnnealingScheduler` (line 917)
- `TrainingMetricsMonitor` (line 947)
- `CheckpointManager` (line 1050)
- `GlassStateDetector` (line 1102)
- `WeightIntegrityChecker` (line 1158)
- `TrainingEngine` (line 1190)
- `BatchSizeProspector` (line 1335)
- `SeedMiner` (line 1395)
- `FullTrainingOrchestrator` (line 1491)
- `RefinementOrchestrator` (line 1619)
- `ExperimentOrchestrator` (line 1737)

**Functions:**
- `build_argument_parser` (line 1847)
- `main` (line 1914)
- `set_seed` (line 155)
- `create_logger` (line 167)
- `analyze` (line 182)
- `compute` (line 188)
- `__init__` (line 193)
- `_precompute_spectral_operators` (line 197)
- `apply` (line 203)
- `time_evolution` (line 208)
- `__init__` (line 217)
- `forward` (line 228)
- `__init__` (line 255)
- `forward` (line 270)
- `__init__` (line 282)
- `_try_load_backbone` (line 289)
- `apply_hamiltonian` (line 323)
- `time_evolve` (line 329)
- `__init__` (line 340)
- `harmonic_potential` (line 344)
- `double_well_potential` (line 352)
- `coulomb_like_potential` (line 360)
- `periodic_lattice_potential` (line 368)
- `generate_mixed_potential` (line 374)
- `__init__` (line 390)
- `_solve_schrodinger_sample` (line 435)
- `_time_evolve_wavefunction` (line 465)
- `__len__` (line 498)
- `__getitem__` (line 501)
- `get_validation_batch` (line 504)
- `__init__` (line 509)
- `forward` (line 531)
- `compute_local_complexity` (line 544)
- `compute_superposition` (line 561)
- `__init__` (line 581)
- `compute` (line 585)
- `compute_kappa` (line 590)
- `compute_discretization_margin` (line 643)
- `compute_alpha_purity` (line 651)
- `compute_kappa_quantum` (line 657)
- `compute_poynting_vector` (line 678)
- `compute_hbar_effective` (line 734)
- `compute_all_metrics` (line 743)
- `__init__` (line 791)
- `compute` (line 794)
- `compute_effective_temperature` (line 807)
- `compute_specific_heat` (line 831)
- `__init__` (line 847)
- `compute` (line 850)
- `compute_weight_diffraction` (line 854)
- `_compute_spectral_entropy` (line 874)
- `__init__` (line 884)
- `current_lambda` (line 893)
- `step` (line 896)
- `compute_regularization_loss` (line 905)
- `set_lambda` (line 913)
- `__init__` (line 918)
- `temperature` (line 926)
- `step` (line 929)
- `accept_perturbation` (line 935)
- `should_restart` (line 943)
- `__init__` (line 948)
- `update_metrics` (line 966)
- `compute_delta_slope` (line 977)
- `format_progress_bar` (line 990)
- `__init__` (line 1051)
- `should_save_checkpoint` (line 1060)
- `save_checkpoint` (line 1065)
- `__init__` (line 1103)
- `should_stop` (line 1109)
- `is_crystal_formed` (line 1144)
- `check` (line 1160)
- `__init__` (line 1191)
- `compute_weight_metrics` (line 1201)
- `compute_norm_conservation_error` (line 1216)
- `train_single_epoch` (line 1229)
- `validate` (line 1273)
- `collect_all_metrics` (line 1286)
- `__init__` (line 1336)
- `prospect` (line 1341)
- `__init__` (line 1396)
- `mine` (line 1407)
- `__init__` (line 1492)
- `run_phase3_training` (line 1505)
- `__init__` (line 1620)
- `run_phase4_refinement` (line 1639)
- `__init__` (line 1738)
- `run` (line 1742)
- `_save_final_results` (line 1787)
- `safe_compute` (line 756)
- `safe_get` (line 996)

#### `orbital_visualizer2.py`
**Path:** `orbital_visualizer2.py`

**Classs:**
- `Config` (line 44)
- `WavefunctionCalculator` (line 83) - *Calculates hydrogen atom wavefunctions.*
- `HamiltonianNNProcessor` (line 126) - *Uses YOUR TRAINED MODEL for calculations.*
- `MonteCarloSampler` (line 160) - *Monte Carlo sampling for orbital visualization.*
- `OrbitalVisualizer` (line 265) - *HIGH RESOLUTION visualization - NOT 16x16!*

**Functions:**
- `main` (line 433)
- `radial_wavefunction` (line 87)
- `spherical_harmonic_real` (line 97)
- `psi_on_grid` (line 107)
- `__init__` (line 129)
- `is_model_loaded` (line 133)
- `compute_expected_energy` (line 136)
- `__init__` (line 163)
- `find_max_probability` (line 166)
- `sample` (line 195)
- `visualize` (line 268)
- `_plotly` (line 396)

#### `schrodinger_crystal_fixed.py`
**Path:** `schrodinger_crystal_fixed.py`

**Classs:**
- `Config` (line 53)
- `IPhaseDetector` (line 212)
- `IMetricCalculator` (line 218)
- `SeedManager` (line 224)
- `LoggerFactory` (line 236)
- `HamiltonianOperator` (line 251)
- `SpectralLayer` (line 275)
- `HamiltonianBackbone` (line 313)
- `HamiltonianInferenceEngine` (line 340)
- `SchrodingerPotentialGenerator` (line 398)
- `SchrodingerDataset` (line 448)
- `SchrodingerSpectralNetwork` (line 567)
- `FullFourierAnalyzer` (line 601) - *Complete 2D Fourier Transform analysis for resonance detection.
Implements full spectral analysis including power spectrum density,
phase coherence, and harmonic ratio detection for crystalline structure.*
- `FourierMassCenterAnalyzer` (line 793) - *Analyzes center of mass in the 2D torus Fourier space.
Detects spatial alignments indicating liquid -> crystal transition.
Enhanced with full 2D FFT integration.*
- `TopologicalPhaseDetector` (line 873) - *Detects topological phase transition (liquid -> crystal) using
Fourier mass center analysis with hysteresis and full spectral integration.*
- `SpectralFieldExtractor` (line 946)
- `TopologicalCrystallizationLoss` (line 968)
- `CrystallizationPressureApplicator` (line 1006)
- `TopologicalMetricsCalculator` (line 1021)
- `LocalComplexityAnalyzer` (line 1084)
- `SuperpositionAnalyzer` (line 1101)
- `CrystallographyMetricsCalculator` (line 1122)
- `ThermodynamicMetricsCalculator` (line 1332)
- `SpectralGeometryCalculator` (line 1412)
- `RicciCurvatureCalculator` (line 1465)
- `SpectroscopyMetricsCalculator` (line 1509)
- `LambdaPressureScheduler` (line 1546)
- `AdaptiveLambdaScheduler` (line 1580)
- `QuadruplePrecisionLambdaScheduler` (line 1601) - *Lambda scheduler using quadruple precision (float128) for Phase 5.
Provides extreme precision for crystallization pressure.*
- `AnnealingScheduler` (line 1639)
- `TopologicalAnnealingScheduler` (line 1669)
- `TrainingMetricsMonitor` (line 1688)
- `CheckpointManager` (line 1844)
- `Phase5CheckpointManager` (line 1902) - *Specialized checkpoint manager for Phase 5 with quadruple precision.
Only overwrites latest.pth when new checkpoint is better (higher accuracy).*
- `GlassStateDetector` (line 2011)
- `WeightIntegrityChecker` (line 2067)
- `TrainingEngine` (line 2099)
- `BatchSizeProspector` (line 2284)
- `SeedMiner` (line 2355)
- `FullTrainingOrchestrator` (line 2496)
- `RefinementOrchestrator` (line 2628)
- `Phase5Orchestrator` (line 2757) - *Phase 5: Quadruple precision (float128) high-pressure crystallization.
Uses extreme lambda pressure with thermal injection for final crystallization.*
- `ExperimentOrchestrator` (line 2910)

**Functions:**
- `build_argument_parser` (line 3065)
- `main` (line 3160)
- `detect` (line 214)
- `compute` (line 220)
- `set_seed` (line 226)
- `create_logger` (line 238)
- `__init__` (line 252)
- `_precompute_spectral_operators` (line 256)
- `apply` (line 262)
- `time_evolution` (line 267)
- `__init__` (line 276)
- `forward` (line 287)
- `__init__` (line 314)
- `forward` (line 329)
- `__init__` (line 341)
- `_try_load_backbone` (line 348)
- `apply_hamiltonian` (line 382)
- `time_evolve` (line 388)
- `__init__` (line 399)
- `harmonic_potential` (line 403)
- `double_well_potential` (line 411)
- `coulomb_like_potential` (line 419)
- `periodic_lattice_potential` (line 427)
- `generate_mixed_potential` (line 433)
- `__init__` (line 449)
- `_solve_schrodinger_sample` (line 494)
- `_time_evolve_wavefunction` (line 524)
- `__len__` (line 557)
- `__getitem__` (line 560)
- `get_validation_batch` (line 563)
- `__init__` (line 568)
- `forward` (line 590)
- `__init__` (line 607)
- `compute_full_spectrum` (line 615) - *Compute complete 2D Fourier spectrum with phase and magnitude analysis.*
- `detect_bragg_peaks` (line 692) - *Detect Bragg peaks in power spectrum for crystalline structure identification.*
- `compute_resonance_metrics` (line 749) - *Compute resonance metrics for crystallization detection.*
- `__init__` (line 799)
- `compute_mass_center` (line 807) - *Compute center of mass of weight spectrum on the torus.
Integrates with full Fourier analysis for comprehensive detection.*
- `__init__` (line 878)
- `detect` (line 885)
- `extract` (line 948)
- `__init__` (line 969)
- `forward` (line 974)
- `__init__` (line 1007)
- `apply` (line 1011)
- `__init__` (line 1022)
- `compute` (line 1029)
- `apply_crystallization_pressure` (line 1062)
- `_empty_metrics` (line 1068)
- `compute_local_complexity` (line 1086)
- `compute_superposition` (line 1103)
- `__init__` (line 1123)
- `compute` (line 1127)
- `compute_kappa` (line 1132)
- `compute_discretization_margin` (line 1185)
- `compute_alpha_purity` (line 1193)
- `compute_kappa_quantum` (line 1199)
- `compute_poynting_vector` (line 1220)
- `compute_hbar_effective` (line 1276)
- `compute_all_metrics` (line 1285)
- `__init__` (line 1333)
- `compute` (line 1336)
- `compute_effective_temperature` (line 1363)
- `compute_specific_heat` (line 1387)
- `compute_gibbs_free_energy` (line 1401)
- `compute_critical_temperature` (line 1408)
- `__init__` (line 1413)
- `compute` (line 1416)
- `_compute_level_spacing_ratio` (line 1453)
- `__init__` (line 1466)
- `compute` (line 1469)
- `_compute_ricci_scalar` (line 1488)
- `_estimate_sectional_curvatures` (line 1496)
- `__init__` (line 1510)
- `compute` (line 1513)
- `compute_weight_diffraction` (line 1517)
- `_compute_spectral_entropy` (line 1537)
- `__init__` (line 1547)
- `current_lambda` (line 1556)
- `step` (line 1559)
- `compute_regularization_loss` (line 1568)
- `set_lambda` (line 1576)
- `__init__` (line 1581)
- `step_adaptive` (line 1586)
- `__init__` (line 1606)
- `current_lambda` (line 1615)
- `step` (line 1618)
- `compute_regularization_loss` (line 1627)
- `set_lambda` (line 1635)
- `__init__` (line 1640)
- `temperature` (line 1648)
- `step` (line 1651)
- `accept_perturbation` (line 1657)
- `should_restart` (line 1665)
- `__init__` (line 1670)
- `step_adaptive` (line 1674)
- `__init__` (line 1689)
- `update_metrics` (line 1719)
- `compute_delta_slope` (line 1730)
- `format_progress_bar` (line 1743)
- `__init__` (line 1845)
- `should_save_checkpoint` (line 1854)
- `save_checkpoint` (line 1859)
- `load_latest_checkpoint` (line 1895)
- `__init__` (line 1907)
- `_load_best_metrics` (line 1918)
- `should_save` (line 1937)
- `save_checkpoint` (line 1948)
- `load_checkpoint` (line 1992)
- `__init__` (line 2012)
- `should_stop` (line 2018)
- `is_crystal_formed` (line 2053)
- `check` (line 2069)
- `__init__` (line 2100)
- `compute_weight_metrics` (line 2113)
- `compute_norm_conservation_error` (line 2128)
- `train_single_epoch` (line 2141)
- `validate` (line 2185)
- `collect_all_metrics` (line 2198)
- `__init__` (line 2285)
- `prospect` (line 2290)
- `__init__` (line 2356)
- `mine` (line 2367)
- `__init__` (line 2497)
- `run_phase3_training` (line 2510)
- `__init__` (line 2629)
- `run_phase4_refinement` (line 2648)
- `__init__` (line 2762)
- `run_phase5_crystallization` (line 2780)
- `__init__` (line 2911)
- `run` (line 2915)
- `_save_final_results` (line 2990)
- `safe_compute` (line 1298)
- `safe_get` (line 1747)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
