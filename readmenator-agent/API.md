# API

## berry_phase_calculator.py

### visualize_results `def visualize_results(result, output_path)`
- Defined: `berry_phase_calculator.py:388`
- Doc: Create visualization of Berry phase results.

### main `def main()`
- Defined: `berry_phase_calculator.py:514`

### __init__ `def __init__(self, device)`
- Defined: `berry_phase_calculator.py:45`

### load_checkpoints `def load_checkpoints(self, checkpoint_dir)`
- Defined: `berry_phase_calculator.py:48`
- Doc: Load all checkpoints from directory in chronological order.

### _extract_epoch `def _extract_epoch(self, filepath)`
- Defined: `berry_phase_calculator.py:69`
- Doc: Extract epoch number from checkpoint filename.

### extract_spectral_kernels `def extract_spectral_kernels(self, state_dict)`
- Defined: `berry_phase_calculator.py:74`
- Doc: Extract spectral kernels from model state dict.

### flatten_kernel_params `def flatten_kernel_params(self, state_dict)`
- Defined: `berry_phase_calculator.py:107`
- Doc: Flatten all kernel parameters into a single complex vector.

### compute_spectral_density `def compute_spectral_density(self, kernel)`
- Defined: `berry_phase_calculator.py:119`
- Doc: Compute spectral density |W(k)|².

### compute_center_of_mass `def compute_center_of_mass(self, kernel)`
- Defined: `berry_phase_calculator.py:123`
- Doc: Compute center of mass in 2D Fourier space.

### compute_berry_connection_discrete `def compute_berry_connection_discrete(self, theta_prev, theta_curr)`
- Defined: `berry_phase_calculator.py:157`
- Doc: Compute discrete Berry connection between two parameter states.

### compute_eigenvalue_spectrum `def compute_eigenvalue_spectrum(self, kernel)`
- Defined: `berry_phase_calculator.py:188`
- Doc: Compute eigenvalue spectrum of kernel Gram matrix.

### compute_eigenvalue_gap `def compute_eigenvalue_gap(self, eigenvalues)`
- Defined: `berry_phase_calculator.py:209`
- Doc: Compute gap between two largest eigenvalues.

### compute_trajectory_metrics `def compute_trajectory_metrics(self, kernels)`
- Defined: `berry_phase_calculator.py:215`
- Doc: Compute trajectory metrics in parameter space.

### calculate_berry_phase `def calculate_berry_phase(self, checkpoint_dir)`
- Defined: `berry_phase_calculator.py:248`
- Doc: Main method to calculate Berry phase from checkpoint directory.

### calculate_from_final_checkpoint `def calculate_from_final_checkpoint(self, checkpoint_path)`
- Defined: `berry_phase_calculator.py:334`
- Doc: Calculate Berry phase estimates from final checkpoint metrics history.

## crystallographer.py

### build_argument_parser `def build_argument_parser()`
- Defined: `crystallographer.py:2551`
- Doc: Build the command-line argument parser.

### main `def main()`
- Defined: `crystallographer.py:2611`
- Doc: Main entry point for the definitive crystallographer.

### create_logger `def create_logger(name, level)`
- Defined: `crystallographer.py:159`
- Doc: Create and configure a logger with standardized formatting.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:189`
- Doc: Compute metrics for the given model.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:209`
- Doc: Initialize the Hamiltonian operator with precomputed spectral operators.

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `crystallographer.py:220`
- Doc: Precompute the Laplacian spectrum in Fourier space for efficient application.

### apply `def apply(self, field)`
- Defined: `crystallographer.py:229`
- Doc: Apply the Hamiltonian operator to a field using spectral methods.

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `crystallographer.py:243`
- Doc: Perform time evolution of the field under the Hamiltonian.

### __init__ `def __init__(self, channels, grid_size, config)`
- Defined: `crystallographer.py:268`
- Doc: Initialize spectral layer with complex-valued kernels.

### forward `def forward(self, x)`
- Defined: `crystallographer.py:287`
- Doc: Apply spectral convolution in Fourier domain.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:322`
- Doc: Initialize the Hamiltonian backbone network.

### forward `def forward(self, x)`
- Defined: `crystallographer.py:338`
- Doc: Forward pass through the Hamiltonian backbone.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:364`
- Doc: Initialize the Schrodinger spectral network.

### forward `def forward(self, x)`
- Defined: `crystallographer.py:384`
- Doc: Forward pass through the Schrodinger network.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:410`
- Doc: Initialize the Hamiltonian inference engine.

### _try_load_backbone `def _try_load_backbone(self)`
- Defined: `crystallographer.py:423`
- Doc: Attempt to load a pretrained backbone for Hamiltonian inference.

### apply_hamiltonian `def apply_hamiltonian(self, field)`
- Defined: `crystallographer.py:454`
- Doc: Apply the Hamiltonian to a field using backbone or analytical operator.

### time_evolve `def time_evolve(self, field, dt)`
- Defined: `crystallographer.py:469`
- Doc: Perform time evolution using backbone or analytical operator.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:496`
- Doc: Initialize the potential generator.

### harmonic_potential `def harmonic_potential(self)`
- Defined: `crystallographer.py:506`
- Doc: Generate a harmonic oscillator potential.

### double_well_potential `def double_well_potential(self)`
- Defined: `crystallographer.py:520`
- Doc: Generate a double-well potential.

### coulomb_like_potential `def coulomb_like_potential(self)`
- Defined: `crystallographer.py:534`
- Doc: Generate a Coulomb-like central potential.

### periodic_lattice_potential `def periodic_lattice_potential(self)`
- Defined: `crystallographer.py:548`
- Doc: Generate a periodic lattice potential.

### generate_mixed_potential `def generate_mixed_potential(self, seed)`
- Defined: `crystallographer.py:560`
- Doc: Generate a mixed potential combining multiple potential types.

### __init__ `def __init__(self, config, hamiltonian_engine)`
- Defined: `crystallographer.py:590`
- Doc: Initialize the synthetic data generator.

### generate_batch `def generate_batch(self, seed)`
- Defined: `crystallographer.py:603`
- Doc: Generate a batch of validation data.

### _solve_schrodinger_sample `def _solve_schrodinger_sample(self, potential, sample_seed)`
- Defined: `crystallographer.py:633`
- Doc: Solve the Schrodinger equation for a single sample.

### _time_evolve_wavefunction `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential)`
- Defined: `crystallographer.py:673`
- Doc: Time evolve a wavefunction under the given potential.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:714`
- Doc: Initialize the weight integrity calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:723`
- Doc: Compute weight integrity metrics for the model.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:778`
- Doc: Initialize the discretization calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:787`
- Doc: Compute discretization metrics for the model.

### _compute_spectral_entropy `def _compute_spectral_entropy(self, weights)`
- Defined: `crystallographer.py:828`
- Doc: Compute the spectral entropy of the weight distribution.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:857`
- Doc: Initialize the local complexity calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:866`
- Doc: Compute local complexity for the model weights.

### _compute_local_complexity `def _compute_local_complexity(self, weights)`
- Defined: `crystallographer.py:887`
- Doc: Compute local complexity for a weight matrix.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:916`
- Doc: Initialize the superposition calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:925`
- Doc: Compute superposition metrics for the model.

### _compute_superposition `def _compute_superposition(self, weights)`
- Defined: `crystallographer.py:946`
- Doc: Compute superposition metric for a weight matrix.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:979`
- Doc: Initialize the gradient dynamics calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:988`
- Doc: Compute gradient dynamics metrics.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1097`
- Doc: Initialize the spectral geometry calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1106`
- Doc: Compute spectral geometry metrics.

### _compute_level_spacing_ratio `def _compute_level_spacing_ratio(self, spacings)`
- Defined: `crystallographer.py:1161`
- Doc: Compute the level spacing ratio for MBL analysis.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1187`
- Doc: Initialize the Ricci curvature calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1196`
- Doc: Compute Ricci curvature metrics.

### _compute_ricci_scalar `def _compute_ricci_scalar(self, metric)`
- Defined: `crystallographer.py:1225`
- Doc: Compute the Ricci scalar from the metric tensor.

### _estimate_sectional_curvatures `def _estimate_sectional_curvatures(self, metric, samples)`
- Defined: `crystallographer.py:1243`
- Doc: Estimate sectional curvatures by sampling 2D sections.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1271`
- Doc: Initialize the thermodynamic calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1280`
- Doc: Compute thermodynamic metrics.

### _classify_phase `def _classify_phase(self, delta, kappa, temp, alpha)`
- Defined: `crystallographer.py:1320`
- Doc: Classify the thermodynamic phase based on metrics.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1351`
- Doc: Initialize the kappa quantum calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1360`
- Doc: Compute quantum condition number.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1402`
- Doc: Initialize the Poynting vector calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1411`
- Doc: Compute Poynting vector metrics.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1495`
- Doc: Initialize the hbar effective calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1504`
- Doc: Compute effective hbar.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1534`
- Doc: Initialize the weight diffraction calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1543`
- Doc: Compute weight diffraction metrics with Bragg peak detection.

### _compute_spectral_entropy `def _compute_spectral_entropy(self, power_spectrum)`
- Defined: `crystallographer.py:1602`
- Doc: Compute spectral entropy of the power spectrum.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1625`
- Doc: Initialize the phase structure calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1634`
- Doc: Compute phase structure metrics.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1684`
- Doc: Initialize the holomorphy calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1693`
- Doc: Compute holomorphy metrics for complex kernels.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1767`
- Doc: Initialize the norm conservation calculator.

### compute `def compute(self, model)`
- Defined: `crystallographer.py:1776`
- Doc: Compute norm conservation error.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1806`
- Doc: Initialize the crystallographic grader.

### assign_grade `def assign_grade(self, delta, alpha, kappa, num_bragg_peaks)`
- Defined: `crystallographer.py:1815`
- Doc: Assign a crystallographic grade based on multiple metrics.

### classify `def classify(metrics, config)`
- Defined: `crystallographer.py:1879`
- Doc: Classify the thermodynamic phase based on computed metrics.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:1948`
- Doc: Initialize the checkpoint loader.

### load `def load(self, checkpoint_path)`
- Defined: `crystallographer.py:1958`
- Doc: Load a model from checkpoint file.

### extract_metadata `def extract_metadata(self, checkpoint_path)`
- Defined: `crystallographer.py:1995`
- Doc: Extract metadata from checkpoint file.

### categorize_weights `def categorize_weights(self, state_dict)`
- Defined: `crystallographer.py:2028`
- Doc: Categorize weights by layer type for detailed analysis.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:2079`
- Doc: Initialize the crystallography suite with all calculators.

### scan_directory `def scan_directory(self, directory)`
- Defined: `crystallographer.py:2112`
- Doc: Scan directory for checkpoint files.

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path, seed)`
- Defined: `crystallographer.py:2131`
- Doc: Perform comprehensive analysis on a single checkpoint.

### run_full_analysis `def run_full_analysis(self, directory, seed)`
- Defined: `crystallographer.py:2215`
- Doc: Run comprehensive analysis on all checkpoints in a directory.

### generate_report `def generate_report(self, results, output_path)`
- Defined: `crystallographer.py:2236`
- Doc: Generate JSON report from analysis results.

### generate_summary `def generate_summary(self, results)`
- Defined: `crystallographer.py:2249`
- Doc: Generate aggregate summary from analysis results.

### print_summary `def print_summary(self, results)`
- Defined: `crystallographer.py:2322`
- Doc: Print formatted summary table to console.

### __init__ `def __init__(self, config)`
- Defined: `crystallographer.py:2359`
- Doc: Initialize the batch analyzer.

### analyze_directory `def analyze_directory(self, directory, seed)`
- Defined: `crystallographer.py:2372`
- Doc: Analyze all checkpoints in a directory with visualization.

### _save_summary `def _save_summary(self, summary, results)`
- Defined: `crystallographer.py:2404`
- Doc: Save analysis summary and individual reports.

### _generate_visualization `def _generate_visualization(self, results)`
- Defined: `crystallographer.py:2426`
- Doc: Generate comprehensive visualization of analysis results.

## experiment2.py

### main `def main()`
- Defined: `experiment2.py:1334`

### set_seed `def set_seed(seed)`
- Defined: `experiment2.py:76`

### create_logger `def create_logger(name, level)`
- Defined: `experiment2.py:86`

### analyze `def analyze(self, model)`
- Defined: `experiment2.py:101`

### compute `def compute(self, model)`
- Defined: `experiment2.py:107`

### __init__ `def __init__(self, grid_size)`
- Defined: `experiment2.py:112`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `experiment2.py:116`

### apply `def apply(self, field)`
- Defined: `experiment2.py:122`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `experiment2.py:127`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- Defined: `experiment2.py:134`

### __len__ `def __len__(self)`
- Defined: `experiment2.py:173`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `experiment2.py:176`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `experiment2.py:179`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `experiment2.py:184`

### forward `def forward(self, x)`
- Defined: `experiment2.py:195`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `experiment2.py:228`

### forward `def forward(self, x)`
- Defined: `experiment2.py:243`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `experiment2.py:259`

### compute_superposition `def compute_superposition(weights)`
- Defined: `experiment2.py:275`

### compute `def compute(self, model, val_x, val_y)`
- Defined: `experiment2.py:303`
- Doc: Implementación de interfaz IMetricsCalculator.

### compute_gradient_covariance_kappa `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- Defined: `experiment2.py:311`

### compute_discretization_margin_from_state_dict `def compute_discretization_margin_from_state_dict(model)`
- Defined: `experiment2.py:348`
- Doc: Calcula el margen de discretización desde los parámetros del modelo.

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `experiment2.py:361`
- Doc: Calcula el margen de discretización desde un diccionario de coeficientes.

### compute_alpha_purity_from_model `def compute_alpha_purity_from_model(model)`
- Defined: `experiment2.py:373`
- Doc: Calcula el índice de pureza alpha directamente desde el modelo.

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `experiment2.py:383`
- Doc: Calcula el índice de pureza alpha desde un diccionario de coeficientes.

### compute_kappa `def compute_kappa(model, val_x, val_y, num_batches)`
- Defined: `experiment2.py:393`
- Doc: Número de condición de la matriz de covarianza de gradientes.

### compute_kappa_quantum `def compute_kappa_quantum(model, hbar)`
- Defined: `experiment2.py:464`
- Doc: Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.

### compute_kappa_quantum_from_coeffs `def compute_kappa_quantum_from_coeffs(coeffs, hbar)`
- Defined: `experiment2.py:492`
- Doc: Versión del cálculo cuántico de kappa desde diccionario de coeficientes.

### _compute_crystallography_metrics `def _compute_crystallography_metrics(self, model, val_x, val_y)`
- Defined: `experiment2.py:511`
- Doc: Métricas cristalográficas con aislamiento completo de errores.

### _check_weight_integrity `def _check_weight_integrity(self, model)`
- Defined: `experiment2.py:539`
- Doc: Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.

### compute_poynting_vector `def compute_poynting_vector(model)`
- Defined: `experiment2.py:603`
- Doc: Vector de Poynting: flujo de energía en el espacio de parámetros.

### compute_all_metrics `def compute_all_metrics(model, val_x, val_y)`
- Defined: `experiment2.py:679`
- Doc: Calcula todas las métricas cristalográficas con manejo de errores.

### compute `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- Defined: `experiment2.py:738`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `experiment2.py:747`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `experiment2.py:760`

### compute `def compute(self, model)`
- Defined: `experiment2.py:771`

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `experiment2.py:776`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `experiment2.py:795`

### __init__ `def __init__(self, interval_minutes, max_checkpoints)`
- Defined: `experiment2.py:805`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `experiment2.py:813`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `experiment2.py:818`

### __init__ `def __init__(self)`
- Defined: `experiment2.py:875`

### update_metrics `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- Defined: `experiment2.py:895`

### __init__ `def __init__(self, patience_epochs)`
- Defined: `experiment2.py:913`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `experiment2.py:918`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `experiment2.py:963`

### __init__ `def __init__(self, model, optimizer, device, logger)`
- Defined: `experiment2.py:974`

### train_epoch `def train_epoch(self, dataloader, epoch)`
- Defined: `experiment2.py:997`

### validate `def validate(self, val_x, val_y)`
- Defined: `experiment2.py:1027`

### compute_weight_metrics `def compute_weight_metrics(self)`
- Defined: `experiment2.py:1040`

### execute_training `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- Defined: `experiment2.py:1056`

### __init__ `def __init__(self, max_attempts)`
- Defined: `experiment2.py:1114`

### mine `def mine(self)`
- Defined: `experiment2.py:1118`

### __init__ `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- Defined: `experiment2.py:1165`

### run `def run(self)`
- Defined: `experiment2.py:1175`

### __init__ `def __init__(self, checkpoint_path, results_dir)`
- Defined: `experiment2.py:1224`

### analyze `def analyze(self)`
- Defined: `experiment2.py:1230`

### __init__ `def __init__(self)`
- Defined: `experiment2.py:1276`

### _create_argument_parser `def _create_argument_parser(self)`
- Defined: `experiment2.py:1280`

### run `def run(self)`
- Defined: `experiment2.py:1294`

### safe_compute `def safe_compute(func)`
- Defined: `experiment2.py:694`

## main.py

### build_argument_parser `def build_argument_parser()`
- Defined: `main.py:1847`

### main `def main()`
- Defined: `main.py:1914`

### set_seed `def set_seed(seed, device)`
- Defined: `main.py:155`

### create_logger `def create_logger(name, level)`
- Defined: `main.py:167`

### analyze `def analyze(self, model)`
- Defined: `main.py:182`

### compute `def compute(self, model)`
- Defined: `main.py:188`

### __init__ `def __init__(self, grid_size)`
- Defined: `main.py:193`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `main.py:197`

### apply `def apply(self, field)`
- Defined: `main.py:203`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `main.py:208`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `main.py:217`

### forward `def forward(self, x)`
- Defined: `main.py:228`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `main.py:255`

### forward `def forward(self, x)`
- Defined: `main.py:270`

### __init__ `def __init__(self, config)`
- Defined: `main.py:282`

### _try_load_backbone `def _try_load_backbone(self)`
- Defined: `main.py:289`

### apply_hamiltonian `def apply_hamiltonian(self, field)`
- Defined: `main.py:323`

### time_evolve `def time_evolve(self, field, dt)`
- Defined: `main.py:329`

### __init__ `def __init__(self, config)`
- Defined: `main.py:340`

### harmonic_potential `def harmonic_potential(self)`
- Defined: `main.py:344`

### double_well_potential `def double_well_potential(self)`
- Defined: `main.py:352`

### coulomb_like_potential `def coulomb_like_potential(self)`
- Defined: `main.py:360`

### periodic_lattice_potential `def periodic_lattice_potential(self)`
- Defined: `main.py:368`

### generate_mixed_potential `def generate_mixed_potential(self, seed)`
- Defined: `main.py:374`

### __init__ `def __init__(self, config, hamiltonian_engine, seed)`
- Defined: `main.py:390`

### _solve_schrodinger_sample `def _solve_schrodinger_sample(self, potential, sample_seed)`
- Defined: `main.py:435`

### _time_evolve_wavefunction `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential, energy)`
- Defined: `main.py:465`

### __len__ `def __len__(self)`
- Defined: `main.py:498`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `main.py:501`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `main.py:504`

### __init__ `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, input_channels, output_channels)`
- Defined: `main.py:509`

### forward `def forward(self, x)`
- Defined: `main.py:531`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `main.py:544`

### compute_superposition `def compute_superposition(weights)`
- Defined: `main.py:561`

### __init__ `def __init__(self, config)`
- Defined: `main.py:581`

### compute `def compute(self, model)`
- Defined: `main.py:585`

### compute_kappa `def compute_kappa(self, model, val_x, val_y, num_batches)`
- Defined: `main.py:590`

### compute_discretization_margin `def compute_discretization_margin(self, model)`
- Defined: `main.py:643`

### compute_alpha_purity `def compute_alpha_purity(self, model)`
- Defined: `main.py:651`

### compute_kappa_quantum `def compute_kappa_quantum(self, model)`
- Defined: `main.py:657`

### compute_poynting_vector `def compute_poynting_vector(self, model)`
- Defined: `main.py:678`

### compute_hbar_effective `def compute_hbar_effective(self, model, lambda_pressure)`
- Defined: `main.py:734`

### compute_all_metrics `def compute_all_metrics(self, model, val_x, val_y)`
- Defined: `main.py:743`

### __init__ `def __init__(self, config)`
- Defined: `main.py:791`

### compute `def compute(self, model)`
- Defined: `main.py:794`

### compute_effective_temperature `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- Defined: `main.py:807`

### compute_specific_heat `def compute_specific_heat(self, loss_history, temp_history)`
- Defined: `main.py:831`

### __init__ `def __init__(self, config)`
- Defined: `main.py:847`

### compute `def compute(self, model)`
- Defined: `main.py:850`

### compute_weight_diffraction `def compute_weight_diffraction(self, coeffs)`
- Defined: `main.py:854`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `main.py:874`

### __init__ `def __init__(self, config)`
- Defined: `main.py:884`

### current_lambda `def current_lambda(self)`
- Defined: `main.py:893`

### step `def step(self, epoch)`
- Defined: `main.py:896`

### compute_regularization_loss `def compute_regularization_loss(self, model)`
- Defined: `main.py:905`

### set_lambda `def set_lambda(self, value)`
- Defined: `main.py:913`

### __init__ `def __init__(self, config)`
- Defined: `main.py:918`

### temperature `def temperature(self)`
- Defined: `main.py:926`

### step `def step(self)`
- Defined: `main.py:929`

### accept_perturbation `def accept_perturbation(self, delta_loss)`
- Defined: `main.py:935`

### should_restart `def should_restart(self, current_delta, best_delta)`
- Defined: `main.py:943`

### __init__ `def __init__(self, config)`
- Defined: `main.py:948`

### update_metrics `def update_metrics(self)`
- Defined: `main.py:966`

### compute_delta_slope `def compute_delta_slope(self)`
- Defined: `main.py:977`

### format_progress_bar `def format_progress_bar(self, epoch, total_epochs, phase)`
- Defined: `main.py:990`

### __init__ `def __init__(self, config, checkpoint_dir)`
- Defined: `main.py:1051`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `main.py:1060`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- Defined: `main.py:1065`

### __init__ `def __init__(self, config)`
- Defined: `main.py:1103`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `main.py:1109`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `main.py:1144`

### check `def check(model)`
- Defined: `main.py:1160`

### __init__ `def __init__(self, config)`
- Defined: `main.py:1191`

### compute_weight_metrics `def compute_weight_metrics(self, model)`
- Defined: `main.py:1201`

### compute_norm_conservation_error `def compute_norm_conservation_error(self, model, val_x)`
- Defined: `main.py:1216`

### train_single_epoch `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler)`
- Defined: `main.py:1229`

### validate `def validate(self, model, val_x, val_y)`
- Defined: `main.py:1273`

### collect_all_metrics `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr)`
- Defined: `main.py:1286`

### __init__ `def __init__(self, config, hamiltonian_engine)`
- Defined: `main.py:1336`

### prospect `def prospect(self)`
- Defined: `main.py:1341`

### __init__ `def __init__(self, config, hamiltonian_engine, batch_size)`
- Defined: `main.py:1396`

### mine `def mine(self)`
- Defined: `main.py:1407`

### __init__ `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- Defined: `main.py:1492`

### run_phase3_training `def run_phase3_training(self)`
- Defined: `main.py:1505`

### __init__ `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- Defined: `main.py:1620`

### run_phase4_refinement `def run_phase4_refinement(self)`
- Defined: `main.py:1639`

### __init__ `def __init__(self, config)`
- Defined: `main.py:1738`

### run `def run(self)`
- Defined: `main.py:1742`

### _save_final_results `def _save_final_results(self, model, monitor, seed, batch_size)`
- Defined: `main.py:1787`

### safe_compute `def safe_compute(func)`
- Defined: `main.py:756`

### safe_get `def safe_get(key)`
- Defined: `main.py:996`

## orbital_visualizer2.py

### main `def main()`
- Defined: `orbital_visualizer2.py:433`

### radial_wavefunction `def radial_wavefunction(n, l, r)`
- Defined: `orbital_visualizer2.py:87`

### spherical_harmonic_real `def spherical_harmonic_real(l, m, theta, phi)`
- Defined: `orbital_visualizer2.py:97`

### psi_on_grid `def psi_on_grid(n, l, m, grid_size)`
- Defined: `orbital_visualizer2.py:107`

### __init__ `def __init__(self, engine)`
- Defined: `orbital_visualizer2.py:129`

### is_model_loaded `def is_model_loaded(self)`
- Defined: `orbital_visualizer2.py:133`

### compute_expected_energy `def compute_expected_energy(self, n, l, m)`
- Defined: `orbital_visualizer2.py:136`

### __init__ `def __init__(self, hamiltonian_processor)`
- Defined: `orbital_visualizer2.py:163`

### find_max_probability `def find_max_probability(self, n, l, m)`
- Defined: `orbital_visualizer2.py:166`

### sample `def sample(self, n, l, m, num_samples)`
- Defined: `orbital_visualizer2.py:195`

### visualize `def visualize(self, data, save_path, hamiltonian_processor)`
- Defined: `orbital_visualizer2.py:268`

### _plotly `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`
- Defined: `orbital_visualizer2.py:396`

## schrodinger_crystal_fixed.py

### build_argument_parser `def build_argument_parser()`
- Defined: `schrodinger_crystal_fixed.py:3065`

### main `def main()`
- Defined: `schrodinger_crystal_fixed.py:3160`

### detect `def detect(self, spectral_field)`
- Defined: `schrodinger_crystal_fixed.py:214`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:220`

### set_seed `def set_seed(seed, device)`
- Defined: `schrodinger_crystal_fixed.py:226`

### create_logger `def create_logger(name, level)`
- Defined: `schrodinger_crystal_fixed.py:238`

### __init__ `def __init__(self, grid_size)`
- Defined: `schrodinger_crystal_fixed.py:252`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `schrodinger_crystal_fixed.py:256`

### apply `def apply(self, field)`
- Defined: `schrodinger_crystal_fixed.py:262`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `schrodinger_crystal_fixed.py:267`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `schrodinger_crystal_fixed.py:276`

### forward `def forward(self, x)`
- Defined: `schrodinger_crystal_fixed.py:287`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `schrodinger_crystal_fixed.py:314`

### forward `def forward(self, x)`
- Defined: `schrodinger_crystal_fixed.py:329`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:341`

### _try_load_backbone `def _try_load_backbone(self)`
- Defined: `schrodinger_crystal_fixed.py:348`

### apply_hamiltonian `def apply_hamiltonian(self, field)`
- Defined: `schrodinger_crystal_fixed.py:382`

### time_evolve `def time_evolve(self, field, dt)`
- Defined: `schrodinger_crystal_fixed.py:388`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:399`

### harmonic_potential `def harmonic_potential(self)`
- Defined: `schrodinger_crystal_fixed.py:403`

### double_well_potential `def double_well_potential(self)`
- Defined: `schrodinger_crystal_fixed.py:411`

### coulomb_like_potential `def coulomb_like_potential(self)`
- Defined: `schrodinger_crystal_fixed.py:419`

### periodic_lattice_potential `def periodic_lattice_potential(self)`
- Defined: `schrodinger_crystal_fixed.py:427`

### generate_mixed_potential `def generate_mixed_potential(self, seed)`
- Defined: `schrodinger_crystal_fixed.py:433`

### __init__ `def __init__(self, config, hamiltonian_engine, seed)`
- Defined: `schrodinger_crystal_fixed.py:449`

### _solve_schrodinger_sample `def _solve_schrodinger_sample(self, potential, sample_seed)`
- Defined: `schrodinger_crystal_fixed.py:494`

### _time_evolve_wavefunction `def _time_evolve_wavefunction(self, psi_real, psi_imag, potential, energy)`
- Defined: `schrodinger_crystal_fixed.py:524`

### __len__ `def __len__(self)`
- Defined: `schrodinger_crystal_fixed.py:557`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `schrodinger_crystal_fixed.py:560`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `schrodinger_crystal_fixed.py:563`

### __init__ `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, input_channels, output_channels)`
- Defined: `schrodinger_crystal_fixed.py:568`

### forward `def forward(self, x)`
- Defined: `schrodinger_crystal_fixed.py:590`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:607`

### compute_full_spectrum `def compute_full_spectrum(self, spectral_field)`
- Defined: `schrodinger_crystal_fixed.py:615`
- Doc: Compute complete 2D Fourier spectrum with phase and magnitude analysis.

### detect_bragg_peaks `def detect_bragg_peaks(self, power_spectrum, threshold_sigma)`
- Defined: `schrodinger_crystal_fixed.py:692`
- Doc: Detect Bragg peaks in power spectrum for crystalline structure identification.

### compute_resonance_metrics `def compute_resonance_metrics(self, spectral_field)`
- Defined: `schrodinger_crystal_fixed.py:749`
- Doc: Compute resonance metrics for crystallization detection.

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:799`

### compute_mass_center `def compute_mass_center(self, spectral_field)`
- Defined: `schrodinger_crystal_fixed.py:807`
- Doc: Compute center of mass of weight spectrum on the torus.

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:878`

### detect `def detect(self, spectral_field)`
- Defined: `schrodinger_crystal_fixed.py:885`

### extract `def extract(model, grid_size)`
- Defined: `schrodinger_crystal_fixed.py:948`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:969`

### forward `def forward(self, phase_info, epoch)`
- Defined: `schrodinger_crystal_fixed.py:974`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1007`

### apply `def apply(self, model, phase_info)`
- Defined: `schrodinger_crystal_fixed.py:1011`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1022`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1029`

### apply_crystallization_pressure `def apply_crystallization_pressure(self, model, topo_metrics)`
- Defined: `schrodinger_crystal_fixed.py:1062`

### _empty_metrics `def _empty_metrics()`
- Defined: `schrodinger_crystal_fixed.py:1068`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `schrodinger_crystal_fixed.py:1086`

### compute_superposition `def compute_superposition(weights)`
- Defined: `schrodinger_crystal_fixed.py:1103`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1123`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1127`

### compute_kappa `def compute_kappa(self, model, val_x, val_y, num_batches)`
- Defined: `schrodinger_crystal_fixed.py:1132`

### compute_discretization_margin `def compute_discretization_margin(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1185`

### compute_alpha_purity `def compute_alpha_purity(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1193`

### compute_kappa_quantum `def compute_kappa_quantum(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1199`

### compute_poynting_vector `def compute_poynting_vector(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1220`

### compute_hbar_effective `def compute_hbar_effective(self, model, lambda_pressure)`
- Defined: `schrodinger_crystal_fixed.py:1276`

### compute_all_metrics `def compute_all_metrics(self, model, val_x, val_y)`
- Defined: `schrodinger_crystal_fixed.py:1285`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1333`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1336`

### compute_effective_temperature `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- Defined: `schrodinger_crystal_fixed.py:1363`

### compute_specific_heat `def compute_specific_heat(self, loss_history, temp_history)`
- Defined: `schrodinger_crystal_fixed.py:1387`

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self, delta, alpha, temperature)`
- Defined: `schrodinger_crystal_fixed.py:1401`

### compute_critical_temperature `def compute_critical_temperature(self, alpha)`
- Defined: `schrodinger_crystal_fixed.py:1408`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1413`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1416`

### _compute_level_spacing_ratio `def _compute_level_spacing_ratio(self, spacings)`
- Defined: `schrodinger_crystal_fixed.py:1453`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1466`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1469`

### _compute_ricci_scalar `def _compute_ricci_scalar(self, metric)`
- Defined: `schrodinger_crystal_fixed.py:1488`

### _estimate_sectional_curvatures `def _estimate_sectional_curvatures(self, metric)`
- Defined: `schrodinger_crystal_fixed.py:1496`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1510`

### compute `def compute(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1513`

### compute_weight_diffraction `def compute_weight_diffraction(self, coeffs)`
- Defined: `schrodinger_crystal_fixed.py:1517`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `schrodinger_crystal_fixed.py:1537`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1547`

### current_lambda `def current_lambda(self)`
- Defined: `schrodinger_crystal_fixed.py:1556`

### step `def step(self, epoch)`
- Defined: `schrodinger_crystal_fixed.py:1559`

### compute_regularization_loss `def compute_regularization_loss(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1568`

### set_lambda `def set_lambda(self, value)`
- Defined: `schrodinger_crystal_fixed.py:1576`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1581`

### step_adaptive `def step_adaptive(self, epoch, topo_phase_state)`
- Defined: `schrodinger_crystal_fixed.py:1586`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1606`

### current_lambda `def current_lambda(self)`
- Defined: `schrodinger_crystal_fixed.py:1615`

### step `def step(self, epoch, improvement)`
- Defined: `schrodinger_crystal_fixed.py:1618`

### compute_regularization_loss `def compute_regularization_loss(self, model)`
- Defined: `schrodinger_crystal_fixed.py:1627`

### set_lambda `def set_lambda(self, value)`
- Defined: `schrodinger_crystal_fixed.py:1635`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1640`

### temperature `def temperature(self)`
- Defined: `schrodinger_crystal_fixed.py:1648`

### step `def step(self)`
- Defined: `schrodinger_crystal_fixed.py:1651`

### accept_perturbation `def accept_perturbation(self, delta_loss)`
- Defined: `schrodinger_crystal_fixed.py:1657`

### should_restart `def should_restart(self, current_delta, best_delta)`
- Defined: `schrodinger_crystal_fixed.py:1665`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1670`

### step_adaptive `def step_adaptive(self, alignment_trend, resonance_score)`
- Defined: `schrodinger_crystal_fixed.py:1674`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1689`

### update_metrics `def update_metrics(self)`
- Defined: `schrodinger_crystal_fixed.py:1719`

### compute_delta_slope `def compute_delta_slope(self)`
- Defined: `schrodinger_crystal_fixed.py:1730`

### format_progress_bar `def format_progress_bar(self, epoch, total_epochs, phase)`
- Defined: `schrodinger_crystal_fixed.py:1743`

### __init__ `def __init__(self, config, checkpoint_dir)`
- Defined: `schrodinger_crystal_fixed.py:1845`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `schrodinger_crystal_fixed.py:1854`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- Defined: `schrodinger_crystal_fixed.py:1859`

### load_latest_checkpoint `def load_latest_checkpoint(self)`
- Defined: `schrodinger_crystal_fixed.py:1895`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:1907`

### _load_best_metrics `def _load_best_metrics(self)`
- Defined: `schrodinger_crystal_fixed.py:1918`

### should_save `def should_save(self, current_delta, current_alpha, current_acc)`
- Defined: `schrodinger_crystal_fixed.py:1937`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics, lambda_value)`
- Defined: `schrodinger_crystal_fixed.py:1948`

### load_checkpoint `def load_checkpoint(self, model, optimizer)`
- Defined: `schrodinger_crystal_fixed.py:1992`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:2012`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `schrodinger_crystal_fixed.py:2018`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `schrodinger_crystal_fixed.py:2053`

### check `def check(model)`
- Defined: `schrodinger_crystal_fixed.py:2069`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:2100`

### compute_weight_metrics `def compute_weight_metrics(self, model)`
- Defined: `schrodinger_crystal_fixed.py:2113`

### compute_norm_conservation_error `def compute_norm_conservation_error(self, model, val_x)`
- Defined: `schrodinger_crystal_fixed.py:2128`

### train_single_epoch `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler)`
- Defined: `schrodinger_crystal_fixed.py:2141`

### validate `def validate(self, model, val_x, val_y)`
- Defined: `schrodinger_crystal_fixed.py:2185`

### collect_all_metrics `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr, epoch)`
- Defined: `schrodinger_crystal_fixed.py:2198`

### __init__ `def __init__(self, config, hamiltonian_engine)`
- Defined: `schrodinger_crystal_fixed.py:2285`

### prospect `def prospect(self)`
- Defined: `schrodinger_crystal_fixed.py:2290`

### __init__ `def __init__(self, config, hamiltonian_engine, batch_size)`
- Defined: `schrodinger_crystal_fixed.py:2356`

### mine `def mine(self)`
- Defined: `schrodinger_crystal_fixed.py:2367`

### __init__ `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- Defined: `schrodinger_crystal_fixed.py:2497`

### run_phase3_training `def run_phase3_training(self)`
- Defined: `schrodinger_crystal_fixed.py:2510`

### __init__ `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- Defined: `schrodinger_crystal_fixed.py:2629`

### run_phase4_refinement `def run_phase4_refinement(self)`
- Defined: `schrodinger_crystal_fixed.py:2648`

### __init__ `def __init__(self, config, hamiltonian_engine, model, monitor, seed, batch_size)`
- Defined: `schrodinger_crystal_fixed.py:2762`

### run_phase5_crystallization `def run_phase5_crystallization(self)`
- Defined: `schrodinger_crystal_fixed.py:2780`

### __init__ `def __init__(self, config)`
- Defined: `schrodinger_crystal_fixed.py:2911`

### run `def run(self)`
- Defined: `schrodinger_crystal_fixed.py:2915`

### _save_final_results `def _save_final_results(self, model, monitor, seed, batch_size)`
- Defined: `schrodinger_crystal_fixed.py:2990`

### safe_compute `def safe_compute(func)`
- Defined: `schrodinger_crystal_fixed.py:1298`

### safe_get `def safe_get(key)`
- Defined: `schrodinger_crystal_fixed.py:1747`
