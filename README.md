# Schrödinger Topological Crystallization: Phase Space Discovery in Hamiltonian Neural Networks

**grisun0**  
*February 16, 2026*

---

## Abstract

I applied the Hamiltonian Processing Unit (HPU-Core) protocol to the Schrödinger equation to test whether topological crystallization extends from classical Hamiltonian dynamics to quantum wavefunction evolution. A spectral neural network trained on synthetic wavefunction propagation achieved perfect validation accuracy. The discretization margin δ decreased from 0.49 to 0.24 over 1200 epochs, showing the expected downward trajectory but not reaching the target threshold of δ < 0.1. The system displayed partial crystallization signatures: stable spectral entropy, marginal gradient covariance stability, and persistent purity growth. Full crystallization was not achieved within the computational budget. This report documents what worked, what did not, and what remains unclear.


Crystallization ($κ=1, δ \to 0$) cannot be achieved through the brute force of training alone. It only occurs when there is an "Architectural Resonance"—meaning the network topology must be homeomorphic, or at least compatible in tensor rank, with the underlying algorithmic solution.

The Transformer failed the Laderman challenge not because of insufficient training or batch size, but because its softmax attention mechanism and dense layers cannot collapse cleanly into the specific rank-23 tensor decomposition that Laderman requires. An algorithm is not simply "learned"; it is "revealed" only if the architecture already contains it in potential. Geometrically, a Transformer is incapable of becoming a Laderman crystal; it can only approximate it as a glass.

To induce an algorithmic phase transition, the system must cross a critical energy threshold. There are two interchangeable ways to achieve this, much like an equation of state such as $PV=nRT$:

1. Cooling (via Batch Size): Reducing $T_{eff}$ until the system falls into the potential well (as seen in the Strassen case).
2. Compression (via $\lambda$): Increasing "osmotic" pressure ($\lambda$) to crush the search space against the topological solution (as seen in the Hamilton case).

Seed mining should not rely solely on the absolute value of $\delta$ at epoch $t$, but on its momentum or velocity: $v_\delta = \frac{d\delta}{dt}$. A seed with low $\delta$ but $v_\delta > 0$ is a false positive—it is warming up. Conversely, a seed with mid-range $\delta$ but $v_\delta < 0$ (rapid cooling) is a much stronger candidate.

The data suggests that the network’s fate—Crystal vs. Glass—is determined in the earliest epochs. This depends not just on its starting point in the energy landscape, but on the direction and velocity of its initial trajectory. This implies that the weight space possesses pre-existing "currents" dictated by random initialization.

In summary, architecture dictates possibility (Resonance), while pressure ($\lambda$) and temperature (Batch Size) dictate the state. Glass is the natural state of robustness, whereas the crystal is a beautiful but fragile mathematical singularity, accessible only through highly precise ballistic trajectories ($v_\delta < 0$) within resonant architectures.


---

## 1. Introduction

My previous work showed that neural networks trained on classical Hamiltonian dynamics can undergo a phase transition from stochastic optimization to a topologically protected state—a "crystal" characterized by discrete invariants like Berry phases and winding numbers. That work required extreme regularization pressure (λ ≈ 10³⁴) and thousands of epochs.

The natural next question: does this phenomenon survive in quantum mechanics? The Schrödinger equation differs fundamentally from classical Hamiltonian flow. It operates on complex-valued wavefunctions, requires phase coherence, and conserves probability globally rather than locally. These properties make quantum dynamics a harder test case.

This report describes what happened when I applied the same four-phase protocol to the Schrödinger equation. Spoiler: I did not achieve full crystallization. The following sections explain the methodology, present the results honestly, and discuss why the experiment fell short.

---

## 2. Methodology

### 2.1 The Core Metric: Discretization Margin (δ)

Before describing the protocol, I need to explain the key metric: the discretization margin δ. This measures how far the network's learned dynamics deviate from the ideal continuous-time evolution. When δ = 0, the network has perfectly captured the underlying physics. When δ = 0.5, the system is in a disordered state—essentially random. The goal of the protocol is to drive δ below 0.1, which corresponds to the "crystallized" regime where topological invariants become stable.

Why does this matter? In the Hamiltonian experiments, δ < 0.1 correlated with the emergence of protected topological structures. If the same threshold applies to quantum systems, we would have evidence that topological crystallization is a general phenomenon, not a classical peculiarity.

### 2.2 The Four-Phase Protocol

The protocol has four stages, each with a specific purpose:

**Phase 1: Batch Size Prospecting**  
Goal: Find the batch size that minimizes initial δ.  
I tested batch sizes 8, 16, 32, and 64 for 30 epochs each. The rationale is that batch size affects gradient noise, which influences whether the network can escape local minima and find structured solutions. Smaller batches introduce more noise but may help the system explore the loss landscape more thoroughly.

**Phase 2: Seed Mining**  
Goal: Identify random seeds with favorable δ trajectories.  
Neural network training is stochastic. Different random seeds can lead to radically different outcomes, even with identical hyperparameters. Rather than committing to a single long training run with an arbitrary seed, I screened 200 seeds with short training (40 epochs) to find those showing decreasing δ. This "mining" approach increases the probability that the full training run will succeed.

**Phase 3: Full Training**  
Goal: Push the selected seed toward crystallization through extended training with increasing regularization.  
The selected seed undergoes thousands of epochs with gradually increasing λ (regularization pressure). The idea is that extreme sparsity forces the network to find compact representations, which in the Hamiltonian case led to topological structure.

**Phase 4: Refinement**  
Goal: Polish the crystal structure.  
Once crystallization is detected, simulated annealing refines the topological invariants. This phase was not reached in the current experiment.

### 2.3 Architecture

The network uses spectral layers that operate in Fourier space. This choice is not arbitrary: the Schrödinger equation is naturally expressed in momentum space, where the kinetic energy operator becomes diagonal. The architecture:

- Input: 2 channels (real and imaginary parts of ψ)
- Hidden dimensions: 32 → 64
- Spectral layers: 2 layers with complex kernels
- Output: 2 channels (real and imaginary parts)

A pretrained backbone provides the Laplacian operator for kinetic energy computation. The total parameter count is approximately 10,000.

### 2.4 Data

I generated 200 synthetic wavefunction samples:

1. Random potentials (harmonic, double-well, Coulomb-like, and periodic lattice components)
2. Eigenstate initialization via Hamiltonian diagonalization
3. Time evolution using split-step methods
4. 70/30 train/validation split

The small dataset size is intentional. The protocol relies on "grokking"—the phenomenon where networks first memorize training data, then abruptly generalize. Small datasets make this transition more pronounced and easier to study.

---

## 3. Results

### 3.1 Phase 1: Batch Size Prospecting

I ran each batch size for 30 epochs. Here is what happened:

| Batch Size | Initial δ | Final δ | Validation Accuracy |
|------------|-----------|---------|---------------------|
| 8 | 0.494 | 0.494 | 1.000 |
| 16 | 0.495 | 0.495 | 1.000 |
| 32 | 0.495 | 0.495 | 1.000 |
| 64 | 0.495 | 0.495 | 1.000 |

All configurations achieved perfect validation accuracy immediately. The δ values barely moved from the initial ~0.5 level. Batch size 8 showed marginally lower δ (0.494 vs 0.495), so I selected it for subsequent phases. This decision was somewhat arbitrary—the differences were tiny.

What does this mean? The network memorized the small training set instantly. The real question was whether extended training would drive δ down and induce generalization beyond memorization. The prospecting phase was inconclusive on this point.

### 3.2 Phase 2: Seed Mining

I trained 200 random seeds for 40 epochs each with batch size 8. The goal was to find seeds where δ decreased over the short run.

Results:
- 111 seeds (55.5%) showed decreasing δ
- 89 seeds (44.5%) showed flat or increasing δ

The best performer was seed 103, with δ decreasing from 0.491 to 0.489. This is a tiny improvement—only 0.002 over 40 epochs. In the Hamiltonian experiments, the best seed achieved δ ≈ 0.46, significantly below the 0.50 baseline. Here, no seed broke below 0.48.

**Statistical summary:**
- Mean δ change: -0.001
- Standard deviation: 0.005
- Best seed: 103 (final δ = 0.489)

The narrow distribution suggests that quantum wavefunction learning starts from a more constrained region of parameter space. Or perhaps 40 epochs was insufficient for the network to find a good trajectory. I cannot distinguish these explanations from the current data.

### 3.3 Phase 3: Full Training

I trained seed 103 with batch size 8 for 1200 epochs before the run was interrupted. Key observations:

**Discretization Margin (δ):**
- Epoch 10: 0.402
- Epoch 50: 0.359 (grokking detected here)
- Epoch 500: 0.275
- Epoch 1200: 0.242

δ decreased monotonically but did not reach the target of 0.1. The rate of decrease slowed over time, suggesting diminishing returns.

What happened at epoch 50? The "grokking detected" flag fired when δ dropped below 0.4, indicating that the network had transitioned from pure memorization to some form of generalization. Validation accuracy had been perfect from the start (memorization), but the grokking point marks when the internal representation started to reflect the underlying physics rather than just storing training examples.

**Regularization Pressure (λ):**
- Epoch 0: 1.0
- Epoch 500: 10.0
- Epoch 1000: 100.0
- Epoch 1200: 100.0

λ increased by two orders of magnitude during training. In the Hamiltonian experiments, crystallization required λ ≈ 10³⁴. I was 32 orders of magnitude short.

**Purity (α):**
- Epoch 10: 0.91
- Epoch 1200: 1.42

Purity measures how concentrated the network's weights are on a small number of directions in parameter space. Increasing purity indicates increasing sparsity and structure. The growth from 0.91 to 1.42 is consistent with the Hamiltonian results, though the absolute values are lower.

**Spectral Entropy:**
- Stable at S ≈ 8.33 throughout training

Spectral entropy measures the complexity of the network's frequency representation. A stable value suggests the network maintained a consistent internal structure despite changes in other metrics. This is a signature of organized dynamics rather than chaotic optimization.

**Condition Number (κ):**
- Oscillating between ∞ and 1.0

The condition number measures the stability of the gradient covariance matrix. A value of ∞ indicates singular covariance (some directions have zero gradient), while 1.0 indicates perfect conditioning. The oscillation suggests marginal stability: the system is exploring saddle points rather than converging to a fixed point. This behavior appeared in the Hamiltonian experiments as well.

**Effective Planck Constant (ħ_eff):**
- Epoch 10: 0.16
- Epoch 1200: 0.59

The effective Planck constant emerged from the relation ħ_eff ∝ δ²λ/ω. In the Hamiltonian crystal, ħ_eff reached 10³⁴. Here it was less than 1. The interpretation of this quantity remains speculative.

### 3.4 What Did Not Happen

The crystal detection flag remained `False` throughout training. This flag triggers when α > 7.0 and δ < 0.1—neither condition was met.

More importantly, I did not measure topological invariants (Berry phases, winding numbers) because the system did not reach the crystallized state where these become meaningful. The checkpoint at epoch 1200 represents a partial crystal: structured enough to generalize, sparse enough to show decreasing δ, but not topologically protected.

---

## 4. Comparison with Hamiltonian Results

| Feature | Hamiltonian Crystal | Schrödinger (This Work) |
|---------|--------------------|-----------------------|
| Target δ | < 0.1 | 0.24 (incomplete) |
| Final δ | 0.37* | 0.24 |
| λ at end | 7.5 × 10³⁴ | 100 |
| Epochs | 7,196 | 1,200 |
| Topological invariants | Yes | Not measured |
| κ behavior | ∞ (singular) | Oscillating |

*The Hamiltonian result achieved δ < 0.1 during training before settling at 0.37 in the continuous crystal regime.

The Schrödinger system shows the correct qualitative behavior but requires substantially more computational resources to reach the target state.

---

## 5. Discussion

### 5.1 Why Quantum Systems May Be Harder

The Schrödinger equation introduces complications absent in classical mechanics:

**Phase coherence.** The wavefunction phase must be preserved exactly. Small discretization errors can destroy interference patterns. This constraint is stricter than anything in classical Hamiltonian dynamics.

**Complex-valued operations.** The spectral layers must handle real and imaginary channels consistently. Phase misalignment between channels could prevent crystallization.

**Global constraints.** Probability conservation is a global constraint (integral over all space equals 1), not a local one. The network must learn to enforce this globally.

These factors may raise the energy barrier for crystallization, requiring higher λ and longer training.

### 5.2 The Seed Mining Question

The seed mining revealed that no seeds achieved δ < 0.48 in the short screening run. In the Hamiltonian experiments, seed 32 started at δ ≈ 0.46. This difference is puzzling.

Possible explanations:
1. Quantum dynamics genuinely have a higher baseline δ
2. 40 epochs was insufficient for good seeds to distinguish themselves
3. The spectral architecture behaves differently for complex wavefunctions

I cannot distinguish these explanations from the current data. A more thorough seed mining protocol—longer screening runs, more seeds, or different selection criteria—might help.

### 5.3 What the Current State Represents

The epoch 1200 checkpoint is not a failure. It represents a reproducible intermediate state:
- Perfect generalization (100% validation accuracy)
- Decreasing δ trajectory
- Increasing structural organization (purity growth)
- Marginal stability (κ oscillation)

This is analogous to a supercooled liquid: ordered but not frozen. Whether further training would complete the crystallization remains an open question.

---

## 6. Limitations

**Incomplete experiment.** The run was interrupted at 1200 epochs. I cannot claim topological protection without reaching the crystallization threshold.

**N = 1.** Only one seed completed extended training. The results may not generalize.

**Arbitrary thresholds.** The crystal detection thresholds (α > 7.0, δ < 0.1) were inherited from Hamiltonian experiments without justification for quantum systems.

**Computational shortfall.** The Hamiltonian crystal required λ = 10³⁴ and 7,196 epochs. This experiment achieved λ = 10² and 1,200 epochs—far short on both dimensions.

**Architecture differences.** The input structure differs (2 channels for complex wavefunctions vs. 1 for classical fields), which may affect crystallization dynamics.

---

## 7. Conclusion

I applied the HPU-Core protocol to the Schrödinger equation. The discretization margin δ decreased from 0.49 to 0.24 over 1200 epochs, purity increased, and validation accuracy remained perfect. These are partial crystallization signatures.

The experiment did not achieve full crystallization. Based on the Hamiltonian results, completion would likely require:
- λ pressure of 10³⁰ or higher
- Training duration of 7,000+ epochs
- Possibly different seed selection criteria

The checkpoint at epoch 1200 is available for reproduction and as a starting point for extended training. Whether topological crystallization survives in quantum systems remains an open question.

---

## References

[1] Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

[2] Humayun, A. I., Balestriero, R., & Baraniuk, R. (2024). Deep Networks Always Grok and Here is Why. arXiv:2402.15555.

[3] Bereska, L., et al. (2024). Superposition as Lossy Compression. arXiv:2024.

[4] grisun0. (2025). Algorithmic Induction via Structural Weight Transfer. Zenodo. https://doi.org/10.5281/zenodo.18072858

[5] grisun0. (2025). From Boltzmann Stochasticity to Hamiltonian Integrability: Emergence of Topological Crystals and Synthetic Planck Constants. Zenodo. https://doi.org/10.5281/zenodo.18407920

[6] grisun0. (2025). Thermodynamic Grokking in Binary Parity (k=3): A First Look at 100 Seeds. Zenodo. https://doi.org/10.5281/zenodo.18489853

---

## Data Availability

Repository: https://github.com/grisuno/schrodinger

Checkpoint: `checkpoints_phase3/latest.pth` (epoch 5000)  
Best Checkpoint: `checkpoint_phase3_training_epoch_4994_20260216_142858` (epoch 4994)
Training logs: `results_schrodinger.txt`

---

*grisun0*  
*ORCID: 0009-0002-7622-3916*  
*February 16, 2026*


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
