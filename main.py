#!/usr/bin/env python3
"""
schrodinger_crystal.py

Author: Gris Iscomeback
Email: grisiscomeback@gmail.com
Date of creation: 2026
License: AGPL v3

Description:
Schrodinger Equation Grokking via Hamiltonian Topological Crystallization.
Four-phase protocol:
  Phase 1 - Batch size prospecting
  Phase 2 - Seed mining with decreasing delta criterion
  Phase 3 - Full training of best seed + batch size until grokking
  Phase 4 - Refinement via simulated annealing toward crystal state

Uses a pretrained Hamiltonian backbone (latest.pth) for efficient
Hamiltonian inference instead of direct spectral computation.
Lambda regularization pressure operates at float64 precision.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from collections import deque
import logging
import math
import copy


@dataclass
class Config:
    GRID_SIZE: int = 16
    HIDDEN_DIM: int = 32
    NUM_SPECTRAL_LAYERS: int = 2
    EXPANSION_DIM: int = 64
    SCHRODINGER_CHANNELS: int = 2
    POTENTIAL_DEPTH: float = 5.0
    POTENTIAL_WIDTH: float = 0.3
    NUM_EIGENSTATES: int = 8
    ENERGY_SCALE: float = 1.0
    WAVEFUNCTION_NORM_TARGET: float = 1.0

    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 5000
    REFINEMENT_EPOCHS: int = 2000
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    MAX_CHECKPOINTS: int = 10
    TARGET_ACCURACY: float = 0.95
    TIME_STEPS: int = 2
    DT: float = 0.01
    TRAIN_RATIO: float = 0.7
    NUM_SAMPLES: int = 200
    GRADIENT_CLIP_NORM: float = 1.0
    NOISE_AMPLITUDE: float = 0.01
    NOISE_INTERVAL_EPOCHS: int = 25
    MOMENTUM: float = 0.9
    COSINE_ANNEALING_ETA_MIN_FACTOR: float = 0.01
    MSE_THRESHOLD: float = 0.05
    CYCLIC_LR_BASE_FACTOR: float = 0.01
    CYCLIC_LR_MAX_FACTOR: float = 2.0
    CYCLIC_LR_STEP_SIZE: int = 50

    ENTROPY_BINS: int = 50
    PCA_COMPONENTS: int = 2
    KDE_BANDWIDTH: str = 'scott'
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    ENTROPY_EPS: float = 1e-10
    HBAR: float = 1e-6
    HBAR_PHYSICAL: float = 1.0545718e-34
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    DISCRETIZATION_MARGIN: float = 0.1
    TARGET_SLOTS: int = 7
    KAPPA_MAX_DIM: int = 10000
    EIGENVALUE_TOL: float = 1e-10
    KAPPA_GRADIENT_BATCHES: int = 5
    ALPHA_CRYSTAL_THRESHOLD: float = 7.0
    SPECTRAL_PEAK_LIMIT: int = 10
    SPECTRAL_POWER_LIMIT: int = 100
    PARAM_FLATTEN_LIMIT: int = 1000
    GRADIENT_BUFFER_LIMIT: int = 500
    GRADIENT_BUFFER_WINDOW: int = 10
    LOSS_HISTORY_WINDOW: int = 50
    GRADIENT_BUFFER_MAXLEN: int = 50
    LOSS_HISTORY_MAXLEN: int = 100
    TEMP_HISTORY_MAXLEN: int = 100
    CV_HISTORY_MAXLEN: int = 100
    CV_THRESHOLD: float = 1.0
    WEIGHT_METRIC_DIM_LIMIT: int = 256

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = 'INFO'
    RESULTS_DIR: str = 'schrodinger_results'

    MINING_MAX_ATTEMPTS: int = 200
    MINING_START_SEED: int = 1
    MINING_GLASS_PATIENCE_EPOCHS: int = 50
    MINING_PROSPECT_EPOCHS: int = 40
    MINING_PROSPECT_DELTA_EPOCH_INTERVAL: int = 10
    MINING_TARGET_LC: float = 0.01
    MINING_TARGET_SP: float = 0.01
    MINING_TARGET_KAPPA: float = 1.01
    MINING_TARGET_DELTA: float = 0.001
    MINING_TARGET_TEMP: float = 1e-10
    MINING_TARGET_CV: float = 1e-10

    BATCH_CANDIDATES: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    BATCH_PROSPECT_EPOCHS: int = 30
    BATCH_PROSPECT_SEED: int = 42

    LAMBDA_INITIAL: float = 1.0
    LAMBDA_MAX: float = 1e35
    LAMBDA_GROWTH_FACTOR: float = 10.0
    LAMBDA_GROWTH_INTERVAL_EPOCHS: int = 500
    LAMBDA_PRECISION_DTYPE: str = 'float64'

    ANNEALING_INITIAL_TEMPERATURE: float = 1.0
    ANNEALING_FINAL_TEMPERATURE: float = 1e-6
    ANNEALING_COOLING_RATE: float = 0.995
    ANNEALING_RESTART_THRESHOLD: float = 0.01

    GROKKING_TRAIN_ACC_THRESHOLD: float = 0.99
    GROKKING_VAL_ACC_THRESHOLD: float = 0.95
    GROKKING_PATIENCE: int = 200
    GROKKING_DELTA_SLOPE_WINDOW: int = 50
    GROKKING_DELTA_SLOPE_THRESHOLD: float = -1e-6

    BACKBONE_CHECKPOINT_PATH: str = 'weights/latest.pth'
    BACKBONE_ENABLED: bool = True

    LOG_INTERVAL_EPOCHS: int = 10

    NORMALIZATION_EPS: float = 1e-8
    POYNTING_OUTER_LIMIT: int = 50


class SeedManager:
    @staticmethod
    def set_seed(seed: int, device: str = Config.DEVICE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class LoggerFactory:
    @staticmethod
    def create_logger(name: str, level: str = Config.LOG_LEVEL) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class IAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class IMetricsCalculator(ABC):
    @abstractmethod
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class HamiltonianOperator:
    def __init__(self, grid_size: int = Config.GRID_SIZE):
        self.grid_size = grid_size
        self._precompute_spectral_operators()

    def _precompute_spectral_operators(self):
        kx = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.laplacian_spectrum = -(KX**2 + KY**2).float()

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        field_fft = torch.fft.fft2(field)
        laplacian_fft = field_fft * self.laplacian_spectrum.to(field.device)
        return torch.fft.ifft2(laplacian_fft).real

    def time_evolution(self, field: torch.Tensor, dt: float = Config.DT) -> torch.Tensor:
        hamiltonian_action = self.apply(field)
        evolved = field + hamiltonian_action * dt
        norm_original = torch.norm(field) + Config.NORMALIZATION_EPS
        norm_evolved = torch.norm(evolved) + Config.NORMALIZATION_EPS
        return evolved / norm_evolved * norm_original


class SpectralLayer(nn.Module):
    def __init__(self, channels: int, grid_size: int):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        batch, channels, freq_h, freq_w = x_fft.shape
        kernel_real = self.kernel_real.mean(dim=0)
        kernel_imag = self.kernel_imag.mean(dim=0)
        kernel_real_exp = kernel_real.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_imag_exp = kernel_imag.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_real_interp = F.interpolate(
            kernel_real_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        kernel_imag_interp = F.interpolate(
            kernel_imag_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        real_part = x_fft.real * kernel_real_interp - x_fft.imag * kernel_imag_interp
        imag_part = x_fft.real * kernel_imag_interp + x_fft.imag * kernel_real_interp
        output_fft = torch.complex(real_part, imag_part)
        output = torch.fft.irfft2(output_fft, s=(self.grid_size, self.grid_size))
        return output


class HamiltonianBackbone(nn.Module):
    def __init__(
        self,
        grid_size: int = Config.GRID_SIZE,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_spectral_layers: int = Config.NUM_SPECTRAL_LAYERS
    ):
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(hidden_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        return self.output_proj(x).squeeze(1)


class HamiltonianInferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("HamiltonianInferenceEngine")
        self.backbone = None
        self.fallback_operator = HamiltonianOperator(config.GRID_SIZE)
        self._try_load_backbone()

    def _try_load_backbone(self):
        if not self.config.BACKBONE_ENABLED:
            self.logger.info("Backbone disabled, using analytical Hamiltonian operator")
            return
        checkpoint_path = self.config.BACKBONE_CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            self.logger.info(
                f"Backbone checkpoint not found at {checkpoint_path}, "
                "using analytical Hamiltonian operator"
            )
            return
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
            self.backbone = HamiltonianBackbone(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            if 'model_state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.backbone.load_state_dict(checkpoint)
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.logger.info(f"Backbone loaded from {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load backbone: {e}, using analytical operator")
            self.backbone = None

    def apply_hamiltonian(self, field: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            with torch.no_grad():
                return self.backbone(field.to(self.config.DEVICE))
        return self.fallback_operator.apply(field)

    def time_evolve(self, field: torch.Tensor, dt: float = Config.DT) -> torch.Tensor:
        if self.backbone is not None:
            with torch.no_grad():
                evolved = self.backbone(field.to(self.config.DEVICE))
                norm_original = torch.norm(field) + Config.NORMALIZATION_EPS
                norm_evolved = torch.norm(evolved) + Config.NORMALIZATION_EPS
                return evolved / norm_evolved * norm_original
        return self.fallback_operator.time_evolution(field, dt)


class SchrodingerPotentialGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.grid_size = config.GRID_SIZE

    def harmonic_potential(self) -> torch.Tensor:
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx = np.pi
        cy = np.pi
        return 0.5 * self.config.POTENTIAL_DEPTH * ((X - cx)**2 + (Y - cy)**2) / (np.pi**2)

    def double_well_potential(self) -> torch.Tensor:
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx = np.pi
        w = self.config.POTENTIAL_WIDTH * np.pi
        return self.config.POTENTIAL_DEPTH * ((X - cx)**2 / w**2 - 1)**2

    def coulomb_like_potential(self) -> torch.Tensor:
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx, cy = np.pi, np.pi
        r = torch.sqrt((X - cx)**2 + (Y - cy)**2) + self.config.POTENTIAL_WIDTH
        return -self.config.POTENTIAL_DEPTH / r

    def periodic_lattice_potential(self) -> torch.Tensor:
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return self.config.POTENTIAL_DEPTH * (torch.cos(2 * X) + torch.cos(2 * Y))

    def generate_mixed_potential(self, seed: int) -> torch.Tensor:
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        potentials = [
            self.harmonic_potential(),
            self.double_well_potential(),
            self.coulomb_like_potential(),
            self.periodic_lattice_potential()
        ]
        result = torch.zeros(self.grid_size, self.grid_size)
        for w, v in zip(weights, potentials):
            result += w * v
        return result


class SchrodingerDataset(Dataset):
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        seed: int = Config.RANDOM_SEED,
    ):
        self.config = config
        self.num_samples = config.NUM_SAMPLES
        self.grid_size = config.GRID_SIZE
        self.train_ratio = config.TRAIN_RATIO
        self.hamiltonian_engine = hamiltonian_engine
        self.potential_generator = SchrodingerPotentialGenerator(config)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.initial_states = []
        self.target_states = []
        self.potentials = []
        self.energies = []

        for i in range(self.num_samples):
            potential = self.potential_generator.generate_mixed_potential(seed + i)
            psi_real, psi_imag, energy = self._solve_schrodinger_sample(potential, seed + i)
            initial = torch.stack([psi_real, psi_imag], dim=0)
            evolved_real, evolved_imag = self._time_evolve_wavefunction(
                psi_real, psi_imag, potential, energy
            )
            target = torch.stack([evolved_real, evolved_imag], dim=0)
            self.initial_states.append(initial)
            self.target_states.append(target)
            self.potentials.append(potential)
            self.energies.append(energy)

        self.initial_states = torch.stack(self.initial_states)
        self.target_states = torch.stack(self.target_states)
        self.potentials = torch.stack(self.potentials)
        self.energies = torch.tensor(self.energies)

        split_idx = int(self.num_samples * self.train_ratio)
        self.train_states = self.initial_states[:split_idx]
        self.train_targets = self.target_states[:split_idx]
        self.val_states = self.initial_states[split_idx:]
        self.val_targets = self.target_states[split_idx:]

    def _solve_schrodinger_sample(
        self, potential: torch.Tensor, sample_seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        h_field = self.hamiltonian_engine.apply_hamiltonian(
            torch.randn(self.grid_size, self.grid_size)
        )
        if h_field.dim() > 2:
            h_field = h_field.squeeze()
        kinetic = -0.5 * h_field
        hamiltonian_matrix = kinetic + potential
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian_matrix)
        except Exception:
            eigenvalues = torch.zeros(self.grid_size)
            eigenvectors = torch.eye(self.grid_size)
        rng = np.random.RandomState(sample_seed)
        n_states = min(self.config.NUM_EIGENSTATES, self.grid_size)
        state_idx = rng.randint(0, n_states)
        energy = eigenvalues[state_idx].item() * self.config.ENERGY_SCALE
        psi_column = eigenvectors[:, state_idx]
        psi_2d = psi_column.unsqueeze(1).expand(-1, self.grid_size)
        perturbation = torch.randn_like(psi_2d) * 0.1
        psi_2d = psi_2d + perturbation
        norm = torch.sqrt(torch.sum(psi_2d**2)) + Config.NORMALIZATION_EPS
        psi_2d = psi_2d / norm * self.config.WAVEFUNCTION_NORM_TARGET
        phase = torch.randn(self.grid_size, self.grid_size) * 0.5
        psi_real = psi_2d * torch.cos(phase)
        psi_imag = psi_2d * torch.sin(phase)
        return psi_real, psi_imag, energy

    def _time_evolve_wavefunction(
        self,
        psi_real: torch.Tensor,
        psi_imag: torch.Tensor,
        potential: torch.Tensor,
        energy: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = self.config.DT
        for _ in range(self.config.TIME_STEPS):
            h_psi_real = self.hamiltonian_engine.apply_hamiltonian(psi_real)
            if h_psi_real.dim() > 2:
                h_psi_real = h_psi_real.squeeze()
            h_psi_imag = self.hamiltonian_engine.apply_hamiltonian(psi_imag)
            if h_psi_imag.dim() > 2:
                h_psi_imag = h_psi_imag.squeeze()
            kinetic_real = -0.5 * h_psi_real
            kinetic_imag = -0.5 * h_psi_imag
            total_h_real = kinetic_real + potential * psi_real
            total_h_imag = kinetic_imag + potential * psi_imag
            new_real = psi_real + dt * total_h_imag
            new_imag = psi_imag - dt * total_h_real
            norm = torch.sqrt(
                torch.sum(new_real**2 + new_imag**2)
            ) + Config.NORMALIZATION_EPS
            target_norm = torch.sqrt(
                torch.sum(psi_real**2 + psi_imag**2)
            ) + Config.NORMALIZATION_EPS
            new_real = new_real / norm * target_norm
            new_imag = new_imag / norm * target_norm
            psi_real = new_real
            psi_imag = new_imag
        return psi_real, psi_imag

    def __len__(self):
        return len(self.train_states)

    def __getitem__(self, idx):
        return self.train_states[idx], self.train_targets[idx]

    def get_validation_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.val_states, self.val_targets


class SchrodingerSpectralNetwork(nn.Module):
    def __init__(
        self,
        grid_size: int = Config.GRID_SIZE,
        hidden_dim: int = Config.HIDDEN_DIM,
        expansion_dim: int = Config.EXPANSION_DIM,
        num_spectral_layers: int = Config.NUM_SPECTRAL_LAYERS,
        input_channels: int = Config.SCHRODINGER_CHANNELS,
        output_channels: int = Config.SCHRODINGER_CHANNELS
    ):
        super().__init__()
        self.grid_size = grid_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(expansion_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class LocalComplexityAnalyzer:
    @staticmethod
    def compute_local_complexity(weights: torch.Tensor, epsilon: float = 1e-6) -> float:
        if weights.numel() == 0:
            return 0.0
        w = weights.detach().flatten()
        w = w / (torch.norm(w) + epsilon)
        w_expanded = w.unsqueeze(0)
        similarities = F.cosine_similarity(w_expanded, w_expanded.unsqueeze(1), dim=2)
        mask = ~torch.eye(similarities.size(0), device=similarities.device, dtype=torch.bool)
        if mask.sum() == 0:
            return 0.0
        avg_similarity = (similarities.abs() * mask).sum() / mask.sum()
        lc = 1.0 - avg_similarity.item()
        return max(0.0, min(1.0, lc))


class SuperpositionAnalyzer:
    @staticmethod
    def compute_superposition(weights: torch.Tensor) -> float:
        w = weights.detach()
        if w.dim() > 2:
            w = w.reshape(w.size(0), -1)
        if w.size(0) < 2 or w.size(1) < 2:
            return 0.0
        try:
            correlation_matrix = torch.corrcoef(w)
            correlation_matrix = correlation_matrix.nan_to_num(nan=0.0)
            n = correlation_matrix.size(0)
            mask = ~torch.eye(n, device=correlation_matrix.device, dtype=torch.bool)
            if mask.sum() == 0:
                return 0.0
            avg_correlation = (correlation_matrix.abs() * mask).sum() / mask.sum()
            return avg_correlation.item()
        except Exception:
            return 0.0


class CrystallographyMetricsCalculator(IMetricsCalculator):
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("CrystallographyMetrics")

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        val_x = kwargs.get('val_x')
        val_y = kwargs.get('val_y')
        return self.compute_all_metrics(model, val_x, val_y)

    def compute_kappa(
        self,
        model: nn.Module,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        num_batches: int = None
    ) -> float:
        if num_batches is None:
            num_batches = self.config.KAPPA_GRADIENT_BATCHES
        model.eval()
        grads = []
        for i in range(num_batches):
            try:
                model.zero_grad()
                noise_scale = self.config.NOISE_AMPLITUDE * (i + 1) / num_batches
                val_x_perturbed = val_x + torch.randn_like(val_x) * noise_scale
                outputs = model(val_x_perturbed)
                loss = F.mse_loss(outputs, val_y)
                loss.backward()
                grad_list = []
                for p in model.parameters():
                    if p.grad is not None and p.grad.numel() > 0:
                        grad_list.append(p.grad.flatten())
                if grad_list:
                    grad_vector = torch.cat(grad_list)
                    if torch.isfinite(grad_vector).all():
                        grads.append(grad_vector.detach())
            except Exception as e:
                self.logger.debug(f"Gradient computation failed batch {i}: {e}")
                continue
        if len(grads) < 2:
            return float('inf')
        grads_tensor = torch.stack(grads)
        n_samples, n_dims = grads_tensor.shape
        if n_dims > self.config.KAPPA_MAX_DIM:
            indices = torch.randperm(n_dims, device=grads_tensor.device)[:self.config.KAPPA_MAX_DIM]
            grads_tensor = grads_tensor[:, indices]
            n_dims = self.config.KAPPA_MAX_DIM
        try:
            if n_samples < n_dims:
                gram = torch.mm(grads_tensor, grads_tensor.t()) / max(n_samples - 1, 1)
                eigenvals = torch.linalg.eigvalsh(gram)
            else:
                cov = torch.cov(grads_tensor.t())
                eigenvals = torch.linalg.eigvalsh(cov).real
            eigenvals = eigenvals[eigenvals > self.config.EIGENVALUE_TOL]
            if len(eigenvals) == 0:
                return float('inf')
            return (eigenvals.max() / eigenvals.min()).item()
        except Exception as e:
            self.logger.debug(f"Eigenvalue computation failed: {e}")
            return float('inf')

    def compute_discretization_margin(self, model: nn.Module) -> float:
        margins = []
        for param in model.parameters():
            if param.numel() > 0:
                margin = (param.data - param.data.round()).abs().max().item()
                margins.append(margin)
        return max(margins) if margins else 0.0

    def compute_alpha_purity(self, model: nn.Module) -> float:
        delta = self.compute_discretization_margin(model)
        if delta < self.config.MIN_VARIANCE_THRESHOLD:
            return 20.0
        return -np.log(delta + self.config.ENTROPY_EPS)

    def compute_kappa_quantum(self, model: nn.Module) -> float:
        flat_params = []
        for param in model.parameters():
            if param.numel() > 0:
                flat_params.append(param.data.detach().flatten())
        if not flat_params:
            return 1.0
        W = torch.cat(flat_params)[:self.config.KAPPA_MAX_DIM]
        n = W.numel()
        if n < 2:
            return 1.0
        params_centered = W - W.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + self.config.HBAR * torch.eye(n, device=W.device)
        try:
            eigenvals = torch.linalg.eigvalsh(cov_matrix)
            eigenvals = eigenvals[eigenvals > self.config.HBAR]
            return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
        except Exception:
            return 1.0

    def compute_poynting_vector(self, model: nn.Module) -> Dict[str, Any]:
        all_params = []
        for param in model.parameters():
            if param is not None and param.numel() > 0:
                all_params.append(param.data.detach().flatten())
        if not all_params:
            return {
                'poynting_magnitude': 0.0,
                'energy_distribution': {},
                'is_radiating': False,
                'field_orthogonality': 0.0
            }
        E = torch.cat(all_params)[:self.config.PARAM_FLATTEN_LIMIT]
        state_dict = model.state_dict()
        spectral_norms = []
        spectral_indices = set()
        for key in state_dict.keys():
            if key.startswith('spectral_layers.'):
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        spectral_indices.add(idx)
                    except ValueError:
                        continue
        for idx in sorted(spectral_indices):
            layer_param_keys = [
                k for k in state_dict.keys()
                if k.startswith(f'spectral_layers.{idx}.')
            ]
            if layer_param_keys:
                layer_params = [state_dict[k] for k in layer_param_keys]
                concatenated = torch.cat([p.flatten() for p in layer_params])
                layer_norm = torch.norm(concatenated)
                spectral_norms.append(layer_norm)
        if len(spectral_norms) > 1:
            differences = []
            for i in range(len(spectral_norms) - 1):
                diff = torch.abs(spectral_norms[i] - spectral_norms[i + 1])
                differences.append(diff)
            H_magnitude = torch.stack(differences).sum()
        else:
            H_magnitude = torch.tensor(0.0, device=E.device)
        poynting_magnitude = torch.norm(E) * H_magnitude * self.config.ENERGY_FLOW_SCALE
        energy_distribution = {
            'total_norm': float(torch.norm(E).item()),
            'spectral_total': float(sum(sn.item() for sn in spectral_norms)) if spectral_norms else 0.0,
            'n_spectral_layers': len(spectral_norms)
        }
        return {
            'poynting_magnitude': float(poynting_magnitude.item()),
            'energy_distribution': energy_distribution,
            'is_radiating': float(poynting_magnitude.item()) > self.config.POYNTING_THRESHOLD,
            'field_orthogonality': float(H_magnitude.item())
        }

    def compute_hbar_effective(self, model: nn.Module, lambda_pressure: float) -> float:
        delta = self.compute_discretization_margin(model)
        if lambda_pressure <= 0:
            return 0.0
        omega = math.sqrt(abs(lambda_pressure))
        if omega < self.config.NORMALIZATION_EPS:
            return 0.0
        return (delta**2 * lambda_pressure) / omega

    def compute_all_metrics(
        self,
        model: nn.Module,
        val_x: torch.Tensor,
        val_y: torch.Tensor
    ) -> Dict[str, Any]:
        try:
            delta = self.compute_discretization_margin(model)
            alpha = self.compute_alpha_purity(model)
        except Exception as e:
            self.logger.warning(f"Basic crystallography failed: {e}")
            delta, alpha = 1.0, 0.0

        def safe_compute(func, *args, default=None, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.debug(f"{func.__name__} failed: {e}")
                return default

        kappa = safe_compute(self.compute_kappa, model, val_x, val_y, default=float('inf'))
        kappa_q = safe_compute(self.compute_kappa_quantum, model, default=1.0)
        poynting = safe_compute(
            self.compute_poynting_vector, model,
            default={
                'poynting_magnitude': 0.0,
                'is_radiating': False,
                'energy_distribution': {},
                'field_orthogonality': 0.0
            }
        )
        metrics = {
            'kappa': kappa,
            'delta': delta,
            'alpha': alpha,
            'kappa_q': kappa_q,
            'poynting': poynting
        }
        metrics['purity_index'] = 1.0 - delta
        metrics['is_crystal'] = alpha > self.config.ALPHA_CRYSTAL_THRESHOLD
        if isinstance(poynting, dict):
            metrics['energy_flow'] = poynting.get('poynting_magnitude', 0.0)
        else:
            metrics['energy_flow'] = 0.0
        return metrics


class ThermodynamicMetricsCalculator(IMetricsCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        gradient_buffer = kwargs.get('gradient_buffer', [])
        learning_rate = kwargs.get('learning_rate', self.config.LEARNING_RATE)
        loss_history = kwargs.get('loss_history', [])
        temp_history = kwargs.get('temp_history', [])
        temperature = self.compute_effective_temperature(gradient_buffer, learning_rate)
        cv, is_transition = self.compute_specific_heat(loss_history, temp_history)
        return {
            'temperature': temperature,
            'specific_heat': cv,
            'is_phase_transition': is_transition
        }

    def compute_effective_temperature(
        self, gradient_buffer: Any, learning_rate: float
    ) -> float:
        buf = list(gradient_buffer) if not isinstance(gradient_buffer, list) else gradient_buffer
        if len(buf) < 2:
            return 0.0
        grads = []
        window = self.config.GRADIENT_BUFFER_WINDOW
        limit = self.config.GRADIENT_BUFFER_LIMIT
        for g in buf[-window:]:
            flat = g.detach().flatten()
            if flat.numel() > 0:
                grads.append(flat[:limit])
        if not grads:
            return 0.0
        try:
            grads_stacked = torch.stack(grads)
            second_moment = torch.mean(torch.norm(grads_stacked, dim=1)**2)
            first_moment_sq = torch.norm(torch.mean(grads_stacked, dim=0))**2
            variance = second_moment - first_moment_sq
            return float((learning_rate / 2.0) * variance)
        except Exception:
            return 0.0

    def compute_specific_heat(
        self, loss_history: Any, temp_history: Any
    ) -> Tuple[float, bool]:
        l_hist = list(loss_history) if not isinstance(loss_history, list) else loss_history
        t_hist = list(temp_history) if not isinstance(temp_history, list) else temp_history
        window = self.config.LOSS_HISTORY_WINDOW
        if len(l_hist) < 2 or len(t_hist) < 2:
            return 0.0, False
        u_var = np.var(l_hist[-window:])
        t_mean = np.mean(t_hist[-window:]) + self.config.NORMALIZATION_EPS
        cv = u_var / (t_mean**2)
        is_latent_crystallization = cv > self.config.CV_THRESHOLD
        return float(cv), is_latent_crystallization


class SpectroscopyMetricsCalculator(IMetricsCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        coeffs = {name: param.data for name, param in model.named_parameters()}
        return self.compute_weight_diffraction(coeffs)

    def compute_weight_diffraction(self, coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        limit = self.config.PARAM_FLATTEN_LIMIT
        W = torch.cat([c.detach().flatten()[:limit] for c in coeffs.values()])
        fft_spectrum = torch.fft.fft(W)
        power_spectrum = torch.abs(fft_spectrum)**2
        peaks = []
        threshold = torch.mean(power_spectrum) + 2 * torch.std(power_spectrum)
        peak_limit = self.config.SPECTRAL_PEAK_LIMIT
        for i, power in enumerate(power_spectrum):
            if power > threshold and len(peaks) < peak_limit:
                peaks.append({'frequency': i, 'intensity': float(power)})
        is_crystalline = len(peaks) > 0 and len(peaks) < len(power_spectrum) // 2
        return {
            'bragg_peaks': peaks,
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': float(self._compute_spectral_entropy(power_spectrum)),
            'num_peaks': len(peaks)
        }

    @staticmethod
    def _compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy)


class LambdaPressureScheduler:
    def __init__(self, config: Config):
        self.config = config
        self._lambda = np.float64(config.LAMBDA_INITIAL)
        self._lambda_max = np.float64(config.LAMBDA_MAX)
        self._growth_factor = np.float64(config.LAMBDA_GROWTH_FACTOR)
        self._growth_interval = config.LAMBDA_GROWTH_INTERVAL_EPOCHS
        self.logger = LoggerFactory.create_logger("LambdaPressureScheduler")

    @property
    def current_lambda(self) -> np.float64:
        return self._lambda

    def step(self, epoch: int):
        if epoch > 0 and epoch % self._growth_interval == 0:
            new_lambda = self._lambda * self._growth_factor
            if new_lambda <= self._lambda_max:
                self._lambda = np.float64(new_lambda)
                self.logger.info(
                    f"Lambda pressure increased to {self._lambda:.6e} at epoch {epoch}"
                )

    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        delta = torch.tensor(0.0, device=self.config.DEVICE)
        for param in model.parameters():
            if param.numel() > 0:
                margin = (param - param.round()).abs().max()
                delta = torch.max(delta, margin)
        return delta * float(self._lambda)

    def set_lambda(self, value: float):
        self._lambda = np.float64(value)


class AnnealingScheduler:
    def __init__(self, config: Config):
        self.config = config
        self._temperature = config.ANNEALING_INITIAL_TEMPERATURE
        self._cooling_rate = config.ANNEALING_COOLING_RATE
        self._final_temp = config.ANNEALING_FINAL_TEMPERATURE
        self.logger = LoggerFactory.create_logger("AnnealingScheduler")

    @property
    def temperature(self) -> float:
        return self._temperature

    def step(self):
        self._temperature = max(
            self._temperature * self._cooling_rate,
            self._final_temp
        )

    def accept_perturbation(self, delta_loss: float) -> bool:
        if delta_loss < 0:
            return True
        if self._temperature < self.config.NORMALIZATION_EPS:
            return False
        probability = math.exp(-delta_loss / self._temperature)
        return np.random.random() < probability

    def should_restart(self, current_delta: float, best_delta: float) -> bool:
        return (current_delta - best_delta) > self.config.ANNEALING_RESTART_THRESHOLD


class TrainingMetricsMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.metrics_history = {
            'epoch': [], 'loss': [], 'val_loss': [], 'val_acc': [],
            'train_acc': [], 'lc': [], 'sp': [], 'alpha': [],
            'kappa': [], 'kappa_q': [], 'delta': [], 'temperature': [],
            'specific_heat': [], 'poynting_magnitude': [],
            'energy_flow': [], 'purity_index': [], 'is_crystal': [],
            'lambda_pressure': [], 'hbar_effective': [],
            'spectral_entropy': [], 'num_bragg_peaks': [],
            'learning_rate': [], 'annealing_temp': [],
            'delta_slope': [], 'norm_conservation_error': []
        }
        self.gradient_buffer = deque(maxlen=config.GRADIENT_BUFFER_MAXLEN)
        self.loss_history = deque(maxlen=config.LOSS_HISTORY_MAXLEN)
        self.temp_history = deque(maxlen=config.TEMP_HISTORY_MAXLEN)
        self.cv_history = deque(maxlen=config.CV_HISTORY_MAXLEN)

    def update_metrics(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        if 'loss' in kwargs:
            self.loss_history.append(kwargs['loss'])
        if 'temperature' in kwargs:
            self.temp_history.append(kwargs['temperature'])
        if 'specific_heat' in kwargs:
            self.cv_history.append(kwargs['specific_heat'])

    def compute_delta_slope(self) -> float:
        window = self.config.GROKKING_DELTA_SLOPE_WINDOW
        deltas = self.metrics_history.get('delta', [])
        if len(deltas) < window:
            return 0.0
        recent = deltas[-window:]
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        if np.std(x) < 1e-15:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def format_progress_bar(self, epoch: int, total_epochs: int, phase: str) -> str:
        h = self.metrics_history
        idx = -1
        parts = [
            f"[{phase}] Epoch {epoch}/{total_epochs}",
        ]
        def safe_get(key):
            vals = h.get(key, [])
            return vals[idx] if vals else 0.0

        loss = safe_get('loss')
        val_loss = safe_get('val_loss')
        val_acc = safe_get('val_acc')
        train_acc = safe_get('train_acc')
        lc = safe_get('lc')
        sp = safe_get('sp')
        alpha = safe_get('alpha')
        kappa = safe_get('kappa')
        kappa_q = safe_get('kappa_q')
        delta = safe_get('delta')
        temp = safe_get('temperature')
        cv = safe_get('specific_heat')
        poynting = safe_get('poynting_magnitude')
        lam = safe_get('lambda_pressure')
        hbar = safe_get('hbar_effective')
        s_ent = safe_get('spectral_entropy')
        n_peaks = safe_get('num_bragg_peaks')
        lr = safe_get('learning_rate')
        a_temp = safe_get('annealing_temp')
        d_slope = safe_get('delta_slope')
        norm_err = safe_get('norm_conservation_error')
        purity = safe_get('purity_index')
        is_cryst = safe_get('is_crystal')

        parts.append(
            f"Loss={loss:.6f} ValLoss={val_loss:.6f} "
            f"TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}"
        )
        parts.append(
            f"LC={lc:.4f} SP={sp:.4f} Alpha={alpha:.4f} "
            f"Kappa={kappa:.2e} Kappa_q={kappa_q:.2e}"
        )
        parts.append(
            f"Delta={delta:.6f} Purity={purity:.4f} Crystal={is_cryst}"
        )
        parts.append(
            f"T_eff={temp:.2e} C_v={cv:.2e} Poynting={poynting:.2e} "
            f"E_flow={safe_get('energy_flow'):.2e}"
        )
        parts.append(
            f"Lambda={lam:.2e} hbar_eff={hbar:.2e} "
            f"S_spectral={s_ent:.4f} Bragg_peaks={int(n_peaks)}"
        )
        parts.append(
            f"LR={lr:.2e} AnnealT={a_temp:.2e} "
            f"dDelta/dt={d_slope:.2e} NormErr={norm_err:.2e}"
        )
        return " | ".join(parts)


class CheckpointManager:
    def __init__(self, config: Config, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.interval_minutes = config.CHECKPOINT_INTERVAL_MINUTES
        self.max_checkpoints = config.MAX_CHECKPOINTS
        self.last_checkpoint_time = time.time()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_files = []

    def should_save_checkpoint(self) -> bool:
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.interval_minutes

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        phase: str = "training",
        lambda_value: float = 0.0,
        config_snapshot: Dict[str, Any] = None
    ) -> str:
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'lambda_pressure': float(lambda_value),
            'config': config_snapshot if config_snapshot else asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{phase}_epoch_{epoch}_{timestamp}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_files.append(checkpoint_path)
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_file = self.checkpoint_files.pop(0)
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        self.last_checkpoint_time = time.time()
        return checkpoint_path


class GlassStateDetector:
    def __init__(self, config: Config):
        self.config = config
        self.patience_epochs = config.MINING_GLASS_PATIENCE_EPOCHS
        self.metrics_buffer = deque(maxlen=self.patience_epochs)
        self.logger = LoggerFactory.create_logger("GlassStateDetector")

    def should_stop(
        self, epoch: int, lc: float, sp: float,
        kappa: float, delta: float, temp: float, cv: float
    ) -> bool:
        self.metrics_buffer.append({
            'epoch': epoch, 'lc': lc, 'sp': sp,
            'kappa': kappa, 'delta': delta, 'temp': temp, 'cv': cv
        })
        if epoch > self.patience_epochs:
            recent = list(self.metrics_buffer)[-self.patience_epochs:]
            avg_lc = np.mean([m['lc'] for m in recent])
            avg_sp = np.mean([m['sp'] for m in recent])
            avg_kappa = np.mean([m['kappa'] for m in recent])
            avg_delta = np.mean([m['delta'] for m in recent])
            avg_temp = np.mean([m['temp'] for m in recent])
            avg_cv = np.mean([m['cv'] for m in recent])
            is_glass = (
                avg_lc > self.config.MINING_TARGET_LC or
                avg_sp > self.config.MINING_TARGET_SP or
                avg_kappa > self.config.MINING_TARGET_KAPPA or
                avg_delta > self.config.MINING_TARGET_DELTA or
                avg_temp > self.config.MINING_TARGET_TEMP or
                avg_cv > self.config.MINING_TARGET_CV
            )
            return is_glass
        if epoch == self.patience_epochs:
            if (lc > self.config.MINING_TARGET_LC or
                sp > self.config.MINING_TARGET_SP or
                kappa > self.config.MINING_TARGET_KAPPA or
                delta > self.config.MINING_TARGET_DELTA or
                temp > self.config.MINING_TARGET_TEMP or
                cv > self.config.MINING_TARGET_CV):
                return True
        return False

    def is_crystal_formed(
        self, lc: float, sp: float, kappa: float,
        delta: float, temp: float, cv: float
    ) -> bool:
        return (
            lc < self.config.MINING_TARGET_LC and
            sp < self.config.MINING_TARGET_SP and
            kappa < self.config.MINING_TARGET_KAPPA and
            delta < self.config.MINING_TARGET_DELTA and
            temp < self.config.MINING_TARGET_TEMP and
            cv < self.config.MINING_TARGET_CV
        )


class WeightIntegrityChecker:
    @staticmethod
    def check(model: nn.Module) -> Dict[str, Any]:
        has_nan = False
        has_inf = False
        total_params = 0
        nan_count = 0
        inf_count = 0
        for name, param in model.named_parameters():
            data = param.data
            numel = data.numel()
            total_params += numel
            n_nan = torch.isnan(data).sum().item()
            n_inf = torch.isinf(data).sum().item()
            if n_nan > 0:
                has_nan = True
                nan_count += n_nan
            if n_inf > 0:
                has_inf = True
                inf_count += n_inf
        corruption_ratio = (nan_count + inf_count) / total_params if total_params > 0 else 0.0
        return {
            'is_valid': not (has_nan or has_inf),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'total_params': total_params,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'corruption_ratio': corruption_ratio
        }


class TrainingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("TrainingEngine")
        self.criterion = nn.MSELoss()
        self.lc_analyzer = LocalComplexityAnalyzer()
        self.sp_analyzer = SuperpositionAnalyzer()
        self.crystal_calc = CrystallographyMetricsCalculator(config)
        self.thermo_calc = ThermodynamicMetricsCalculator(config)
        self.spectro_calc = SpectroscopyMetricsCalculator(config)

    def compute_weight_metrics(self, model: nn.Module) -> Tuple[float, float]:
        lc_values = []
        sp_values = []
        limit = self.config.WEIGHT_METRIC_DIM_LIMIT
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param[:min(param.size(0), limit), :min(param.size(1), limit)]
                lc = self.lc_analyzer.compute_local_complexity(w)
                sp = self.sp_analyzer.compute_superposition(w)
                lc_values.append(lc)
                sp_values.append(sp)
        lc = np.mean(lc_values) if lc_values else 0.0
        sp = np.mean(sp_values) if sp_values else 0.0
        return lc, sp

    def compute_norm_conservation_error(
        self, model: nn.Module, val_x: torch.Tensor
    ) -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(val_x)
            input_norms = torch.norm(val_x.view(val_x.size(0), -1), dim=1)
            output_norms = torch.norm(outputs.view(outputs.size(0), -1), dim=1)
            relative_error = torch.abs(output_norms - input_norms) / (
                input_norms + self.config.NORMALIZATION_EPS
            )
            return relative_error.mean().item()

    def train_single_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        epoch: int,
        lambda_scheduler: Optional[LambdaPressureScheduler] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_samples = 0
        correct = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.config.DEVICE)
            batch_y = batch_y.to(self.config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            mse_loss = self.criterion(outputs, batch_y)
            if lambda_scheduler is not None:
                reg_loss = lambda_scheduler.compute_regularization_loss(model)
                loss = mse_loss + reg_loss
            else:
                loss = mse_loss
            loss.backward()
            if epoch % self.config.NOISE_INTERVAL_EPOCHS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * self.config.NOISE_AMPLITUDE
                        param.grad.add_(noise)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.config.GRADIENT_CLIP_NORM
            )
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_mse += mse_loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            with torch.no_grad():
                per_sample_mse = ((outputs - batch_y)**2).mean(dim=(1, 2, 3))
                correct += (per_sample_mse < self.config.MSE_THRESHOLD).sum().item()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, train_acc

    def validate(
        self, model: nn.Module, val_x: torch.Tensor, val_y: torch.Tensor
    ) -> Tuple[float, float]:
        model.eval()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        with torch.no_grad():
            outputs = model(val_x)
            val_loss = self.criterion(outputs, val_y).item()
            per_sample_mse = ((outputs - val_y)**2).mean(dim=(1, 2, 3))
            val_acc = (per_sample_mse < self.config.MSE_THRESHOLD).float().mean().item()
        return val_loss, val_acc

    def collect_all_metrics(
        self,
        model: nn.Module,
        monitor: TrainingMetricsMonitor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        lambda_scheduler: Optional[LambdaPressureScheduler] = None,
        annealing_scheduler: Optional[AnnealingScheduler] = None,
        current_lr: float = 0.0
    ) -> Dict[str, Any]:
        lc, sp = self.compute_weight_metrics(model)
        crystal_metrics = self.crystal_calc.compute_all_metrics(model, val_x, val_y)
        alpha = crystal_metrics.get('alpha', 0.0)
        kappa = crystal_metrics.get('kappa', float('inf'))
        kappa_q = crystal_metrics.get('kappa_q', 1.0)
        delta = crystal_metrics.get('delta', 1.0)
        poynting = crystal_metrics.get('energy_flow', 0.0)
        purity_index = crystal_metrics.get('purity_index', 0.0)
        is_crystal = crystal_metrics.get('is_crystal', False)
        temp = self.thermo_calc.compute_effective_temperature(
            monitor.gradient_buffer, current_lr if current_lr > 0 else self.config.LEARNING_RATE
        )
        cv, is_transition = self.thermo_calc.compute_specific_heat(
            monitor.loss_history, monitor.temp_history
        )
        spectro = self.spectro_calc.compute(model)
        spectral_entropy = spectro.get('spectral_entropy', 0.0)
        num_peaks = spectro.get('num_peaks', 0)
        lambda_val = float(lambda_scheduler.current_lambda) if lambda_scheduler else 0.0
        hbar_eff = self.crystal_calc.compute_hbar_effective(model, lambda_val)
        annealing_temp = annealing_scheduler.temperature if annealing_scheduler else 0.0
        delta_slope = monitor.compute_delta_slope()
        norm_err = self.compute_norm_conservation_error(model, val_x)
        return {
            'lc': lc, 'sp': sp, 'alpha': alpha,
            'kappa': kappa, 'kappa_q': kappa_q, 'delta': delta,
            'temperature': temp, 'specific_heat': cv,
            'poynting_magnitude': poynting, 'energy_flow': poynting,
            'purity_index': purity_index, 'is_crystal': is_crystal,
            'lambda_pressure': lambda_val, 'hbar_effective': hbar_eff,
            'spectral_entropy': spectral_entropy,
            'num_bragg_peaks': num_peaks,
            'learning_rate': current_lr,
            'annealing_temp': annealing_temp,
            'delta_slope': delta_slope,
            'norm_conservation_error': norm_err
        }


class BatchSizeProspector:
    def __init__(self, config: Config, hamiltonian_engine: HamiltonianInferenceEngine):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.logger = LoggerFactory.create_logger("BatchSizeProspector")

    def prospect(self) -> int:
        self.logger.info("Phase 1: Batch size prospecting started")
        candidates = self.config.BATCH_CANDIDATES
        seed = self.config.BATCH_PROSPECT_SEED
        epochs = self.config.BATCH_PROSPECT_EPOCHS
        results = {}
        for bs in candidates:
            self.logger.info(f"  Testing batch_size={bs}")
            SeedManager.set_seed(seed, self.config.DEVICE)
            dataset = SchrodingerDataset(self.config, self.hamiltonian_engine, seed=seed)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            val_x, val_y = dataset.get_validation_batch()
            val_x = val_x.to(self.config.DEVICE)
            val_y = val_y.to(self.config.DEVICE)
            model = SchrodingerSpectralNetwork(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                expansion_dim=self.config.EXPANSION_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM
            )
            engine = TrainingEngine(self.config)
            crystal_calc = CrystallographyMetricsCalculator(self.config)
            final_delta = 1.0
            final_val_acc = 0.0
            final_kappa = float('inf')
            for ep in range(1, epochs + 1):
                engine.train_single_epoch(model, optimizer, loader, ep)
            val_loss, val_acc = engine.validate(model, val_x, val_y)
            delta = crystal_calc.compute_discretization_margin(model)
            kappa = crystal_calc.compute_kappa(model, val_x, val_y)
            results[bs] = {
                'delta': delta,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'kappa': kappa
            }
            self.logger.info(
                f"    batch_size={bs}: delta={delta:.6f}, "
                f"val_acc={val_acc:.4f}, kappa={kappa:.2e}"
            )
        best_bs = min(results.keys(), key=lambda k: results[k]['delta'])
        self.logger.info(
            f"Phase 1 complete: Best batch_size={best_bs} "
            f"(delta={results[best_bs]['delta']:.6f})"
        )
        return best_bs


class SeedMiner:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("SeedMiner")

    def mine(self) -> Optional[int]:
        self.logger.info("Phase 2: Seed mining started")
        prospect_epochs = self.config.MINING_PROSPECT_EPOCHS
        interval = self.config.MINING_PROSPECT_DELTA_EPOCH_INTERVAL
        max_attempts = self.config.MINING_MAX_ATTEMPTS
        start_seed = self.config.MINING_START_SEED
        seed_results = {}
        for i in range(start_seed, start_seed + max_attempts):
            current_seed = i
            SeedManager.set_seed(current_seed, self.config.DEVICE)
            dataset = SchrodingerDataset(
                self.config, self.hamiltonian_engine, seed=current_seed
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_x, val_y = dataset.get_validation_batch()
            val_x = val_x.to(self.config.DEVICE)
            val_y = val_y.to(self.config.DEVICE)
            model = SchrodingerSpectralNetwork(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                expansion_dim=self.config.EXPANSION_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM
            )
            engine = TrainingEngine(self.config)
            crystal_calc = CrystallographyMetricsCalculator(self.config)
            delta_trajectory = []
            for ep in range(1, prospect_epochs + 1):
                engine.train_single_epoch(model, optimizer, loader, ep)
                if ep % interval == 0:
                    delta = crystal_calc.compute_discretization_margin(model)
                    delta_trajectory.append((ep, delta))
            if len(delta_trajectory) >= 2:
                first_delta = delta_trajectory[0][1]
                last_delta = delta_trajectory[-1][1]
                delta_change = last_delta - first_delta
                is_decreasing = delta_change < 0
            else:
                first_delta = delta_trajectory[0][1] if delta_trajectory else 1.0
                last_delta = first_delta
                delta_change = 0.0
                is_decreasing = False
            seed_results[current_seed] = {
                'first_delta': first_delta,
                'last_delta': last_delta,
                'delta_change': delta_change,
                'is_decreasing': is_decreasing,
                'trajectory': delta_trajectory
            }
            status = "DECREASING" if is_decreasing else "ascending/flat"
            self.logger.info(
                f"  Seed {current_seed:>4} ({i - start_seed + 1}/{max_attempts}): "
                f"delta_0={first_delta:.6f} -> delta_f={last_delta:.6f} "
                f"change={delta_change:+.6f} [{status}]"
            )
        decreasing_seeds = {
            s: r for s, r in seed_results.items() if r['is_decreasing']
        }
        if not decreasing_seeds:
            self.logger.warning("No seeds with decreasing delta found, selecting lowest delta")
            best_seed = min(seed_results.keys(), key=lambda s: seed_results[s]['last_delta'])
        else:
            best_seed = min(
                decreasing_seeds.keys(),
                key=lambda s: decreasing_seeds[s]['last_delta']
            )
        result = seed_results[best_seed]
        self.logger.info(
            f"Phase 2 complete: Best seed={best_seed} "
            f"(delta={result['last_delta']:.6f}, "
            f"change={result['delta_change']:+.6f})"
        )
        self.logger.info(
            f"  Decreasing seeds found: {len(decreasing_seeds)}/{len(seed_results)} "
            f"({100*len(decreasing_seeds)/max(len(seed_results),1):.1f}%)"
        )
        return best_seed


class FullTrainingOrchestrator:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        seed: int,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.seed = seed
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("FullTrainingOrchestrator")

    def run_phase3_training(self) -> Tuple[nn.Module, optim.Optimizer, TrainingMetricsMonitor]:
        self.logger.info(
            f"Phase 3: Full training started (seed={self.seed}, "
            f"batch_size={self.batch_size}, epochs={self.config.EPOCHS})"
        )
        SeedManager.set_seed(self.seed, self.config.DEVICE)
        dataset = SchrodingerDataset(
            self.config, self.hamiltonian_engine, seed=self.seed
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_x, val_y = dataset.get_validation_batch()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        model = SchrodingerSpectralNetwork(
            grid_size=self.config.GRID_SIZE,
            hidden_dim=self.config.HIDDEN_DIM,
            expansion_dim=self.config.EXPANSION_DIM,
            num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
        ).to(self.config.DEVICE)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            momentum=self.config.MOMENTUM
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.EPOCHS,
            eta_min=self.config.LEARNING_RATE * self.config.COSINE_ANNEALING_ETA_MIN_FACTOR
        )
        lambda_scheduler = LambdaPressureScheduler(self.config)
        engine = TrainingEngine(self.config)
        monitor = TrainingMetricsMonitor(self.config)
        checkpoint_mgr = CheckpointManager(self.config, checkpoint_dir="checkpoints_phase3")
        grokking_detected = False
        grokking_epoch = None
        best_delta = float('inf')
        patience_counter = 0
        start_time = time.time()
        for epoch in range(1, self.config.EPOCHS + 1):
            lambda_scheduler.step(epoch)
            train_loss, train_acc = engine.train_single_epoch(
                model, optimizer, loader, epoch, lambda_scheduler
            )
            scheduler.step()
            val_loss, val_acc = engine.validate(model, val_x, val_y)
            current_lr = optimizer.param_groups[0]['lr']
            all_metrics = engine.collect_all_metrics(
                model, monitor, val_x, val_y,
                lambda_scheduler=lambda_scheduler,
                current_lr=current_lr
            )
            monitor.update_metrics(
                epoch=epoch, loss=train_loss, val_loss=val_loss,
                val_acc=val_acc, train_acc=train_acc, **all_metrics
            )
            for param in model.parameters():
                if param.grad is not None:
                    monitor.gradient_buffer.append(param.grad.detach().clone().flatten()[:500])
                    break
            delta = all_metrics['delta']
            if delta < best_delta:
                best_delta = delta
                patience_counter = 0
            else:
                patience_counter += 1
            if (train_acc >= self.config.GROKKING_TRAIN_ACC_THRESHOLD and
                val_acc >= self.config.GROKKING_VAL_ACC_THRESHOLD and
                not grokking_detected):
                delta_slope = monitor.compute_delta_slope()
                if delta_slope < self.config.GROKKING_DELTA_SLOPE_THRESHOLD:
                    grokking_detected = True
                    grokking_epoch = epoch
                    self.logger.info(
                        f"  GROKKING detected at epoch {epoch}: "
                        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                        f"delta={delta:.6f}, delta_slope={delta_slope:.2e}"
                    )
            if epoch % self.config.LOG_INTERVAL_EPOCHS == 0:
                bar = monitor.format_progress_bar(epoch, self.config.EPOCHS, "P3-TRAIN")
                self.logger.info(bar)
            if checkpoint_mgr.should_save_checkpoint():
                metrics_snapshot = {
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'train_acc': train_acc, **all_metrics
                }
                path = checkpoint_mgr.save_checkpoint(
                    model, optimizer, epoch, metrics_snapshot,
                    phase="phase3_training",
                    lambda_value=float(lambda_scheduler.current_lambda)
                )
                self.logger.info(f"  Checkpoint saved: {path}")
            integrity = WeightIntegrityChecker.check(model)
            if not integrity['is_valid']:
                self.logger.warning(
                    f"  Weight integrity compromised at epoch {epoch}: "
                    f"NaN={integrity['nan_count']}, Inf={integrity['inf_count']}"
                )
            if grokking_detected and patience_counter > self.config.GROKKING_PATIENCE:
                self.logger.info(
                    f"  Stopping Phase 3 at epoch {epoch}: "
                    f"grokking achieved and patience exceeded"
                )
                break
        elapsed = time.time() - start_time
        self.logger.info(
            f"Phase 3 complete in {elapsed:.1f}s. "
            f"Grokking={'YES at epoch ' + str(grokking_epoch) if grokking_detected else 'NO'}. "
            f"Best delta={best_delta:.6f}"
        )
        return model, optimizer, monitor


class RefinementOrchestrator:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        model: nn.Module,
        optimizer: optim.Optimizer,
        monitor: TrainingMetricsMonitor,
        seed: int,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.model = model
        self.optimizer = optimizer
        self.monitor = monitor
        self.seed = seed
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("RefinementOrchestrator")

    def run_phase4_refinement(self) -> nn.Module:
        self.logger.info(
            f"Phase 4: Refinement via simulated annealing "
            f"(epochs={self.config.REFINEMENT_EPOCHS})"
        )
        dataset = SchrodingerDataset(
            self.config, self.hamiltonian_engine, seed=self.seed
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_x, val_y = dataset.get_validation_batch()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        lambda_scheduler = LambdaPressureScheduler(self.config)
        lambda_scheduler.set_lambda(self.config.LAMBDA_MAX * 0.1)
        annealing = AnnealingScheduler(self.config)
        engine = TrainingEngine(self.config)
        checkpoint_mgr = CheckpointManager(
            self.config, checkpoint_dir="checkpoints_phase4"
        )
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_delta = CrystallographyMetricsCalculator(self.config).compute_discretization_margin(
            self.model
        )
        self.logger.info(f"  Starting refinement from delta={best_delta:.6f}")
        refinement_lr = self.config.LEARNING_RATE * 0.1
        for pg in self.optimizer.param_groups:
            pg['lr'] = refinement_lr
        start_time = time.time()
        for epoch in range(1, self.config.REFINEMENT_EPOCHS + 1):
            lambda_scheduler.step(epoch)
            annealing.step()
            previous_state = copy.deepcopy(self.model.state_dict())
            train_loss, train_acc = engine.train_single_epoch(
                self.model, self.optimizer, loader, epoch, lambda_scheduler
            )
            val_loss, val_acc = engine.validate(self.model, val_x, val_y)
            current_delta = CrystallographyMetricsCalculator(
                self.config
            ).compute_discretization_margin(self.model)
            delta_diff = current_delta - best_delta
            if not annealing.accept_perturbation(delta_diff):
                self.model.load_state_dict(previous_state)
                current_delta = best_delta
            else:
                if current_delta < best_delta:
                    best_delta = current_delta
                    best_model_state = copy.deepcopy(self.model.state_dict())
            if annealing.should_restart(current_delta, best_delta):
                self.model.load_state_dict(best_model_state)
                self.logger.info(
                    f"  Annealing restart at epoch {epoch}, "
                    f"reverting to best_delta={best_delta:.6f}"
                )
            current_lr = self.optimizer.param_groups[0]['lr']
            all_metrics = engine.collect_all_metrics(
                self.model, self.monitor, val_x, val_y,
                lambda_scheduler=lambda_scheduler,
                annealing_scheduler=annealing,
                current_lr=current_lr
            )
            self.monitor.update_metrics(
                epoch=epoch, loss=train_loss, val_loss=val_loss,
                val_acc=val_acc, train_acc=train_acc, **all_metrics
            )
            if epoch % self.config.LOG_INTERVAL_EPOCHS == 0:
                bar = self.monitor.format_progress_bar(
                    epoch, self.config.REFINEMENT_EPOCHS, "P4-REFINE"
                )
                self.logger.info(bar)
                self.logger.info(
                    f"  Annealing T={annealing.temperature:.2e}, "
                    f"best_delta={best_delta:.6f}, "
                    f"current_delta={current_delta:.6f}"
                )
            if checkpoint_mgr.should_save_checkpoint():
                metrics_snapshot = {
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'train_acc': train_acc,
                    'best_delta': best_delta,
                    'annealing_temperature': annealing.temperature,
                    **all_metrics
                }
                path = checkpoint_mgr.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics_snapshot,
                    phase="phase4_refinement",
                    lambda_value=float(lambda_scheduler.current_lambda)
                )
                self.logger.info(f"  Checkpoint saved: {path}")
        self.model.load_state_dict(best_model_state)
        elapsed = time.time() - start_time
        self.logger.info(
            f"Phase 4 complete in {elapsed:.1f}s. "
            f"Final best_delta={best_delta:.6f}"
        )
        return self.model


class ExperimentOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("ExperimentOrchestrator")

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("SCHRODINGER EQUATION HAMILTONIAN TOPOLOGICAL CRYSTALLIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Device: {self.config.DEVICE}")
        self.logger.info(f"Grid size: {self.config.GRID_SIZE}")
        self.logger.info(f"Hidden dim: {self.config.HIDDEN_DIM}")
        self.logger.info(f"Expansion dim: {self.config.EXPANSION_DIM}")
        self.logger.info(f"Spectral layers: {self.config.NUM_SPECTRAL_LAYERS}")
        self.logger.info(f"Lambda max: {self.config.LAMBDA_MAX:.2e}")
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        hamiltonian_engine = HamiltonianInferenceEngine(self.config)
        self.logger.info("-" * 80)
        self.logger.info("PHASE 1: BATCH SIZE PROSPECTING")
        self.logger.info("-" * 80)
        prospector = BatchSizeProspector(self.config, hamiltonian_engine)
        best_batch_size = prospector.prospect()
        self.logger.info("-" * 80)
        self.logger.info("PHASE 2: SEED MINING WITH DECREASING DELTA")
        self.logger.info("-" * 80)
        miner = SeedMiner(self.config, hamiltonian_engine, best_batch_size)
        best_seed = miner.mine()
        if best_seed is None:
            self.logger.warning("No suitable seed found, using default")
            best_seed = self.config.RANDOM_SEED
        self.logger.info("-" * 80)
        self.logger.info("PHASE 3: FULL TRAINING TO GROKKING")
        self.logger.info("-" * 80)
        trainer = FullTrainingOrchestrator(
            self.config, hamiltonian_engine, best_seed, best_batch_size
        )
        model, optimizer, monitor = trainer.run_phase3_training()
        self.logger.info("-" * 80)
        self.logger.info("PHASE 4: REFINEMENT VIA SIMULATED ANNEALING")
        self.logger.info("-" * 80)
        refiner = RefinementOrchestrator(
            self.config, hamiltonian_engine, model, optimizer,
            monitor, best_seed, best_batch_size
        )
        final_model = refiner.run_phase4_refinement()
        self.logger.info("-" * 80)
        self.logger.info("FINAL RESULTS")
        self.logger.info("-" * 80)
        self._save_final_results(final_model, monitor, best_seed, best_batch_size)

    def _save_final_results(
        self,
        model: nn.Module,
        monitor: TrainingMetricsMonitor,
        seed: int,
        batch_size: int
    ):
        os.makedirs("weights", exist_ok=True)
        crystal_calc = CrystallographyMetricsCalculator(self.config)
        delta = crystal_calc.compute_discretization_margin(model)
        alpha = crystal_calc.compute_alpha_purity(model)
        kappa_q = crystal_calc.compute_kappa_quantum(model)
        poynting = crystal_calc.compute_poynting_vector(model)
        integrity = WeightIntegrityChecker.check(model)
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': asdict(self.config),
            'seed': seed,
            'batch_size': batch_size,
            'final_metrics': {
                'delta': delta,
                'alpha': alpha,
                'kappa_q': kappa_q,
                'poynting_magnitude': poynting['poynting_magnitude'],
                'integrity': integrity
            },
            'metrics_history': monitor.metrics_history,
            'timestamp': datetime.now().isoformat()
        }
        final_path = "weights/schrodinger_crystal_final.pth"
        torch.save(final_checkpoint, final_path)
        self.logger.info(f"Final model saved to {final_path}")
        results_summary = {
            'seed': seed,
            'batch_size': batch_size,
            'delta': delta,
            'alpha': alpha,
            'kappa_q': kappa_q,
            'poynting_magnitude': poynting['poynting_magnitude'],
            'is_valid': integrity['is_valid'],
            'total_params': integrity['total_params'],
            'timestamp': datetime.now().isoformat()
        }
        results_path = os.path.join(self.config.RESULTS_DIR, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        self.logger.info(f"Results summary saved to {results_path}")
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info(f"  Seed: {seed}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Delta: {delta:.6f}")
        self.logger.info(f"  Alpha: {alpha:.4f}")
        self.logger.info(f"  Kappa_q: {kappa_q:.2e}")
        self.logger.info(f"  Poynting: {poynting['poynting_magnitude']:.2e}")
        self.logger.info(f"  Valid: {integrity['is_valid']}")
        self.logger.info(f"  Total params: {integrity['total_params']}")
        self.logger.info("=" * 80)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Schrodinger Equation Hamiltonian Topological Crystallization'
    )
    parser.add_argument(
        '--grid_size', type=int, default=Config.GRID_SIZE,
        help='Grid size for spatial discretization'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=Config.HIDDEN_DIM,
        help='Hidden dimension of spectral network'
    )
    parser.add_argument(
        '--expansion_dim', type=int, default=Config.EXPANSION_DIM,
        help='Expansion dimension for spectral processing'
    )
    parser.add_argument(
        '--num_spectral_layers', type=int, default=Config.NUM_SPECTRAL_LAYERS,
        help='Number of spectral layers'
    )
    parser.add_argument(
        '--epochs', type=int, default=Config.EPOCHS,
        help='Number of training epochs for Phase 3'
    )
    parser.add_argument(
        '--refinement_epochs', type=int, default=Config.REFINEMENT_EPOCHS,
        help='Number of refinement epochs for Phase 4'
    )
    parser.add_argument(
        '--lr', type=float, default=Config.LEARNING_RATE,
        help='Base learning rate'
    )
    parser.add_argument(
        '--lambda_max', type=float, default=Config.LAMBDA_MAX,
        help='Maximum lambda regularization pressure'
    )
    parser.add_argument(
        '--mining_attempts', type=int, default=Config.MINING_MAX_ATTEMPTS,
        help='Number of seed mining attempts'
    )
    parser.add_argument(
        '--backbone_path', type=str, default=Config.BACKBONE_CHECKPOINT_PATH,
        help='Path to Hamiltonian backbone checkpoint'
    )
    parser.add_argument(
        '--no_backbone', action='store_true',
        help='Disable Hamiltonian backbone, use analytical operator'
    )
    parser.add_argument(
        '--num_samples', type=int, default=Config.NUM_SAMPLES,
        help='Number of training samples'
    )
    parser.add_argument(
        '--potential_depth', type=float, default=Config.POTENTIAL_DEPTH,
        help='Depth of potential wells for Schrodinger equation'
    )
    parser.add_argument(
        '--num_eigenstates', type=int, default=Config.NUM_EIGENSTATES,
        help='Number of eigenstates to sample from'
    )
    parser.add_argument(
        '--checkpoint_interval', type=int, default=Config.CHECKPOINT_INTERVAL_MINUTES,
        help='Checkpoint interval in minutes'
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    config = Config()
    config.GRID_SIZE = args.grid_size
    config.HIDDEN_DIM = args.hidden_dim
    config.EXPANSION_DIM = args.expansion_dim
    config.NUM_SPECTRAL_LAYERS = args.num_spectral_layers
    config.EPOCHS = args.epochs
    config.REFINEMENT_EPOCHS = args.refinement_epochs
    config.LEARNING_RATE = args.lr
    config.LAMBDA_MAX = args.lambda_max
    config.MINING_MAX_ATTEMPTS = args.mining_attempts
    config.BACKBONE_CHECKPOINT_PATH = args.backbone_path
    config.BACKBONE_ENABLED = not args.no_backbone
    config.NUM_SAMPLES = args.num_samples
    config.POTENTIAL_DEPTH = args.potential_depth
    config.NUM_EIGENSTATES = args.num_eigenstates
    config.CHECKPOINT_INTERVAL_MINUTES = args.checkpoint_interval
    orchestrator = ExperimentOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()