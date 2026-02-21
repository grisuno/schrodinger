#!/usr/bin/env python3
"""
schrodinger_definitive_crystallographer.py

Author: Gris Iscomeback
Email: grisiscomeback@gmail.com
Date: 2024
License: AGPL v3

Description:
Definitive crystallographic analysis system for Schrodinger equation neural network
checkpoints. Combines thermodynamic, spectral, geometric, and structural metrics
from multiple analysis paradigms into a unified production-ready framework.

Implements:
- Weight Integrity and Discretization Analysis
- Thermodynamic Potentials (Gibbs, Helmholtz Free Energy)
- Spectral Geometry with MBL Level Spacing
- Ricci Curvature Estimation
- Weight Spectroscopy with Advanced Bragg Peak Detection
- Complex Kernel Holomorphy Analysis
- Hamiltonian Inference Engine Integration
- Phase Classification and Crystallographic Grading
- Batch Analysis with Visualization Generation
"""
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import math
import copy
import warnings
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque, Counter
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from scipy import stats, signal, linalg
from scipy.stats import entropy as scipy_entropy
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


@dataclass
class SchrodingerCrystallographyConfig:
    """
    Comprehensive configuration for Schrodinger crystallographic analysis.
    All parameters are centralized here following the Single Responsibility Principle.
    No magic numbers or hardcoded values appear elsewhere in the codebase.
    """
    
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
    
    EIGENVALUE_TOLERANCE: float = 1e-10
    NORMALIZATION_EPSILON: float = 1e-8
    ENTROPY_EPSILON: float = 1e-10
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    
    ALPHA_CRYSTAL_THRESHOLD: float = 7.0
    ALPHA_PERFECT_CRYSTAL_THRESHOLD: float = 10.0
    DELTA_OPTICAL_THRESHOLD: float = 0.01
    DELTA_INDUSTRIAL_THRESHOLD: float = 0.1
    DELTA_POLYCRYSTALLINE_THRESHOLD: float = 0.3
    DELTA_AMORPHOUS_THRESHOLD: float = 0.5
    DELTA_CRYSTAL_THRESHOLD: float = 0.1
    DELTA_GLASS_THRESHOLD: float = 0.4
    
    KAPPA_CRYSTAL_THRESHOLD: float = 1.5
    KAPPA_WELL_CONDITIONED_THRESHOLD: float = 10.0
    KAPPA_MAX_DIMENSION: int = 10000
    KAPPA_GRADIENT_BATCHES: int = 5
    
    TEMPERATURE_CRYSTAL_THRESHOLD: float = 1e-9
    THERMAL_NOISE_SCALE: float = 0.01
    
    ENTROPY_BINS: int = 50
    HISTOGRAM_BINS: int = 100
    PCA_COMPONENTS: int = 3
    
    SPECTRAL_PEAK_THRESHOLD_SIGMA: float = 2.0
    SPECTRAL_PEAK_LIMIT: int = 20
    SPECTRAL_POWER_LIMIT: int = 100
    
    PARAM_FLATTEN_LIMIT: int = 2000
    GRADIENT_BUFFER_LIMIT: int = 500
    GRADIENT_BUFFER_WINDOW: int = 10
    GRADIENT_BUFFER_MAXLEN: int = 50
    LOSS_HISTORY_MAXLEN: int = 100
    TEMP_HISTORY_MAXLEN: int = 100
    CV_HISTORY_MAXLEN: int = 100
    WEIGHT_METRIC_DIM_LIMIT: int = 256
    
    CV_THRESHOLD: float = 1.0
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    POYNTING_OUTER_LIMIT: int = 50
    
    HBAR: float = 1e-6
    HBAR_PHYSICAL: float = 1.0545718e-34
    
    DISCRETIZATION_MARGIN: float = 0.1
    TARGET_SLOTS: int = 7
    
    LEARNING_RATE: float = 0.005
    BATCH_SIZE: int = 32
    
    NUM_SAMPLES: int = 32
    TRAIN_RATIO: float = 0.7
    DT: float = 0.01
    TIME_STEPS: int = 2
    
    BACKBONE_CHECKPOINT_PATH: str = 'weights/latest.pth'
    BACKBONE_ENABLED: bool = True
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = 'INFO'
    RESULTS_DIR: str = 'schrodinger_crystallography_reports'
    VISUALIZATION_DPI: int = 300
    
    HOLMORPHY_THRESHOLD: float = 0.1
    RICCI_CURVATURE_SAMPLES: int = 100
    RICCI_MAX_DIMENSION: int = 5000


class LoggerFactory:
    """
    Factory for creating configured logger instances.
    Follows the Single Responsibility Principle for logging configuration.
    """
    
    @staticmethod
    def create_logger(name: str, level: str = None) -> logging.Logger:
        """
        Create and configure a logger with standardized formatting.
        
        Args:
            name: Identifier for the logger instance.
            level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        
        Returns:
            Configured Logger instance ready for use.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, (level or SchrodingerCrystallographyConfig.LOG_LEVEL).upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class IMetricCalculator(ABC):
    """
    Interface for metric calculation strategies.
    Follows the Interface Segregation Principle by defining a minimal contract.
    """
    
    @abstractmethod
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute metrics for the given model.
        
        Args:
            model: Neural network model to analyze.
            **kwargs: Additional parameters required for computation.
        
        Returns:
            Dictionary containing computed metric values.
        """
        pass


class HamiltonianOperator:
    """
    Analytical Hamiltonian operator for fallback computation.
    Implements spectral operators for Laplacian computation in Fourier space.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Hamiltonian operator with precomputed spectral operators.
        
        Args:
            config: Configuration containing grid size and other parameters.
        """
        self.config = config
        self.grid_size = config.GRID_SIZE
        self._precompute_spectral_operators()
    
    def _precompute_spectral_operators(self) -> None:
        """
        Precompute the Laplacian spectrum in Fourier space for efficient application.
        """
        kx = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.laplacian_spectrum = -(KX**2 + KY**2).float()
    
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply the Hamiltonian operator to a field using spectral methods.
        
        Args:
            field: Input field tensor (2D or batched).
        
        Returns:
            Transformed field after Hamiltonian application.
        """
        field_fft = torch.fft.fft2(field)
        laplacian_fft = field_fft * self.laplacian_spectrum.to(field.device)
        return torch.fft.ifft2(laplacian_fft).real
    
    def time_evolution(self, field: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        Perform time evolution of the field under the Hamiltonian.
        
        Args:
            field: Input field tensor.
            dt: Time step size.
        
        Returns:
            Time-evolved field with preserved norm.
        """
        dt = dt or self.config.DT
        hamiltonian_action = self.apply(field)
        evolved = field + hamiltonian_action * dt
        norm_original = torch.norm(field) + self.config.NORMALIZATION_EPSILON
        norm_evolved = torch.norm(evolved) + self.config.NORMALIZATION_EPSILON
        return evolved / norm_evolved * norm_original


class SpectralLayer(nn.Module):
    """
    Spectral convolution layer operating in Fourier space with complex kernels.
    Implements learnable frequency-domain transformations.
    """
    
    def __init__(self, channels: int, grid_size: int, config: SchrodingerCrystallographyConfig = None):
        """
        Initialize spectral layer with complex-valued kernels.
        
        Args:
            channels: Number of input/output channels.
            grid_size: Spatial dimension of the input grid.
            config: Configuration object (optional for parameter access).
        """
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
        """
        Apply spectral convolution in Fourier domain.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
        
        Returns:
            Transformed tensor after spectral convolution.
        """
        x_fft = torch.fft.rfft2(x)
        batch, channels, freq_h, freq_w = x_fft.shape
        kernel_real = self.kernel_real.mean(dim=0)
        kernel_imag = self.kernel_imag.mean(dim=0)
        kernel_real_exp = kernel_real.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_imag_exp = kernel_imag.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_real_interp = F.interpolate(
            kernel_real_exp, size=(freq_h, freq_w), mode='bilinear', align_corners=False
        )
        kernel_imag_interp = F.interpolate(
            kernel_imag_exp, size=(freq_h, freq_w), mode='bilinear', align_corners=False
        )
        real_part = x_fft.real * kernel_real_interp - x_fft.imag * kernel_imag_interp
        imag_part = x_fft.real * kernel_imag_interp + x_fft.imag * kernel_real_interp
        output_fft = torch.complex(real_part, imag_part)
        output = torch.fft.irfft2(output_fft, s=(self.grid_size, self.grid_size))
        return output


class HamiltonianBackbone(nn.Module):
    """
    Neural network backbone for Hamiltonian inference.
    Learns to approximate Hamiltonian operations from data.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Hamiltonian backbone network.
        
        Args:
            config: Configuration containing architecture parameters.
        """
        super().__init__()
        self.grid_size = config.GRID_SIZE
        self.input_proj = nn.Conv2d(1, config.HIDDEN_DIM, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(config.HIDDEN_DIM, config.GRID_SIZE, config)
            for _ in range(config.NUM_SPECTRAL_LAYERS)
        ])
        self.output_proj = nn.Conv2d(config.HIDDEN_DIM, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Hamiltonian backbone.
        
        Args:
            x: Input tensor of shape (batch, height, width) or (batch, 1, height, width).
        
        Returns:
            Hamiltonian-transformed output tensor.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        return self.output_proj(x).squeeze(1)


class SchrodingerSpectralNetwork(nn.Module):
    """
    Schrodinger equation neural network with spectral layers.
    Implements expansion-contraction architecture with Fourier convolutions.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Schrodinger spectral network.
        
        Args:
            config: Configuration containing all architecture parameters.
        """
        super().__init__()
        self.grid_size = config.GRID_SIZE
        self.input_channels = config.SCHRODINGER_CHANNELS
        self.output_channels = config.SCHRODINGER_CHANNELS
        self.input_proj = nn.Conv2d(config.SCHRODINGER_CHANNELS, config.HIDDEN_DIM, kernel_size=1)
        self.expansion_proj = nn.Conv2d(config.HIDDEN_DIM, config.EXPANSION_DIM, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(config.EXPANSION_DIM, config.GRID_SIZE, config)
            for _ in range(config.NUM_SPECTRAL_LAYERS)
        ])
        self.contraction_proj = nn.Conv2d(config.EXPANSION_DIM, config.HIDDEN_DIM, kernel_size=1)
        self.output_proj = nn.Conv2d(config.HIDDEN_DIM, config.SCHRODINGER_CHANNELS, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Schrodinger network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
        
        Returns:
            Output tensor after expansion, spectral processing, and contraction.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class HamiltonianInferenceEngine:
    """
    Engine for Hamiltonian inference using either a pretrained backbone
    or analytical operators as fallback.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Hamiltonian inference engine.
        
        Args:
            config: Configuration containing backbone path and device settings.
        """
        self.config = config
        self.logger = LoggerFactory.create_logger("HamiltonianInferenceEngine")
        self.backbone = None
        self.fallback_operator = HamiltonianOperator(config)
        self._try_load_backbone()
    
    def _try_load_backbone(self) -> None:
        """
        Attempt to load a pretrained backbone for Hamiltonian inference.
        Falls back to analytical operator if backbone is unavailable.
        """
        if not self.config.BACKBONE_ENABLED:
            self.logger.info("Backbone disabled, using analytical Hamiltonian operator")
            return
        checkpoint_path = self.config.BACKBONE_CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            self.logger.info(
                f"Backbone checkpoint not found at {checkpoint_path}, using analytical operator"
            )
            return
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.config.DEVICE, weights_only=False
            )
            self.backbone = HamiltonianBackbone(self.config).to(self.config.DEVICE)
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
        """
        Apply the Hamiltonian to a field using backbone or analytical operator.
        
        Args:
            field: Input field tensor.
        
        Returns:
            Hamiltonian-transformed field.
        """
        if self.backbone is not None:
            with torch.no_grad():
                return self.backbone(field.to(self.config.DEVICE))
        return self.fallback_operator.apply(field)
    
    def time_evolve(self, field: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        Perform time evolution using backbone or analytical operator.
        
        Args:
            field: Input field tensor.
            dt: Time step size.
        
        Returns:
            Time-evolved field with preserved norm.
        """
        dt = dt or self.config.DT
        if self.backbone is not None:
            with torch.no_grad():
                evolved = self.backbone(field.to(self.config.DEVICE))
                norm_original = torch.norm(field) + self.config.NORMALIZATION_EPSILON
                norm_evolved = torch.norm(evolved) + self.config.NORMALIZATION_EPSILON
                return evolved / norm_evolved * norm_original
        return self.fallback_operator.time_evolution(field, dt)


class SchrodingerPotentialGenerator:
    """
    Generator for various potential energy landscapes used in Schrodinger equation.
    Supports harmonic, double-well, Coulomb-like, and periodic lattice potentials.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the potential generator.
        
        Args:
            config: Configuration containing potential parameters.
        """
        self.config = config
        self.grid_size = config.GRID_SIZE
    
    def harmonic_potential(self) -> torch.Tensor:
        """
        Generate a harmonic oscillator potential.
        
        Returns:
            2D tensor with harmonic potential centered at grid center.
        """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx = np.pi
        cy = np.pi
        return 0.5 * self.config.POTENTIAL_DEPTH * ((X - cx)**2 + (Y - cy)**2) / (np.pi**2)
    
    def double_well_potential(self) -> torch.Tensor:
        """
        Generate a double-well potential.
        
        Returns:
            2D tensor with double-well potential along x-axis.
        """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx = np.pi
        w = self.config.POTENTIAL_WIDTH * np.pi
        return self.config.POTENTIAL_DEPTH * ((X - cx)**2 / w**2 - 1)**2
    
    def coulomb_like_potential(self) -> torch.Tensor:
        """
        Generate a Coulomb-like central potential.
        
        Returns:
            2D tensor with Coulomb potential centered at grid center.
        """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx, cy = np.pi, np.pi
        r = torch.sqrt((X - cx)**2 + (Y - cy)**2) + self.config.POTENTIAL_WIDTH
        return -self.config.POTENTIAL_DEPTH / r
    
    def periodic_lattice_potential(self) -> torch.Tensor:
        """
        Generate a periodic lattice potential.
        
        Returns:
            2D tensor with periodic cosine potential.
        """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return self.config.POTENTIAL_DEPTH * (torch.cos(2 * X) + torch.cos(2 * Y))
    
    def generate_mixed_potential(self, seed: int) -> torch.Tensor:
        """
        Generate a mixed potential combining multiple potential types.
        
        Args:
            seed: Random seed for determining mixture weights.
        
        Returns:
            2D tensor with weighted combination of potential types.
        """
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


class SyntheticDataGenerator:
    """
    Generator for synthetic Schrodinger equation training data.
    Creates initial and target wavefunction pairs for various potentials.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig, 
                 hamiltonian_engine: HamiltonianInferenceEngine):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Configuration containing data generation parameters.
            hamiltonian_engine: Engine for Hamiltonian operations.
        """
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.potential_generator = SchrodingerPotentialGenerator(config)
    
    def generate_batch(self, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of validation data.
        
        Args:
            seed: Random seed for reproducibility.
        
        Returns:
            Tuple of (initial_states, target_states) tensors.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        num_samples = self.config.NUM_SAMPLES
        initial_states = []
        target_states = []
        for i in range(num_samples):
            potential = self.potential_generator.generate_mixed_potential(seed + i)
            psi_real, psi_imag, _ = self._solve_schrodinger_sample(potential, seed + i)
            evolved_real, evolved_imag = self._time_evolve_wavefunction(
                psi_real, psi_imag, potential
            )
            initial = torch.stack([psi_real, psi_imag], dim=0)
            target = torch.stack([evolved_real, evolved_imag], dim=0)
            initial_states.append(initial)
            target_states.append(target)
        x = torch.stack(initial_states).to(self.config.DEVICE)
        y = torch.stack(target_states).to(self.config.DEVICE)
        split = int(num_samples * self.config.TRAIN_RATIO)
        return x[split:], y[split:]
    
    def _solve_schrodinger_sample(
        self, potential: torch.Tensor, sample_seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Solve the Schrodinger equation for a single sample.
        
        Args:
            potential: Potential energy landscape.
            sample_seed: Random seed for eigenstate selection.
        
        Returns:
            Tuple of (real_part, imaginary_part, energy) for the wavefunction.
        """
        h_field = self.hamiltonian_engine.apply_hamiltonian(
            torch.randn(self.config.GRID_SIZE, self.config.GRID_SIZE)
        )
        if h_field.dim() > 2:
            h_field = h_field.squeeze()
        kinetic = -0.5 * h_field
        hamiltonian_matrix = kinetic + potential
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian_matrix)
        except Exception:
            eigenvalues = torch.zeros(self.config.GRID_SIZE)
            eigenvectors = torch.eye(self.config.GRID_SIZE)
        rng = np.random.RandomState(sample_seed)
        n_states = min(self.config.NUM_EIGENSTATES, self.config.GRID_SIZE)
        state_idx = rng.randint(0, n_states)
        energy = eigenvalues[state_idx].item() * self.config.ENERGY_SCALE
        psi_column = eigenvectors[:, state_idx]
        psi_2d = psi_column.unsqueeze(1).expand(-1, self.config.GRID_SIZE)
        perturbation = torch.randn_like(psi_2d) * 0.1
        psi_2d = psi_2d + perturbation
        norm = torch.sqrt(torch.sum(psi_2d**2)) + self.config.NORMALIZATION_EPSILON
        psi_2d = psi_2d / norm * self.config.WAVEFUNCTION_NORM_TARGET
        phase = torch.randn(self.config.GRID_SIZE, self.config.GRID_SIZE) * 0.5
        psi_real = psi_2d * torch.cos(phase)
        psi_imag = psi_2d * torch.sin(phase)
        return psi_real, psi_imag, energy
    
    def _time_evolve_wavefunction(
        self, psi_real: torch.Tensor, psi_imag: torch.Tensor, potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Time evolve a wavefunction under the given potential.
        
        Args:
            psi_real: Real part of the wavefunction.
            psi_imag: Imaginary part of the wavefunction.
            potential: Potential energy landscape.
        
        Returns:
            Tuple of (evolved_real, evolved_imag) wavefunction components.
        """
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
            norm = torch.sqrt(torch.sum(new_real**2 + new_imag**2)) + self.config.NORMALIZATION_EPSILON
            target_norm = torch.sqrt(torch.sum(psi_real**2 + psi_imag**2)) + self.config.NORMALIZATION_EPSILON
            new_real = new_real / norm * target_norm
            new_imag = new_imag / norm * target_norm
            psi_real, psi_imag = new_real, new_imag
        return psi_real, psi_imag


class WeightIntegrityCalculator(IMetricCalculator):
    """
    Calculator for weight integrity metrics including NaN and Inf detection.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the weight integrity calculator.
        
        Args:
            config: Configuration containing tolerance parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute weight integrity metrics for the model.
        
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
                - corruption_ratio: Ratio of corrupted to total parameters
        """
        has_nan = False
        has_inf = False
        total_params = 0
        nan_count = 0
        inf_count = 0
        
        for param in model.parameters():
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


class DiscretizationCalculator(IMetricCalculator):
    """
    Calculator for discretization margin and alpha purity metrics.
    These metrics quantify how close weights are to integer values.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the discretization calculator.
        
        Args:
            config: Configuration containing discretization thresholds.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute discretization metrics for the model.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - delta: Maximum discretization margin
                - alpha: Purity index (-log(delta))
                - spectral_entropy: Entropy of weight power spectrum
                - is_discrete: Boolean indicating discretization below threshold
                - layer_deltas: Per-layer discretization margins
        """
        margins = []
        all_params = []
        layer_deltas = {}
        
        for name, param in model.named_parameters():
            if param.numel() > 0:
                p_data = param.data.detach()
                all_params.append(p_data.flatten())
                margin = (p_data - p_data.round()).abs().max().item()
                margins.append(margin)
                layer_deltas[name] = margin
        
        delta = max(margins) if margins else 0.0
        alpha = -np.log(delta + self.config.ENTROPY_EPSILON) if delta > 0 else 20.0
        
        flat_params = torch.cat(all_params)[:self.config.PARAM_FLATTEN_LIMIT]
        spectral_entropy = self._compute_spectral_entropy(flat_params)
        
        return {
            'delta': delta,
            'alpha': alpha,
            'spectral_entropy': spectral_entropy,
            'is_discrete': delta < self.config.DELTA_CRYSTAL_THRESHOLD,
            'layer_deltas': layer_deltas
        }
    
    def _compute_spectral_entropy(self, weights: torch.Tensor) -> float:
        """
        Compute the spectral entropy of the weight distribution.
        
        Args:
            weights: Flattened weight tensor.
        
        Returns:
            Spectral entropy value.
        """
        if weights.numel() == 0:
            return 0.0
        w = weights.detach().cpu()
        fft_spectrum = torch.fft.fft(w)
        power_spectrum = torch.abs(fft_spectrum)**2
        
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy.item())


class LocalComplexityCalculator(IMetricCalculator):
    """
    Calculator for local complexity metrics measuring weight diversity.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the local complexity calculator.
        
        Args:
            config: Configuration containing dimension limits.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute local complexity for the model weights.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing local_complexity value.
        """
        lc_values = []
        limit = self.config.WEIGHT_METRIC_DIM_LIMIT
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param.data.detach()
                w = w[:min(w.size(0), limit), :min(w.size(1), limit)]
                lc = self._compute_local_complexity(w)
                lc_values.append(lc)
        avg_lc = np.mean(lc_values) if lc_values else 0.0
        return {'local_complexity': avg_lc}
    
    def _compute_local_complexity(self, weights: torch.Tensor) -> float:
        """
        Compute local complexity for a weight matrix.
        
        Args:
            weights: 2D weight tensor.
        
        Returns:
            Local complexity value between 0 and 1.
        """
        if weights.numel() == 0:
            return 0.0
        w = weights.flatten()
        w = w / (torch.norm(w) + self.config.MIN_VARIANCE_THRESHOLD)
        w_expanded = w.unsqueeze(0)
        similarities = F.cosine_similarity(w_expanded, w_expanded.unsqueeze(1), dim=2)
        mask = ~torch.eye(similarities.size(0), device=similarities.device, dtype=torch.bool)
        if mask.sum() == 0:
            return 0.0
        avg_similarity = (similarities.abs() * mask).sum() / mask.sum()
        lc = 1.0 - avg_similarity.item()
        return max(0.0, min(1.0, lc))


class SuperpositionCalculator(IMetricCalculator):
    """
    Calculator for superposition metrics measuring weight correlations.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the superposition calculator.
        
        Args:
            config: Configuration containing computation parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute superposition metrics for the model.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing superposition value.
        """
        sp_values = []
        limit = self.config.WEIGHT_METRIC_DIM_LIMIT
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param.data.detach()
                w = w[:min(w.size(0), limit), :min(w.size(1), limit)]
                sp = self._compute_superposition(w)
                sp_values.append(sp)
        avg_sp = np.mean(sp_values) if sp_values else 0.0
        return {'superposition': avg_sp}
    
    def _compute_superposition(self, weights: torch.Tensor) -> float:
        """
        Compute superposition metric for a weight matrix.
        
        Args:
            weights: 2D weight tensor.
        
        Returns:
            Superposition value indicating average correlation.
        """
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


class GradientDynamicsCalculator(IMetricCalculator):
    """
    Calculator for gradient-based metrics including condition number and effective temperature.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the gradient dynamics calculator.
        
        Args:
            config: Configuration containing gradient computation parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute gradient dynamics metrics.
        
        Args:
            model: Neural network model to analyze.
            **kwargs: Must contain 'val_x' and 'val_y' tensors.
        
        Returns:
            Dictionary containing:
                - kappa: Condition number of gradient covariance
                - effective_temperature: Temperature derived from gradient variance
                - gradient_variance: Variance of gradient samples
        """
        val_x = kwargs.get('val_x')
        val_y = kwargs.get('val_y')
        
        if val_x is None or val_y is None:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }
        
        model.train()
        grads = []
        
        for i in range(self.config.KAPPA_GRADIENT_BATCHES):
            try:
                model.zero_grad()
                noise_scale = self.config.THERMAL_NOISE_SCALE * (i + 1) / self.config.KAPPA_GRADIENT_BATCHES
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
                        grads.append(grad_vector.detach().clone())
            except Exception:
                continue
        
        model.eval()
        
        if len(grads) < 2:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }
        
        grads_tensor = torch.stack(grads)
        n_samples, n_dims = grads_tensor.shape
        
        if n_dims > self.config.KAPPA_MAX_DIMENSION:
            indices = torch.randperm(n_dims, device=grads_tensor.device)[:self.config.KAPPA_MAX_DIMENSION]
            grads_tensor = grads_tensor[:, indices]
            n_dims = self.config.KAPPA_MAX_DIMENSION
        
        try:
            if n_samples < n_dims:
                gram = torch.mm(grads_tensor, grads_tensor.t()) / max(n_samples - 1, 1)
                eigenvals = torch.linalg.eigvalsh(gram)
            else:
                cov = torch.cov(grads_tensor.t())
                eigenvals = torch.linalg.eigvalsh(cov).real
            
            eigenvals = eigenvals[eigenvals > self.config.EIGENVALUE_TOLERANCE]
            
            if len(eigenvals) == 0:
                return {
                    'kappa': float('inf'),
                    'effective_temperature': 0.0,
                    'gradient_variance': 0.0
                }
            
            kappa = (eigenvals.max() / eigenvals.min()).item()
            
            second_moment = torch.mean(torch.norm(grads_tensor, dim=1)**2)
            first_moment_sq = torch.norm(torch.mean(grads_tensor, dim=0))**2
            variance = second_moment - first_moment_sq
            
            temperature = float(variance / (2.0 * grads_tensor.shape[1]))
            
            return {
                'kappa': kappa,
                'effective_temperature': temperature,
                'gradient_variance': float(variance)
            }
        except Exception:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }


class SpectralGeometryCalculator(IMetricCalculator):
    """
    Calculator for spectral geometry metrics including MBL level spacing.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the spectral geometry calculator.
        
        Args:
            config: Configuration containing spectral analysis parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute spectral geometry metrics.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - spectral_gap: Gap between largest eigenvalues
                - effective_dimension: Number of significant eigenvalues
                - participation_ratio: Measure of eigenvalue distribution
                - level_spacing_ratio: MBL indicator
                - largest_eigenvalue: Maximum eigenvalue
                - smallest_eigenvalue: Minimum eigenvalue
        """
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        all_weights = all_weights[:self.config.PARAM_FLATTEN_LIMIT].cpu().numpy()
        
        n = len(all_weights)
        outer_product = np.outer(all_weights, all_weights) / n
        outer_product += np.eye(n) * self.config.EIGENVALUE_TOLERANCE
        
        try:
            eigenvalues = eigh(outer_product, eigvals_only=True)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            effective_dim = np.sum(eigenvalues > self.config.EIGENVALUE_TOLERANCE)
            
            spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0
            
            participation_ratio = (np.sum(eigenvalues)**2) / (np.sum(eigenvalues**2) + 1e-10)
            
            level_spacing = np.diff(eigenvalues)
            level_spacing_ratio = self._compute_level_spacing_ratio(level_spacing)
            
            return {
                'spectral_gap': float(spectral_gap),
                'effective_dimension': int(effective_dim),
                'participation_ratio': float(participation_ratio),
                'level_spacing_ratio': float(level_spacing_ratio),
                'largest_eigenvalue': float(eigenvalues[0]),
                'smallest_eigenvalue': float(eigenvalues[-1])
            }
        except Exception as e:
            return {
                'spectral_gap': 0.0,
                'effective_dimension': 0,
                'participation_ratio': 0.0,
                'level_spacing_ratio': 0.0,
                'largest_eigenvalue': 0.0,
                'smallest_eigenvalue': 0.0,
                'error': str(e)
            }
    
    def _compute_level_spacing_ratio(self, spacings: np.ndarray) -> float:
        """
        Compute the level spacing ratio for MBL analysis.
        
        Args:
            spacings: Array of eigenvalue level spacings.
        
        Returns:
            Average level spacing ratio.
        """
        if len(spacings) < 2:
            return 0.0
        ratios = []
        for i in range(len(spacings) - 1):
            s1 = abs(spacings[i])
            s2 = abs(spacings[i+1])
            if s1 > 1e-15 and s2 > 1e-15:
                ratios.append(min(s1, s2) / max(s1, s2))
        return np.mean(ratios) if ratios else 0.0


class RicciCurvatureCalculator(IMetricCalculator):
    """
    Calculator for Ricci curvature estimation in weight space.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Ricci curvature calculator.
        
        Args:
            config: Configuration containing curvature estimation parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute Ricci curvature metrics.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - ricci_scalar: Estimated Ricci scalar curvature
                - mean_sectional_curvature: Average sectional curvature
                - curvature_variance: Variance of sectional curvatures
        """
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        n = min(len(all_weights), self.config.PARAM_FLATTEN_LIMIT)
        w = all_weights[:n].cpu().numpy()
        
        metric_tensor = np.outer(w, w) / n
        metric_tensor += np.eye(n) * self.config.EIGENVALUE_TOLERANCE
        
        ricci_scalar = self._compute_ricci_scalar(metric_tensor)
        sectional_curvatures = self._estimate_sectional_curvatures(metric_tensor)
        
        return {
            'ricci_scalar': float(ricci_scalar),
            'mean_sectional_curvature': float(np.mean(sectional_curvatures)),
            'curvature_variance': float(np.var(sectional_curvatures))
        }
    
    def _compute_ricci_scalar(self, metric: np.ndarray) -> float:
        """
        Compute the Ricci scalar from the metric tensor.
        
        Args:
            metric: Metric tensor as numpy array.
        
        Returns:
            Estimated Ricci scalar.
        """
        eigenvalues = eigh(metric, eigvals_only=True)
        eigenvalues = eigenvalues[eigenvalues > self.config.EIGENVALUE_TOLERANCE]
        n = len(eigenvalues)
        if n < 2:
            return 0.0
        ricci_scalar = n * np.sum(1.0 / eigenvalues)
        return ricci_scalar
    
    def _estimate_sectional_curvatures(self, metric: np.ndarray, samples: int = None) -> np.ndarray:
        """
        Estimate sectional curvatures by sampling 2D sections.
        
        Args:
            metric: Metric tensor as numpy array.
            samples: Number of sectional curvature samples.
        
        Returns:
            Array of estimated sectional curvatures.
        """
        samples = samples or self.config.RICCI_CURVATURE_SAMPLES
        curvatures = []
        n = metric.shape[0]
        for _ in range(samples):
            i, j = np.random.choice(n, 2, replace=False)
            block = metric[np.ix_([i, j], [i, j])]
            det = np.linalg.det(block)
            if det > self.config.EIGENVALUE_TOLERANCE:
                curvatures.append(1.0 / det)
        return np.array(curvatures) if curvatures else np.array([0.0])


class ThermodynamicCalculator(IMetricCalculator):
    """
    Calculator for thermodynamic potentials including Gibbs free energy.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the thermodynamic calculator.
        
        Args:
            config: Configuration containing thermodynamic parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute thermodynamic metrics.
        
        Args:
            model: Neural network model (unused but required by interface).
            **kwargs: Must contain delta, alpha, kappa, effective_temperature.
        
        Returns:
            Dictionary containing:
                - gibbs_free_energy: Gibbs free energy estimate
                - entropy_proxy: Entropy approximation
                - critical_temperature_estimate: Predicted critical temperature
                - phase_stability: Stability classification
                - phase_type: Phase classification string
        """
        delta = kwargs.get('delta', 1.0)
        alpha = kwargs.get('alpha', 0.0)
        kappa = kwargs.get('kappa', 1.0)
        t_eff = kwargs.get('effective_temperature', 1.0)
        
        internal_energy_proxy = delta
        entropy_proxy = -alpha
        
        gibbs_free_energy = internal_energy_proxy - (t_eff * entropy_proxy) if t_eff > 0 else internal_energy_proxy
        
        T_0 = 1e-3
        c = 0.5
        T_critical_predicted = T_0 * np.exp(-c * alpha)
        
        phase_stability = "stable" if t_eff < T_critical_predicted else "unstable"
        
        return {
            'gibbs_free_energy': float(gibbs_free_energy),
            'entropy_proxy': float(entropy_proxy),
            'critical_temperature_estimate': float(T_critical_predicted),
            'phase_stability': phase_stability,
            'phase_type': self._classify_phase(delta, kappa, t_eff, alpha)
        }
    
    def _classify_phase(self, delta: float, kappa: float, temp: float, alpha: float) -> str:
        """
        Classify the thermodynamic phase based on metrics.
        
        Args:
            delta: Discretization margin.
            kappa: Condition number.
            temp: Effective temperature.
            alpha: Purity index.
        
        Returns:
            Phase classification string.
        """
        if delta < self.config.DELTA_CRYSTAL_THRESHOLD and kappa < self.config.KAPPA_CRYSTAL_THRESHOLD and temp < self.config.TEMPERATURE_CRYSTAL_THRESHOLD:
            return "Perfect Crystal"
        if delta < self.config.DELTA_CRYSTAL_THRESHOLD and kappa >= self.config.KAPPA_CRYSTAL_THRESHOLD:
            return "Polycrystalline"
        if delta >= self.config.DELTA_GLASS_THRESHOLD and temp < self.config.TEMPERATURE_CRYSTAL_THRESHOLD:
            return "Cold Glass"
        if kappa > 1e6:
            return "Amorphous Glass"
        if alpha > self.config.ALPHA_CRYSTAL_THRESHOLD:
            return "Topological Insulator"
        return "Functional Glass"


class KappaQuantumCalculator(IMetricCalculator):
    """
    Calculator for quantum condition number of the weight covariance.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the kappa quantum calculator.
        
        Args:
            config: Configuration containing quantum parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute quantum condition number.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing kappa_quantum value.
        """
        flat_params = []
        for param in model.parameters():
            if param.numel() > 0:
                flat_params.append(param.data.detach().flatten())
        
        if not flat_params:
            return {'kappa_quantum': 1.0}
        
        W = torch.cat(flat_params)[:self.config.KAPPA_MAX_DIMENSION]
        n = W.numel()
        
        if n < 2:
            return {'kappa_quantum': 1.0}
        
        params_centered = W - W.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + self.config.HBAR * torch.eye(n, device=W.device)
        
        try:
            eigenvals = torch.linalg.eigvalsh(cov_matrix)
            eigenvals = eigenvals[eigenvals > self.config.HBAR]
            kappa_q = (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
            return {'kappa_quantum': kappa_q}
        except Exception:
            return {'kappa_quantum': 1.0}


class PoyntingVectorCalculator(IMetricCalculator):
    """
    Calculator for Poynting vector magnitude representing energy flow.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the Poynting vector calculator.
        
        Args:
            config: Configuration containing energy flow parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute Poynting vector metrics.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - poynting_magnitude: Magnitude of energy flow
                - is_radiating: Boolean indicating significant energy flow
                - field_orthogonality: Measure of field orthogonality
                - energy_distribution: Dictionary of energy distribution metrics
        """
        all_params = []
        for param in model.parameters():
            if param is not None and param.numel() > 0:
                all_params.append(param.data.detach().flatten())
        
        if not all_params:
            return {
                'poynting_magnitude': 0.0,
                'is_radiating': False,
                'field_orthogonality': 0.0,
                'energy_distribution': {}
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
            'is_radiating': float(poynting_magnitude.item()) > self.config.POYNTING_THRESHOLD,
            'field_orthogonality': float(H_magnitude.item()),
            'energy_distribution': energy_distribution
        }


class HbarEffectiveCalculator(IMetricCalculator):
    """
    Calculator for effective Planck constant under lambda pressure.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the hbar effective calculator.
        
        Args:
            config: Configuration containing physical constants.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute effective hbar.
        
        Args:
            model: Neural network model to analyze.
            **kwargs: Must contain delta and lambda_pressure.
        
        Returns:
            Dictionary containing hbar_effective value.
        """
        delta = kwargs.get('delta', 0.0)
        lambda_pressure = kwargs.get('lambda_pressure', 1.0)
        
        if lambda_pressure <= 0:
            return {'hbar_effective': 0.0}
        
        omega = math.sqrt(abs(lambda_pressure))
        if omega < self.config.NORMALIZATION_EPSILON:
            return {'hbar_effective': 0.0}
        
        hbar_eff = (delta**2 * lambda_pressure) / omega
        return {'hbar_effective': hbar_eff}


class WeightDiffractionCalculator(IMetricCalculator):
    """
    Calculator for weight diffraction analysis with advanced Bragg peak detection.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the weight diffraction calculator.
        
        Args:
            config: Configuration containing spectral analysis parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute weight diffraction metrics with Bragg peak detection.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - bragg_peaks: List of detected Bragg peaks
                - is_crystalline_structure: Boolean indicating crystalline pattern
                - spectral_entropy: Entropy of power spectrum
                - num_peaks: Number of detected peaks
        """
        all_weights = []
        for param in model.parameters():
            if param.numel() > 0:
                all_weights.append(param.data.detach().flatten()[:self.config.PARAM_FLATTEN_LIMIT])
        
        if not all_weights:
            return {
                'bragg_peaks': [],
                'is_crystalline_structure': False,
                'spectral_entropy': 0.0,
                'num_peaks': 0
            }
        
        W = torch.cat(all_weights)[:self.config.PARAM_FLATTEN_LIMIT]
        w_numpy = W.cpu().numpy()
        
        fft_coeffs = np.fft.fft(w_numpy)
        power_spectrum = np.abs(fft_coeffs)**2
        frequencies = np.fft.fftfreq(len(w_numpy))
        
        threshold = np.mean(power_spectrum) + self.config.SPECTRAL_PEAK_THRESHOLD_SIGMA * np.std(power_spectrum)
        
        peaks, properties = signal.find_peaks(power_spectrum, height=threshold)
        
        bragg_peaks = []
        for peak_idx in peaks[:self.config.SPECTRAL_PEAK_LIMIT]:
            bragg_peaks.append({
                'frequency': float(frequencies[peak_idx]),
                'intensity': float(power_spectrum[peak_idx]),
                'q_vector': float(2 * np.pi * frequencies[peak_idx])
            })
        
        bragg_peaks.sort(key=lambda x: x['intensity'], reverse=True)
        
        is_crystalline = len(bragg_peaks) > 0 and len(bragg_peaks) < len(power_spectrum) // 2
        
        spectral_entropy = self._compute_spectral_entropy(torch.from_numpy(power_spectrum))
        
        return {
            'bragg_peaks': bragg_peaks,
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': spectral_entropy,
            'num_peaks': len(bragg_peaks)
        }
    
    def _compute_spectral_entropy(self, power_spectrum: torch.Tensor) -> float:
        """
        Compute spectral entropy of the power spectrum.
        
        Args:
            power_spectrum: Power spectrum tensor.
        
        Returns:
            Spectral entropy value.
        """
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy.item())


class PhaseStructureCalculator(IMetricCalculator):
    """
    Calculator for phase structure analysis using histogram-based methods.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the phase structure calculator.
        
        Args:
            config: Configuration containing histogram parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute phase structure metrics.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - per_layer: Per-layer phase classifications
                - distribution: Count of each phase type
                - dominant_phase: Most common phase type
        """
        phase_classifications = {}
        
        for name, param in model.named_parameters():
            w = param.detach().cpu().flatten().numpy()
            
            hist, bin_edges = np.histogram(w, bins=self.config.HISTOGRAM_BINS)
            smoothed_hist = gaussian_filter(hist.astype(float), sigma=2)
            
            peaks, _ = signal.find_peaks(smoothed_hist, prominence=np.max(smoothed_hist) * 0.1)
            
            if len(peaks) == 0:
                phase = 'amorphous'
            elif len(peaks) == 1:
                phase = 'single_phase'
            elif len(peaks) == 2:
                phase = 'two_phase'
            else:
                phase = 'multi_phase'
            
            phase_classifications[name] = phase
        
        phase_counts = {}
        for phase in phase_classifications.values():
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            'per_layer': phase_classifications,
            'distribution': phase_counts,
            'dominant_phase': max(phase_counts, key=phase_counts.get) if phase_counts else 'unknown'
        }


class ComplexKernelHolomorphyCalculator(IMetricCalculator):
    """
    Calculator for analyzing holomorphy of complex spectral kernels using Cauchy-Riemann equations.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the holomorphy calculator.
        
        Args:
            config: Configuration containing holomorphy threshold.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute holomorphy metrics for complex kernels.
        
        Args:
            model: Neural network model to analyze.
        
        Returns:
            Dictionary containing:
                - per_layer: Per-layer holomorphy analysis
                - holomorphic_fraction: Fraction of holomorphic layers
                - average_cr_error: Average Cauchy-Riemann error
        """
        state_dict = model.state_dict()
        
        real_weights = {}
        imag_weights = {}
        
        for name, tensor in state_dict.items():
            if 'kernel_real' in name:
                real_weights[name] = tensor
            elif 'kernel_imag' in name:
                imag_weights[name] = tensor
        
        results = {}
        
        for real_name, real_tensor in real_weights.items():
            imag_name = real_name.replace('kernel_real', 'kernel_imag')
            if imag_name not in imag_weights:
                continue
            
            imag_tensor = imag_weights[imag_name]
            
            real_part = real_tensor.detach().cpu().numpy()
            imag_part = imag_tensor.detach().cpu().numpy()
            
            if real_part.ndim >= 2 and imag_part.ndim >= 2:
                if real_part.ndim > 2:
                    real_2d = real_part.reshape(real_part.shape[0], -1)
                    imag_2d = imag_part.reshape(imag_part.shape[0], -1)
                else:
                    real_2d = real_part
                    imag_2d = imag_part
                
                du_dx = np.gradient(real_2d, axis=1)
                du_dy = np.gradient(real_2d, axis=0)
                dv_dx = np.gradient(imag_2d, axis=1)
                dv_dy = np.gradient(imag_2d, axis=0)
                
                cr_residual_1 = np.abs(du_dx - dv_dy)
                cr_residual_2 = np.abs(du_dy + dv_dx)
                
                total_residual = np.mean(cr_residual_1) + np.mean(cr_residual_2)
                
                results[real_name] = {
                    'is_holomorphic': bool(total_residual < self.config.HOLMORPHY_THRESHOLD),
                    'cauchy_riemann_error': float(total_residual),
                    'holomorphic_score': float(1.0 / (1.0 + total_residual))
                }
        
        holomorphic_count = sum(1 for r in results.values() if r['is_holomorphic'])
        
        return {
            'per_layer': results,
            'holomorphic_fraction': float(holomorphic_count / len(results)) if results else 0.0,
            'average_cr_error': float(np.mean([r['cauchy_riemann_error'] for r in results.values()])) if results else 0.0
        }


class NormConservationCalculator(IMetricCalculator):
    """
    Calculator for norm conservation error in Schrodinger dynamics.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the norm conservation calculator.
        
        Args:
            config: Configuration containing normalization parameters.
        """
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Compute norm conservation error.
        
        Args:
            model: Neural network model to analyze.
            **kwargs: Must contain val_x tensor.
        
        Returns:
            Dictionary containing norm_conservation_error value.
        """
        val_x = kwargs.get('val_x')
        if val_x is None:
            return {'norm_conservation_error': 0.0}
        
        model.eval()
        with torch.no_grad():
            outputs = model(val_x)
            input_norms = torch.norm(val_x.view(val_x.size(0), -1), dim=1)
            output_norms = torch.norm(outputs.view(outputs.size(0), -1), dim=1)
            relative_error = torch.abs(output_norms - input_norms) / (input_norms + self.config.NORMALIZATION_EPSILON)
            return {'norm_conservation_error': float(relative_error.mean().item())}


class CrystallographicGrader:
    """
    Advanced grader for crystallographic quality assessment.
    Implements refined threshold-based grading system.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the crystallographic grader.
        
        Args:
            config: Configuration containing grading thresholds.
        """
        self.config = config
    
    def assign_grade(
        self, delta: float, alpha: float, kappa: float, num_bragg_peaks: int
    ) -> Dict[str, Any]:
        """
        Assign a crystallographic grade based on multiple metrics.
        
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
                - is_crystalline: Boolean indicating crystalline classification
        """
        if delta < self.config.DELTA_OPTICAL_THRESHOLD and alpha > self.config.ALPHA_PERFECT_CRYSTAL_THRESHOLD:
            grade = "Optical Crystal"
            description = "Perfect quantum crystalline structure. Ideal for quantum information processing."
            quality_score = 1.0
        elif delta < self.config.DELTA_INDUSTRIAL_THRESHOLD and alpha > self.config.ALPHA_CRYSTAL_THRESHOLD:
            grade = "Industrial Crystal"
            description = "High-quality crystalline structure. Robust for production quantum computing."
            quality_score = 0.85
        elif delta < self.config.DELTA_POLYCRYSTALLINE_THRESHOLD:
            grade = "Polycrystalline"
            description = "Multiple crystalline domains. Good generalization but some structural defects."
            quality_score = 0.65
        elif delta < self.config.DELTA_AMORPHOUS_THRESHOLD:
            grade = "Amorphous Glass"
            description = "Glassy structure with local order. Trapped in local minimum."
            quality_score = 0.40
        else:
            grade = "Defective"
            description = "Highly disordered structure. Requires retraining."
            quality_score = 0.20
        
        crystalline_features = 0
        if alpha > self.config.ALPHA_CRYSTAL_THRESHOLD:
            crystalline_features += 1
        if kappa < self.config.KAPPA_WELL_CONDITIONED_THRESHOLD:
            crystalline_features += 1
        if num_bragg_peaks > 5:
            crystalline_features += 1
        
        return {
            'grade': grade,
            'description': description,
            'quality_score': quality_score,
            'crystalline_features': crystalline_features,
            'is_crystalline': alpha > self.config.ALPHA_CRYSTAL_THRESHOLD
        }


class PhaseClassifier:
    """
    Classifier for thermodynamic phase identification.
    """
    
    @staticmethod
    def classify(
        metrics: Dict[str, Any],
        config: SchrodingerCrystallographyConfig
    ) -> Dict[str, Any]:
        """
        Classify the thermodynamic phase based on computed metrics.
        
        Args:
            metrics: Dictionary of computed metrics.
            config: Configuration containing phase thresholds.
        
        Returns:
            Dictionary containing:
                - phase: Phase classification string
                - confidence: Classification confidence (0-1)
                - is_crystal: Boolean indicating crystalline phase
        """
        delta = metrics.get('discretization', {}).get('delta', 1.0)
        kappa = metrics.get('gradient_dynamics', {}).get('kappa', float('inf'))
        temp = metrics.get('gradient_dynamics', {}).get('effective_temperature', 1.0)
        alpha = metrics.get('discretization', {}).get('alpha', 0.0)
        lc = metrics.get('local_complexity', {}).get('local_complexity', 0.5)
        sp = metrics.get('superposition', {}).get('superposition', 0.5)
        
        phase = "Unknown"
        confidence = 0.0
        
        if delta < config.DELTA_CRYSTAL_THRESHOLD and kappa < config.KAPPA_CRYSTAL_THRESHOLD and temp < config.TEMPERATURE_CRYSTAL_THRESHOLD:
            phase = "Perfect Crystal"
            confidence = 0.95
        elif delta < config.DELTA_CRYSTAL_THRESHOLD and kappa >= config.KAPPA_CRYSTAL_THRESHOLD:
            phase = "Polycrystalline"
            confidence = 0.8
        elif delta >= config.DELTA_GLASS_THRESHOLD and temp < config.TEMPERATURE_CRYSTAL_THRESHOLD:
            phase = "Cold Glass"
            confidence = 0.7
        elif kappa > 1e6:
            phase = "Amorphous Glass"
            confidence = 0.9
        elif alpha > config.ALPHA_CRYSTAL_THRESHOLD:
            phase = "Topological Insulator"
            confidence = 0.6
        else:
            phase = "Functional Glass"
            confidence = 0.4
        
        if delta < 0.001 and alpha > 7.0 and kappa < 1.1 and lc < 0.1 and sp < 0.1 and temp < 1e-10:
            phase = "Perfect Crystal (Optical Crystal)"
            confidence = 0.98
        elif delta < 0.1 and alpha > 1.0:
            if phase != "Perfect Crystal":
                phase = "Polycrystalline"
                confidence = max(confidence, 0.75)
        elif delta > 0.4 and alpha < 1.0:
            phase = "Amorphous Glass"
            confidence = 0.85
        
        return {
            'phase': phase,
            'confidence': confidence,
            'is_crystal': phase in ["Perfect Crystal", "Polycrystalline", "Perfect Crystal (Optical Crystal)"]
        }


class CheckpointLoader:
    """
    Loader for neural network checkpoints with architecture reconstruction.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the checkpoint loader.
        
        Args:
            config: Configuration containing architecture parameters.
        """
        self.config = config
        self.logger = LoggerFactory.create_logger("CheckpointLoader")
    
    def load(self, checkpoint_path: str) -> Optional[nn.Module]:
        """
        Load a model from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        
        Returns:
            Loaded model or None if loading fails.
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
            
            model = SchrodingerSpectralNetwork(self.config).to(self.config.DEVICE)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.logger.info(f"Model loaded successfully from {checkpoint_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def extract_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Extract metadata from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        
        Returns:
            Dictionary containing checkpoint metadata.
        """
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False
            )
            metadata = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'phase': checkpoint.get('phase', 'unknown'),
                'lambda_pressure': checkpoint.get('lambda_pressure', 0.0)
            }
            return metadata
        except Exception:
            return {
                'epoch': 'unknown',
                'timestamp': 'unknown',
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'phase': 'unknown',
                'lambda_pressure': 0.0
            }
    
    def categorize_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Categorize weights by layer type for detailed analysis.
        
        Args:
            state_dict: Model state dictionary.
        
        Returns:
            Dictionary of categorized weight tensors.
        """
        categories = {
            'spectral_real': {},
            'spectral_imaginary': {},
            'encoder': {},
            'decoder': {},
            'projection': {},
            'expansion': {},
            'contraction': {},
            'other': {}
        }
        
        for name, tensor in state_dict.items():
            if 'spectral' in name.lower():
                if 'real' in name.lower() or 'kernel_real' in name.lower():
                    categories['spectral_real'][name] = tensor
                elif 'imag' in name.lower() or 'kernel_imag' in name.lower():
                    categories['spectral_imaginary'][name] = tensor
                else:
                    categories['spectral_real'][name] = tensor
            elif 'encoder' in name.lower() or 'input_proj' in name.lower():
                categories['encoder'][name] = tensor
            elif 'decoder' in name.lower() or 'output_proj' in name.lower():
                categories['decoder'][name] = tensor
            elif 'expansion' in name.lower():
                categories['expansion'][name] = tensor
            elif 'contraction' in name.lower():
                categories['contraction'][name] = tensor
            elif 'proj' in name.lower():
                categories['projection'][name] = tensor
            else:
                categories['other'][name] = tensor
        
        return categories


class DefinitiveCrystallographySuite:
    """
    Comprehensive crystallographic analysis suite combining all metric calculators.
    Follows the Single Responsibility Principle for orchestration.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the crystallography suite with all calculators.
        
        Args:
            config: Configuration containing all analysis parameters.
        """
        self.config = config
        self.logger = LoggerFactory.create_logger("DefinitiveCrystallographySuite")
        self.loader = CheckpointLoader(config)
        self.grader = CrystallographicGrader(config)
        
        self.hamiltonian_engine = HamiltonianInferenceEngine(config)
        self.data_generator = SyntheticDataGenerator(config, self.hamiltonian_engine)
        
        self.calculators: Dict[str, IMetricCalculator] = {
            'integrity': WeightIntegrityCalculator(config),
            'discretization': DiscretizationCalculator(config),
            'local_complexity': LocalComplexityCalculator(config),
            'superposition': SuperpositionCalculator(config),
            'gradient_dynamics': GradientDynamicsCalculator(config),
            'spectral_geometry': SpectralGeometryCalculator(config),
            'ricci_curvature': RicciCurvatureCalculator(config),
            'thermodynamics': ThermodynamicCalculator(config),
            'kappa_quantum': KappaQuantumCalculator(config),
            'poynting_vector': PoyntingVectorCalculator(config),
            'hbar_effective': HbarEffectiveCalculator(config),
            'weight_diffraction': WeightDiffractionCalculator(config),
            'phase_structure': PhaseStructureCalculator(config),
            'complex_kernel_holomorphy': ComplexKernelHolomorphyCalculator(config),
            'norm_conservation': NormConservationCalculator(config)
        }
    
    def scan_directory(self, directory: str) -> List[str]:
        """
        Scan directory for checkpoint files.
        
        Args:
            directory: Path to checkpoint directory.
        
        Returns:
            List of checkpoint file paths sorted by modification time.
        """
        checkpoint_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pth') or file.endswith('.pt'):
                    checkpoint_files.append(os.path.join(root, file))
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        self.logger.info(f"Found {len(checkpoint_files)} checkpoints in {directory}")
        return checkpoint_files
    
    def analyze_checkpoint(self, checkpoint_path: str, seed: int = 42) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a single checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            seed: Random seed for reproducible analysis.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        self.logger.info(f"Analyzing: {os.path.basename(checkpoint_path)}")
        
        model = self.loader.load(checkpoint_path)
        if model is None:
            return {'error': 'Failed to load model', 'path': checkpoint_path}
        
        metadata = self.loader.extract_metadata(checkpoint_path)
        val_x, val_y = self.data_generator.generate_batch(seed=seed)
        
        all_metrics: Dict[str, Any] = {
            'metadata': metadata,
            'path': checkpoint_path,
            'checkpoint': os.path.basename(checkpoint_path)
        }
        
        first_pass_calculators = [
            'integrity', 'discretization', 'local_complexity', 'superposition',
            'gradient_dynamics', 'spectral_geometry', 'ricci_curvature',
            'kappa_quantum', 'poynting_vector', 'weight_diffraction',
            'phase_structure', 'complex_kernel_holomorphy', 'norm_conservation'
        ]
        
        for name in first_pass_calculators:
            if name not in self.calculators:
                continue
            try:
                calculator = self.calculators[name]
                metrics = calculator.compute(model, val_x=val_x, val_y=val_y)
                all_metrics[name] = metrics
            except Exception as e:
                self.logger.warning(f"Calculator {name} failed: {e}")
                all_metrics[name] = {'error': str(e)}
        
        try:
            delta = all_metrics.get('discretization', {}).get('delta', 1.0)
            alpha = all_metrics.get('discretization', {}).get('alpha', 0.0)
            kappa = all_metrics.get('gradient_dynamics', {}).get('kappa', float('inf'))
            t_eff = all_metrics.get('gradient_dynamics', {}).get('effective_temperature', 1.0)
            
            thermo_calc = self.calculators['thermodynamics']
            all_metrics['thermodynamics'] = thermo_calc.compute(
                model,
                delta=delta,
                alpha=alpha,
                kappa=kappa,
                effective_temperature=t_eff
            )
            
            hbar_calc = self.calculators['hbar_effective']
            lambda_pressure = metadata.get('lambda_pressure', 1.0)
            all_metrics['hbar_effective'] = hbar_calc.compute(
                model,
                delta=delta,
                lambda_pressure=lambda_pressure
            )
        except Exception as e:
            self.logger.warning(f"Dependent metrics calculation failed: {e}")
        
        num_bragg_peaks = all_metrics.get('weight_diffraction', {}).get('num_peaks', 0)
        all_metrics['crystallographic_grading'] = self.grader.assign_grade(
            delta, alpha, kappa, num_bragg_peaks
        )
        
        all_metrics['phase_classification'] = PhaseClassifier.classify(all_metrics, self.config)
        
        all_metrics['purity_index'] = 1.0 - delta
        all_metrics['is_crystal'] = alpha > self.config.ALPHA_CRYSTAL_THRESHOLD
        
        if isinstance(all_metrics.get('poynting_vector'), dict):
            all_metrics['energy_flow'] = all_metrics['poynting_vector'].get('poynting_magnitude', 0.0)
        
        return all_metrics
    
    def run_full_analysis(self, directory: str, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Run comprehensive analysis on all checkpoints in a directory.
        
        Args:
            directory: Path to checkpoint directory.
            seed: Random seed for reproducible analysis.
        
        Returns:
            List of analysis results for each checkpoint.
        """
        checkpoints = self.scan_directory(directory)
        results = []
        
        for i, ckpt_path in enumerate(checkpoints):
            self.logger.info(f"Processing {i+1}/{len(checkpoints)}: {os.path.basename(ckpt_path)}")
            result = self.analyze_checkpoint(ckpt_path, seed=seed)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate JSON report from analysis results.
        
        Args:
            results: List of analysis results.
            output_path: Path for output JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Report saved to {output_path}")
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate aggregate summary from analysis results.
        
        Args:
            results: List of analysis results.
        
        Returns:
            Dictionary containing aggregate statistics.
        """
        if not results:
            return {'total_checkpoints': 0}
        
        grades = [r.get('crystallographic_grading', {}).get('grade', 'Unknown') for r in results]
        grade_distribution = dict(Counter(grades))
        
        numeric_keys_map = {
            'delta': lambda r: r.get('discretization', {}).get('delta', None),
            'alpha': lambda r: r.get('discretization', {}).get('alpha', None),
            'kappa': lambda r: r.get('gradient_dynamics', {}).get('kappa', None),
            'kappa_quantum': lambda r: r.get('kappa_quantum', {}).get('kappa_quantum', None),
            'local_complexity': lambda r: r.get('local_complexity', {}).get('local_complexity', None),
            'superposition': lambda r: r.get('superposition', {}).get('superposition', None),
            'effective_temperature': lambda r: r.get('gradient_dynamics', {}).get('effective_temperature', None),
            'poynting_magnitude': lambda r: r.get('poynting_vector', {}).get('poynting_magnitude', None),
            'ricci_scalar': lambda r: r.get('ricci_curvature', {}).get('ricci_scalar', None),
            'spectral_entropy': lambda r: r.get('discretization', {}).get('spectral_entropy', None),
        }
        
        statistics = {}
        for key, extractor in numeric_keys_map.items():
            values = []
            for r in results:
                val = extractor(r)
                if val is not None and isinstance(val, (int, float)) and not np.isinf(val) and not np.isnan(val):
                    values.append(val)
            
            if values:
                statistics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        phases = [r.get('phase_classification', {}).get('phase', 'Unknown') for r in results]
        phase_distribution = dict(Counter(phases))
        
        crystalline_count = sum(1 for r in results if r.get('is_crystal', False))
        
        best_checkpoint = None
        best_alpha = 0.0
        for r in results:
            alpha = r.get('discretization', {}).get('alpha', 0.0)
            if alpha > best_alpha:
                best_alpha = alpha
                best_checkpoint = r
        
        return {
            'total_checkpoints': len(results),
            'grade_distribution': grade_distribution,
            'phase_distribution': phase_distribution,
            'statistics': statistics,
            'crystalline_fraction': float(crystalline_count / len(results)),
            'best_checkpoint': {
                'name': best_checkpoint.get('checkpoint', 'Unknown'),
                'alpha': best_alpha,
                'delta': best_checkpoint.get('discretization', {}).get('delta', 0.0),
                'grade': best_checkpoint.get('crystallographic_grading', {}).get('grade', 'Unknown')
            } if best_checkpoint else None,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """
        Print formatted summary table to console.
        
        Args:
            results: List of analysis results.
        """
        print("\n" + "=" * 160)
        print(f"{'CHECKPOINT':<50} | {'GRADE':<20} | {'PHASE':<25} | {'DELTA':<10} | {'ALPHA':<8} | {'KAPPA':<12} | {'T_EFF':<12} | {'RICCI':<12}")
        print("=" * 160)
        
        for res in results:
            path = os.path.basename(res.get('path', 'N/A'))
            grade = res.get('crystallographic_grading', {}).get('grade', 'Error')
            phase = res.get('phase_classification', {}).get('phase', 'Error')
            delta = res.get('discretization', {}).get('delta', float('nan'))
            alpha = res.get('discretization', {}).get('alpha', float('nan'))
            kappa = res.get('gradient_dynamics', {}).get('kappa', float('nan'))
            temp = res.get('gradient_dynamics', {}).get('effective_temperature', float('nan'))
            ricci = res.get('ricci_curvature', {}).get('ricci_scalar', float('nan'))
            
            delta_str = f"{delta:.6f}" if not np.isinf(delta) and not np.isnan(delta) else "inf"
            alpha_str = f"{alpha:.2f}" if not np.isinf(alpha) and not np.isnan(alpha) else "inf"
            kappa_str = f"{kappa:.2e}" if not np.isinf(kappa) and not np.isnan(kappa) else "inf"
            temp_str = f"{temp:.2e}" if not np.isinf(temp) and not np.isnan(temp) else "inf"
            ricci_str = f"{ricci:.2e}" if not np.isinf(ricci) and not np.isnan(ricci) else "inf"
            
            print(f"{path:<50} | {grade:<20} | {phase:<25} | {delta_str:<10} | {alpha_str:<8} | {kappa_str:<12} | {temp_str:<12} | {ricci_str:<12}")
        
        print("=" * 160)


class BatchCrystallographyAnalyzer:
    """
    Batch analysis orchestrator with visualization generation.
    """
    
    def __init__(self, config: SchrodingerCrystallographyConfig):
        """
        Initialize the batch analyzer.
        
        Args:
            config: Configuration containing all analysis parameters.
        """
        self.config = config
        self.logger = LoggerFactory.create_logger("BatchCrystallographyAnalyzer")
        self.suite = DefinitiveCrystallographySuite(config)
        self.results_dir = Path(config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def analyze_directory(self, directory: str, seed: int = 42) -> Dict[str, Any]:
        """
        Analyze all checkpoints in a directory with visualization.
        
        Args:
            directory: Path to checkpoint directory.
            seed: Random seed for reproducible analysis.
        
        Returns:
            Dictionary containing summary and individual results.
        """
        self.logger.info("=" * 80)
        self.logger.info("DEFINITIVE SCHRÖDINGER CRYSTALLOGRAPHIC ANALYSIS - BATCH MODE")
        self.logger.info("=" * 80)
        self.logger.info(f"Checkpoint directory: {directory}")
        
        results = self.suite.run_full_analysis(directory, seed=seed)
        
        if not results:
            self.logger.warning("No valid checkpoints analyzed")
            return {'total_checkpoints': 0, 'results': []}
        
        summary = self.suite.generate_summary(results)
        self._save_summary(summary, results)
        self._generate_visualization(results)
        self.suite.print_summary(results)
        
        return {
            'summary': summary,
            'results': results
        }
    
    def _save_summary(self, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """
        Save analysis summary and individual reports.
        
        Args:
            summary: Aggregate summary dictionary.
            results: List of individual analysis results.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_path = self.results_dir / f"definitive_crystallography_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Summary saved: {summary_path}")
        
        for result in results:
            checkpoint_name = result.get('checkpoint', 'unknown')
            safe_name = checkpoint_name.replace('.', '_').replace(' ', '_')
            report_path = self.results_dir / f"{safe_name}_analysis.json"
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    
    def _generate_visualization(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate comprehensive visualization of analysis results.
        
        Args:
            results: List of analysis results.
        """
        if not results:
            return
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        deltas = [r.get('discretization', {}).get('delta', 0.0) for r in results]
        alphas = [r.get('discretization', {}).get('alpha', 0.0) for r in results]
        kappas = [r.get('gradient_dynamics', {}).get('kappa', 1.0) for r in results]
        kappas_filtered = [k if not np.isinf(k) else 1e10 for k in kappas]
        ricci_scalars = [r.get('ricci_curvature', {}).get('ricci_scalar', 0.0) for r in results]
        temps = [r.get('gradient_dynamics', {}).get('effective_temperature', 0.0) for r in results]
        bragg_counts = [r.get('weight_diffraction', {}).get('num_peaks', 0) for r in results]
        lc_values = [r.get('local_complexity', {}).get('local_complexity', 0.0) for r in results]
        sp_values = [r.get('superposition', {}).get('superposition', 0.0) for r in results]
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(deltas, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(self.config.DELTA_OPTICAL_THRESHOLD, color='gold', linestyle='--', linewidth=2, label='Optical')
        ax1.axvline(self.config.DELTA_INDUSTRIAL_THRESHOLD, color='orange', linestyle='--', linewidth=2, label='Industrial')
        ax1.axvline(self.config.DELTA_POLYCRYSTALLINE_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Poly')
        ax1.set_xlabel('Delta', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Discretization Margin Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(alphas, bins=30, color='gold', alpha=0.7, edgecolor='black')
        ax2.axvline(self.config.ALPHA_CRYSTAL_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Crystal Threshold')
        ax2.set_xlabel('Alpha', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Purity Index Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(np.log10(np.array(kappas_filtered) + 1), bins=30, color='salmon', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('log10(Kappa + 1)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Condition Number Distribution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(np.log10(np.array(temps) + 1e-20), bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('log10(T_eff)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Effective Temperature Distribution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        grades = [r.get('crystallographic_grading', {}).get('grade', 'Unknown') for r in results]
        grade_counts = dict(Counter(grades))
        
        ax5 = fig.add_subplot(gs[1, :2])
        grade_colors = {
            'Optical Crystal': 'gold',
            'Industrial Crystal': 'steelblue',
            'Polycrystalline': 'orange',
            'Amorphous Glass': 'gray',
            'Defective': 'red'
        }
        colors = [grade_colors.get(g, 'blue') for g in grade_counts.keys()]
        ax5.bar(grade_counts.keys(), grade_counts.values(), color=colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Count', fontsize=11)
        ax5.set_title('Crystallographic Grade Distribution', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(bragg_counts, bins=20, color='crimson', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Number of Bragg Peaks', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Diffraction Peak Distribution', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.hist(lc_values, bins=20, color='teal', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Local Complexity', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('Local Complexity Distribution', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, :])
        scatter_colors = [grade_colors.get(r.get('crystallographic_grading', {}).get('grade', 'Unknown'), 'blue') for r in results]
        ax8.scatter(deltas, alphas, c=scatter_colors, s=100, alpha=0.6, edgecolor='black')
        ax8.axhline(self.config.ALPHA_CRYSTAL_THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax8.axvline(self.config.DELTA_INDUSTRIAL_THRESHOLD, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        ax8.set_xlabel('Delta - Discretization Margin', fontsize=12)
        ax8.set_ylabel('Alpha - Purity Index', fontsize=12)
        ax8.set_title('Crystallographic Phase Diagram', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        names = [r.get('checkpoint', '')[:15] for r in results]
        for i, (d, a, name) in enumerate(zip(deltas[:15], alphas[:15], names[:15])):
            ax8.annotate(name, (d, a), fontsize=7, alpha=0.7)
        
        ax9 = fig.add_subplot(gs[3, :2])
        ax9.scatter(np.log10(np.array(kappas_filtered) + 1), alphas, c=scatter_colors, s=80, alpha=0.6, edgecolor='black')
        ax9.set_xlabel('log10(Kappa + 1)', fontsize=11)
        ax9.set_ylabel('Alpha', fontsize=11)
        ax9.set_title('Condition Number vs Purity', fontsize=13, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(gs[3, 2:])
        ax10.scatter(np.log10(np.array(temps) + 1e-20), ricci_scalars, c=scatter_colors, s=80, alpha=0.6, edgecolor='black')
        ax10.set_xlabel('log10(T_eff)', fontsize=11)
        ax10.set_ylabel('Ricci Scalar', fontsize=11)
        ax10.set_title('Temperature vs Curvature', fontsize=13, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.results_dir / f"definitive_crystallography_visualization_{timestamp}.png"
        plt.savefig(viz_path, dpi=self.config.VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved: {viz_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Definitive Schrödinger Crystallographer - Comprehensive Analysis of Quantum Neural Network Checkpoints'
    )
    
    parser.add_argument(
        'checkpoint_dir', type=str,
        help='Directory containing checkpoint files to analyze'
    )
    parser.add_argument(
        '--results_dir', type=str, default=SchrodingerCrystallographyConfig.RESULTS_DIR,
        help='Directory to save analysis reports'
    )
    parser.add_argument(
        '--seed', type=int, default=SchrodingerCrystallographyConfig.RANDOM_SEED,
        help='Random seed for reproducible analysis'
    )
    parser.add_argument(
        '--grid_size', type=int, default=SchrodingerCrystallographyConfig.GRID_SIZE,
        help='Grid size for model architecture'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=SchrodingerCrystallographyConfig.HIDDEN_DIM,
        help='Hidden dimension for model architecture'
    )
    parser.add_argument(
        '--expansion_dim', type=int, default=SchrodingerCrystallographyConfig.EXPANSION_DIM,
        help='Expansion dimension for model architecture'
    )
    parser.add_argument(
        '--num_spectral_layers', type=int, default=SchrodingerCrystallographyConfig.NUM_SPECTRAL_LAYERS,
        help='Number of spectral layers in model architecture'
    )
    parser.add_argument(
        '--backbone_path', type=str, default=SchrodingerCrystallographyConfig.BACKBONE_CHECKPOINT_PATH,
        help='Path to backbone checkpoint for Hamiltonian inference'
    )
    parser.add_argument(
        '--no_backbone', action='store_true',
        help='Disable backbone, use analytical Hamiltonian operator'
    )
    parser.add_argument(
        '--log_level', type=str, default=SchrodingerCrystallographyConfig.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--dpi', type=int, default=SchrodingerCrystallographyConfig.VISUALIZATION_DPI,
        help='DPI for visualization outputs'
    )
    
    return parser


def main() -> None:
    """
    Main entry point for the definitive crystallographer.
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    
    config = SchrodingerCrystallographyConfig()
    config.RESULTS_DIR = args.results_dir
    config.GRID_SIZE = args.grid_size
    config.HIDDEN_DIM = args.hidden_dim
    config.EXPANSION_DIM = args.expansion_dim
    config.NUM_SPECTRAL_LAYERS = args.num_spectral_layers
    config.BACKBONE_CHECKPOINT_PATH = args.backbone_path
    config.BACKBONE_ENABLED = not args.no_backbone
    config.LOG_LEVEL = args.log_level
    config.VISUALIZATION_DPI = args.dpi
    
    logger = LoggerFactory.create_logger("Main", config.LOG_LEVEL)
    logger.info("Initializing Definitive Schrödinger Crystallography System")
    logger.info(f"Target Directory: {args.checkpoint_dir}")
    logger.info(f"Architecture: Grid={config.GRID_SIZE}, Hidden={config.HIDDEN_DIM}, Expansion={config.EXPANSION_DIM}, Layers={config.NUM_SPECTRAL_LAYERS}")
    
    batch_analyzer = BatchCrystallographyAnalyzer(config)
    analysis_result = batch_analyzer.analyze_directory(args.checkpoint_dir, seed=args.seed)
    
    print("\n" + "=" * 80)
    print("DEFINITIVE CRYSTALLOGRAPHIC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Analyzed {analysis_result['summary']['total_checkpoints']} checkpoints")
    print(f"Reports saved in: {config.RESULTS_DIR}/")
    
    if analysis_result['summary'].get('best_checkpoint'):
        best = analysis_result['summary']['best_checkpoint']
        print(f"\nBest Checkpoint: {best['name']}")
        print(f"  Grade: {best['grade']}")
        print(f"  Alpha: {best['alpha']:.2f}")
        print(f"  Delta: {best['delta']:.6f}")
    
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
