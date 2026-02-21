#!/usr/bin/env python3
"""
Berry Phase Calculator for Schrodinger Crystal Training - FIXED VERSION

Calculates the geometric Berry phase accumulated along training trajectories.

Author: Analysis tool for schrodinger_crystal.py training checkpoints
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class BerryPhaseResult:
    """Results from Berry phase calculation."""
    total_berry_phase: float
    berry_phase_mod_2pi: float
    winding_number: int
    phase_discontinuities: List[Tuple[int, float]]
    trajectory_length: float
    mean_local_curvature: float
    topological_invariant: float
    cm_trajectory: List[Tuple[float, float]]
    eigenvalue_gaps: List[float]
    raw_phases: List[float]
    epochs: List[int]


class BerryPhaseCalculator:
    """
    Calculates Berry phase from training checkpoint trajectory.
    
    Uses the spectral kernel parameters as the parameter space θ,
    and computes the geometric phase accumulated during training.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def load_checkpoints(self, checkpoint_dir: str) -> List[Dict[str, Any]]:
        """Load all checkpoints from directory in chronological order."""
        pattern = os.path.join(checkpoint_dir, "*.pth")
        files = sorted(glob.glob(pattern), key=self._extract_epoch)
        
        checkpoints = []
        for f in files:
            try:
                ckpt = torch.load(f, map_location=self.device, weights_only=False)
                checkpoints.append({
                    'path': f,
                    'epoch': self._extract_epoch(f),
                    'state_dict': ckpt.get('model_state_dict', ckpt),
                    'metrics': ckpt.get('metrics', {}),
                    'lambda_pressure': ckpt.get('lambda_pressure', 0.0)
                })
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
        
        return checkpoints
    
    def _extract_epoch(self, filepath: str) -> int:
        """Extract epoch number from checkpoint filename."""
        match = re.search(r'epoch[_]?(\d+)', filepath)
        return int(match.group(1)) if match else 0
    
    def extract_spectral_kernels(self, state_dict: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract spectral kernels from model state dict.
        Returns dict with 'real' and 'imag' kernels per layer.
        """
        kernels = {'real': [], 'imag': [], 'combined': []}
        
        # Find all spectral layer indices
        layer_indices = set()
        for key in state_dict.keys():
            if 'spectral_layers' in key:
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        layer_indices.add(idx)
                    except:
                        pass
        
        # Extract kernels per layer
        for idx in sorted(layer_indices):
            real_key = f'spectral_layers.{idx}.kernel_real'
            imag_key = f'spectral_layers.{idx}.kernel_imag'
            
            if real_key in state_dict and imag_key in state_dict:
                kr = state_dict[real_key]
                ki = state_dict[imag_key]
                kernels['real'].append(kr)
                kernels['imag'].append(ki)
                kernels['combined'].append(torch.complex(kr, ki))
        
        return kernels if kernels['combined'] else None
    
    def flatten_kernel_params(self, state_dict: Dict) -> Optional[torch.Tensor]:
        """Flatten all kernel parameters into a single complex vector."""
        kernels = self.extract_spectral_kernels(state_dict)
        if kernels is None:
            return None
        
        flat_parts = []
        for kc in kernels['combined']:
            flat_parts.append(kc.flatten())
        
        return torch.cat(flat_parts)
    
    def compute_spectral_density(self, kernel: torch.Tensor) -> torch.Tensor:
        """Compute spectral density |W(k)|²."""
        return torch.abs(kernel) ** 2
    
    def compute_center_of_mass(self, kernel: torch.Tensor) -> Tuple[float, float]:
        """
        Compute center of mass in 2D Fourier space.
        
        For each kernel layer [C_out, C_in, freq_h, freq_w]:
        - Map frequency indices to angular coordinates
        - Compute weighted CM
        """
        density = self.compute_spectral_density(kernel)
        
        # Get dimensions
        if density.dim() == 4:
            # Average over channel dimensions
            density_2d = density.mean(dim=(0, 1))  # [freq_h, freq_w]
        else:
            density_2d = density
        
        freq_h, freq_w = density_2d.shape
        
        # Create coordinate grids (angular coordinates on torus)
        kx = torch.linspace(-np.pi, np.pi, freq_h, device=density.device)
        ky = torch.linspace(-np.pi, np.pi, freq_w, device=density.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        
        # Normalize density
        total_mass = density_2d.sum() + 1e-10
        density_norm = density_2d / total_mass
        
        # Center of mass
        cm_x = (KX * density_norm).sum().item()
        cm_y = (KY * density_norm).sum().item()
        
        return cm_x, cm_y
    
    def compute_berry_connection_discrete(self, theta_prev: torch.Tensor, 
                                           theta_curr: torch.Tensor) -> float:
        """
        Compute discrete Berry connection between two parameter states.
        
        Uses the formula for discrete Berry phase:
        A = Im[log(⟨ψ(θ_{n-1})|ψ(θ_n)⟩)]
        
        For parameter vectors, this is:
        A = Im[log(θ_{n-1}^* · θ_n)]
        """
        if theta_prev is None or theta_curr is None:
            return 0.0
        
        # Normalize
        theta_prev_norm = theta_prev / (torch.norm(theta_prev) + 1e-10)
        theta_curr_norm = theta_curr / (torch.norm(theta_curr) + 1e-10)
        
        # Overlap (inner product)
        overlap = torch.sum(torch.conj(theta_prev_norm) * theta_curr_norm)
        
        # Berry connection = phase of overlap
        # Use angle of complex number
        if torch.abs(overlap) < 1e-10:
            return 0.0
        
        # Phase difference
        phase = torch.angle(overlap).item()
        
        return phase
    
    def compute_eigenvalue_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalue spectrum of kernel Gram matrix."""
        # Flatten and build Hermitian matrix
        flat = kernel.flatten()
        n = min(200, len(flat))
        
        # Use outer product for Hermitian matrix
        H = torch.outer(flat[:n].real, flat[:n].real)
        if torch.is_complex(flat):
            H_imag = torch.outer(flat[:n].imag, flat[:n].imag)
            H = H + H_imag
        
        # Make Hermitian
        H = (H + H.conj().T) / 2
        
        try:
            eigenvalues = torch.linalg.eigvalsh(H)
            return torch.sort(eigenvalues, descending=True)[0]
        except:
            return torch.zeros(n)
    
    def compute_eigenvalue_gap(self, eigenvalues: torch.Tensor) -> float:
        """Compute gap between two largest eigenvalues."""
        if len(eigenvalues) < 2:
            return 0.0
        return (eigenvalues[0] - eigenvalues[1]).item()
    
    def compute_trajectory_metrics(self, kernels: List[torch.Tensor]) -> Dict[str, float]:
        """Compute trajectory metrics in parameter space."""
        lengths = []
        curvatures = []
        
        for i in range(1, len(kernels)):
            if kernels[i-1] is not None and kernels[i] is not None:
                # Path length
                length = torch.norm(kernels[i] - kernels[i-1]).item()
                lengths.append(length)
        
        # Curvature from triplets
        for i in range(2, len(kernels)):
            if all(k is not None for k in [kernels[i-2], kernels[i-1], kernels[i]]):
                v1 = kernels[i-1] - kernels[i-2]
                v2 = kernels[i] - kernels[i-1]
                
                norm1 = torch.norm(v1).item() + 1e-10
                norm2 = torch.norm(v2).item() + 1e-10
                
                # Angle between consecutive steps
                cos_angle = torch.sum(v1 * v2.conj()).real.item() / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                curvature = 1 - cos_angle  # 0 for straight, 2 for reversal
                curvatures.append(curvature)
        
        return {
            'total_length': sum(lengths),
            'mean_step_length': np.mean(lengths) if lengths else 0,
            'mean_curvature': np.mean(curvatures) if curvatures else 0,
            'max_curvature': max(curvatures) if curvatures else 0
        }
    
    def calculate_berry_phase(self, checkpoint_dir: str) -> BerryPhaseResult:
        """
        Main method to calculate Berry phase from checkpoint directory.
        """
        print(f"Loading checkpoints from {checkpoint_dir}...")
        checkpoints = self.load_checkpoints(checkpoint_dir)
        
        if len(checkpoints) < 2:
            raise ValueError(f"Need at least 2 checkpoints, found {len(checkpoints)}")
        
        print(f"Loaded {len(checkpoints)} checkpoints")
        
        # Extract kernel parameters
        kernels = []
        epochs = []
        for ckpt in checkpoints:
            kernel = self.flatten_kernel_params(ckpt['state_dict'])
            kernels.append(kernel)
            epochs.append(ckpt['epoch'])
        
        # Compute Berry phases (discrete connection)
        berry_phases = []
        for i in range(1, len(kernels)):
            if kernels[i-1] is not None and kernels[i] is not None:
                phase = self.compute_berry_connection_discrete(kernels[i-1], kernels[i])
                berry_phases.append(phase)
            else:
                berry_phases.append(0.0)
        
        # Detect discontinuities (jumps > π/2)
        discontinuities = []
        for i, phase in enumerate(berry_phases):
            if abs(phase) > np.pi / 2:
                discontinuities.append((epochs[i+1], phase))
        
        # Cumulative Berry phase (unwrapped)
        cumulative_phase = np.cumsum(berry_phases)
        total_phase = cumulative_phase[-1] if len(cumulative_phase) > 0 else 0.0
        
        # Phase mod 2π
        phase_mod_2pi = total_phase % (2 * np.pi)
        if phase_mod_2pi > np.pi:
            phase_mod_2pi -= 2 * np.pi
        
        # Winding number
        winding_number = int(round(total_phase / (2 * np.pi)))
        
        # Trajectory metrics
        traj_metrics = self.compute_trajectory_metrics(kernels)
        
        # Center of mass trajectory
        cm_trajectory = []
        for ckpt in checkpoints:
            kernels_dict = self.extract_spectral_kernels(ckpt['state_dict'])
            if kernels_dict and kernels_dict['combined']:
                # Use first layer for CM
                cm_x, cm_y = self.compute_center_of_mass(kernels_dict['combined'][0])
                cm_trajectory.append((cm_x, cm_y))
        
        # Eigenvalue gaps
        eigenvalue_gaps = []
        for kernel in kernels:
            if kernel is not None:
                eigenvalues = self.compute_eigenvalue_spectrum(kernel)
                gap = self.compute_eigenvalue_gap(eigenvalues)
                eigenvalue_gaps.append(gap)
            else:
                eigenvalue_gaps.append(0.0)
        
        # Topological invariant estimate
        topological_invariant = phase_mod_2pi / np.pi  # Normalized to [-1, 1]
        
        return BerryPhaseResult(
            total_berry_phase=total_phase,
            berry_phase_mod_2pi=phase_mod_2pi,
            winding_number=winding_number,
            phase_discontinuities=discontinuities,
            trajectory_length=traj_metrics['total_length'],
            mean_local_curvature=traj_metrics['mean_curvature'],
            topological_invariant=topological_invariant,
            cm_trajectory=cm_trajectory,
            eigenvalue_gaps=eigenvalue_gaps,
            raw_phases=berry_phases,
            epochs=epochs
        )
    
    def calculate_from_final_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Calculate Berry phase estimates from final checkpoint metrics history."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        history = ckpt.get('metrics_history', {})
        
        if not history:
            return {'error': 'No metrics history found'}
        
        # Extract relevant metrics
        topo_phase = history.get('topo_phase_state', [])
        cm_x = history.get('topo_R_cm_x', [])
        cm_y = history.get('topo_R_cm_y', [])
        delta = history.get('delta', [])
        kappa = history.get('kappa', [])
        
        # Estimate Berry phase from CM angular displacement
        berry_estimate = 0.0
        for i in range(1, len(cm_x)):
            if len(cm_y) > i:
                # Angular displacement on torus
                dx = cm_x[i] - cm_x[i-1] if i < len(cm_x) else 0
                dy = cm_y[i] - cm_y[i-1] if i < len(cm_y) else 0
                
                # Radius from origin
                r = np.sqrt(cm_x[i]**2 + cm_y[i]**2) + 1e-10
                
                # Angular displacement (dθ = (x*dy - y*dx) / r²)
                dtheta = (cm_x[i] * dy - cm_y[i] * dx) / (r**2)
                berry_estimate += dtheta
        
        # Detect phase transitions
        phase_transitions = []
        for i in range(1, len(topo_phase)):
            if len(topo_phase) > i:
                change = topo_phase[i] - topo_phase[i-1]
                if abs(change) > 0.3:
                    phase_transitions.append({
                        'epoch': i * 10,  # Approximate
                        'change': change
                    })
        
        return {
            'berry_phase_estimate': berry_estimate,
            'berry_phase_mod_2pi': berry_estimate % (2 * np.pi),
            'winding_estimate': int(round(berry_estimate / (2 * np.pi))),
            'phase_transitions': phase_transitions,
            'n_epochs': len(topo_phase),
            'final_topo_phase': topo_phase[-1] if topo_phase else 0,
            'final_cm_x': cm_x[-1] if cm_x else 0,
            'final_cm_y': cm_y[-1] if cm_y else 0,
            'final_delta': delta[-1] if delta else 0,
        }


def visualize_results(result: BerryPhaseResult, output_path: str = None):
    """Create visualization of Berry phase results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Berry phase accumulation
        ax1 = axes[0, 0]
        cumulative = np.cumsum(result.raw_phases)
        ax1.plot(result.epochs[1:], cumulative, 'b-', linewidth=1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=np.pi, color='red', linestyle='--', alpha=0.5, label='π')
        ax1.axhline(y=-np.pi, color='red', linestyle='--', alpha=0.5, label='-π')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cumulative Berry Phase (rad)')
        ax1.set_title('Berry Phase Accumulation')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Berry phase on unit circle
        ax2 = axes[0, 1]
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), 'gray', linestyle='--', alpha=0.5)
        
        angle = result.berry_phase_mod_2pi
        ax2.arrow(0, 0, 0.9*np.cos(angle), 0.9*np.sin(angle), 
                  head_width=0.1, head_length=0.05, fc='blue', ec='blue')
        ax2.plot(np.cos(angle), np.sin(angle), 'ro', markersize=10)
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-1.3, 1.3)
        ax2.set_aspect('equal')
        ax2.set_title(f'Final Phase: {result.berry_phase_mod_2pi:.4f} rad')
        ax2.set_xlabel('Re(e^{iγ})')
        ax2.set_ylabel('Im(e^{iγ})')
        ax2.grid(True, alpha=0.3)
        
        # 3. CM trajectory
        ax3 = axes[0, 2]
        if result.cm_trajectory:
            cm_x = [p[0] for p in result.cm_trajectory]
            cm_y = [p[1] for p in result.cm_trajectory]
            
            # Color by epoch
            colors = plt.cm.viridis(np.linspace(0, 1, len(cm_x)))
            for i in range(1, len(cm_x)):
                ax3.plot(cm_x[i-1:i+1], cm_y[i-1:i+1], color=colors[i], linewidth=1)
            
            ax3.scatter(cm_x[0], cm_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
            ax3.scatter(cm_x[-1], cm_y[-1], c='red', s=100, marker='*', label='End', zorder=5)
            ax3.legend()
        ax3.set_xlabel('CM_x (angular)')
        ax3.set_ylabel('CM_y (angular)')
        ax3.set_title('Center of Mass Trajectory in Fourier Space')
        ax3.grid(True, alpha=0.3)
        
        # 4. Eigenvalue gaps
        ax4 = axes[1, 0]
        ax4.plot(result.epochs, result.eigenvalue_gaps, 'g-', linewidth=1)
        ax4.fill_between(result.epochs, result.eigenvalue_gaps, alpha=0.3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gap Size')
        ax4.set_title('Eigenvalue Gap Evolution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Phase discontinuities
        ax5 = axes[1, 1]
        ax5.bar(range(len(result.raw_phases)), result.raw_phases, color='steelblue', alpha=0.7)
        ax5.axhline(y=np.pi/2, color='red', linestyle='--', alpha=0.5, label='π/2')
        ax5.axhline(y=-np.pi/2, color='red', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Phase Increment (rad)')
        ax5.set_title('Phase Increments per Step')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary text
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Interpretation
        if abs(result.berry_phase_mod_2pi) < 0.1:
            interp = "Trivial (γ ≈ 0)"
        elif abs(abs(result.berry_phase_mod_2pi) - np.pi) < 0.1:
            interp = "Non-trivial Z₂ (γ ≈ π)"
        else:
            interp = "Generic topological"
        
        summary = f"""
╔══════════════════════════════════════╗
║     BERRY PHASE ANALYSIS RESULTS     ║
╠══════════════════════════════════════╣
║                                      ║
║  Total Berry Phase:  {result.total_berry_phase:>10.4f} rad  ║
║  Phase (mod 2π):     {result.berry_phase_mod_2pi:>10.4f} rad  ║
║  Winding Number:     {result.winding_number:>10d}       ║
║                                      ║
║  Trajectory Length:  {result.trajectory_length:>10.4f}      ║
║  Mean Curvature:     {result.mean_local_curvature:>10.4f}      ║
║  Topo. Invariant:    {result.topological_invariant:>10.4f}      ║
║                                      ║
║  Discontinuities:    {len(result.phase_discontinuities):>10d}       ║
║  Checkpoints:        {len(result.epochs):>10d}       ║
║                                      ║
║  Interpretation: {interp:<18}  ║
╚══════════════════════════════════════╝
"""
        ax6.text(0.05, 0.5, summary, transform=ax6.transAxes,
                fontsize=9, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available for visualization")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Berry Phase from training checkpoints')
    parser.add_argument('checkpoint_dir', type=str, nargs='?', 
                        default='checkpoints_phase3',
                        help='Directory containing checkpoints')
    parser.add_argument('--output', '-o', type=str, default='berry_phase.png',
                        help='Output path for visualization')
    parser.add_argument('--final', type=str, default=None,
                        help='Alternative: path to final checkpoint with metrics history')
    parser.add_argument('--device', type=str, default='cpu', help='Device for computation')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    calculator = BerryPhaseCalculator(device=args.device)
    
    if args.final:
        # Use final checkpoint metrics
        print(f"Analyzing metrics from: {args.final}")
        result = calculator.calculate_from_final_checkpoint(args.final)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Full calculation from checkpoints
        print(f"Calculating Berry phase from: {args.checkpoint_dir}")
        result = calculator.calculate_berry_phase(args.checkpoint_dir)
        
        print("\n" + "="*50)
        print("BERRY PHASE CALCULATION RESULTS")
        print("="*50)
        print(f"Total Berry Phase:      {result.total_berry_phase:.6f} rad")
        print(f"Phase (mod 2π):         {result.berry_phase_mod_2pi:.6f} rad")
        print(f"                        ({np.degrees(result.berry_phase_mod_2pi):.2f}°)")
        print(f"Winding Number:         {result.winding_number}")
        print(f"Trajectory Length:      {result.trajectory_length:.6f}")
        print(f"Mean Curvature:         {result.mean_local_curvature:.6f}")
        print(f"Topological Invariant:  {result.topological_invariant:.6f}")
        print(f"Phase Discontinuities:  {len(result.phase_discontinuities)}")
        
        if result.phase_discontinuities:
            print("\nLargest Phase Jumps:")
            sorted_jumps = sorted(result.phase_discontinuities, 
                                  key=lambda x: abs(x[1]), reverse=True)[:5]
            for epoch, phase in sorted_jumps:
                print(f"  Epoch {epoch}: {phase:.4f} rad ({np.degrees(phase):.1f}°)")
        
        # Interpretation
        print("\n" + "-"*50)
        print("INTERPRETATION:")
        if abs(result.berry_phase_mod_2pi) < 0.1:
            print("  γ ≈ 0: Trivial topology (contractible loop)")
        elif abs(abs(result.berry_phase_mod_2pi) - np.pi) < 0.3:
            print("  γ ≈ π: Non-trivial Z₂ topology (Möbius-like)")
        elif abs(result.winding_number) > 0:
            print(f"  |winding| = {abs(result.winding_number)}: Non-zero Chern number")
        else:
            print("  Generic topological phase")
        
        if not args.no_viz:
            visualize_results(result, args.output)


if __name__ == "__main__":
    main()