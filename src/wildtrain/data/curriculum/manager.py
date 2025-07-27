"""
Core Curriculum Learning Manager.

This module provides a unified interface for curriculum learning strategies.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import lightning as L


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # General settings
    enabled: bool = False
    type: str = "difficulty"  # "difficulty", "multiscale", "both"
    
    # Difficulty-based curriculum
    difficulty_strategy: str = "linear"  # "linear", "exponential", "step", "random"
    start_difficulty: float = 0.0
    end_difficulty: float = 1.0
    warmup_epochs: int = 0
    
    # Multi-scale curriculum
    multiscale_enabled: bool = False
    base_size: int = 416
    scale_range: Tuple[float, float] = (0.5, 2.0)
    num_scales: int = 5
    scale_strategy: str = "curriculum"  # "curriculum", "random", "cyclic"
    cycle_length: int = 5
    
    # Logging
    log_frequency: int = 10


class DifficultyStrategy(ABC):
    """Abstract base class for difficulty progression strategies."""
    
    @abstractmethod
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        """Return difficulty level (0.0 = easiest, 1.0 = hardest) for given epoch."""
        pass


class LinearDifficultyStrategy(DifficultyStrategy):
    """Linear progression from easy to hard."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        effective_epoch = max(0, epoch - config.warmup_epochs)
        effective_max_epochs = max_epochs - config.warmup_epochs
        
        if effective_max_epochs <= 0:
            return config.end_difficulty
            
        progress = min(1.0, effective_epoch / effective_max_epochs)
        return config.start_difficulty + progress * (config.end_difficulty - config.start_difficulty)


class ExponentialDifficultyStrategy(DifficultyStrategy):
    """Exponential progression from easy to hard."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        effective_epoch = max(0, epoch - config.warmup_epochs)
        effective_max_epochs = max_epochs - config.warmup_epochs
        
        if effective_max_epochs <= 0:
            return config.end_difficulty
            
        progress = min(1.0, effective_epoch / effective_max_epochs)
        return 1.0 - np.exp(-3.0 * progress)  # Growth rate of 3.0


class RandomDifficultyStrategy(DifficultyStrategy):
    """No curriculum - random sampling."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        return config.end_difficulty  # Always max difficulty


class ScaleStrategy(ABC):
    """Abstract base class for multi-scale training strategies."""
    
    @abstractmethod
    def get_scale(self, epoch: int, max_epochs: int, config: CurriculumConfig, available_scales: List[float]) -> float:
        """Return the scale to use for the given epoch."""
        pass


class CurriculumScaleStrategy(ScaleStrategy):
    """Scale progression from small to large (curriculum)."""
    
    def get_scale(self, epoch: int, max_epochs: int, config: CurriculumConfig, available_scales: List[float]) -> float:
        effective_epoch = max(0, epoch - config.warmup_epochs)
        effective_max_epochs = max_epochs - config.warmup_epochs
        
        if effective_max_epochs <= 0:
            return available_scales[-1]
            
        progress = min(1.0, effective_epoch / effective_max_epochs)
        scale_idx = int(progress * (len(available_scales) - 1))
        return available_scales[scale_idx]


class RandomScaleStrategy(ScaleStrategy):
    """Random scale selection."""
    
    def get_scale(self, epoch: int, max_epochs: int, config: CurriculumConfig, available_scales: List[float]) -> float:
        return random.choice(available_scales)


class CyclicScaleStrategy(ScaleStrategy):
    """Cyclic progression through scales."""
    
    def get_scale(self, epoch: int, max_epochs: int, config: CurriculumConfig, available_scales: List[float]) -> float:
        scale_idx = (epoch // config.cycle_length) % len(available_scales)
        return available_scales[scale_idx]


class CurriculumManager:
    """Unified manager for curriculum learning strategies."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_epoch = 0
        self.current_difficulty = 0.0
        self.current_scale = 1.0
        self.max_epochs = 100
        
        # Initialize strategies
        self._init_strategies()
        
        # Generate scale pyramid
        self.available_scales = self._generate_scales()
        
    def _init_strategies(self):
        """Initialize difficulty and scale strategies."""
        # Difficulty strategies
        self.difficulty_strategies = {
            "linear": LinearDifficultyStrategy(),
            "exponential": ExponentialDifficultyStrategy(),
            "random": RandomDifficultyStrategy(),
        }
        
        # Scale strategies
        self.scale_strategies = {
            "curriculum": CurriculumScaleStrategy(),
            "random": RandomScaleStrategy(),
            "cyclic": CyclicScaleStrategy(),
        }
        
    def _generate_scales(self) -> List[float]:
        """Generate the scale pyramid based on configuration."""
        if not self.config.multiscale_enabled:
            return [1.0]
            
        min_scale, max_scale = self.config.scale_range
        
        if self.config.num_scales == 1:
            return [1.0]
        
        # Generate scales logarithmically for better coverage
        log_min = math.log(min_scale)
        log_max = math.log(max_scale)
        log_scales = np.linspace(log_min, log_max, self.config.num_scales)
        scales = [math.exp(log_scale) for log_scale in log_scales]
        
        return scales
    
    def set_max_epochs(self, max_epochs: int):
        """Set the maximum number of epochs for curriculum planning."""
        self.max_epochs = max_epochs
    
    def update_epoch(self, epoch: int) -> Dict[str, Any]:
        """Update curriculum state for new epoch."""
        self.current_epoch = epoch
        
        # Update difficulty if enabled
        if self.config.enabled and self.config.type in ["difficulty", "both"]:
            strategy = self.difficulty_strategies.get(self.config.difficulty_strategy)
            if strategy:
                self.current_difficulty = strategy.get_difficulty(
                    epoch, self.max_epochs, self.config
                )
        
        # Update scale if enabled
        if self.config.multiscale_enabled and self.config.type in ["multiscale", "both"]:
            strategy = self.scale_strategies.get(self.config.scale_strategy)
            if strategy:
                self.current_scale = strategy.get_scale(
                    epoch, self.max_epochs, self.config, self.available_scales
                )
        
        return {
            'epoch': epoch,
            'difficulty': self.current_difficulty,
            'scale': self.current_scale,
            'available_scales': self.available_scales.copy()
        }
    
    def should_include_sample(self, sample_difficulty: float) -> bool:
        """Check if sample should be included based on current curriculum."""
        if not self.config.enabled or self.config.type not in ["difficulty", "both"]:
            return True
            
        # Include samples that are not harder than current difficulty level
        return sample_difficulty <= self.current_difficulty + 0.1  # Small buffer
    
    def get_current_scale(self) -> float:
        """Get current scale for multi-scale training."""
        return self.current_scale
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def get_state(self) -> Dict[str, Any]:
        """Get current curriculum state."""
        return {
            'epoch': self.current_epoch,
            'difficulty': self.current_difficulty,
            'scale': self.current_scale,
            'available_scales': self.available_scales.copy(),
            'config': self.config
        } 