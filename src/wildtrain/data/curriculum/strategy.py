import lightning as L
from pytorch_lightning.callbacks import Callback
import numpy as np

from typing import List, Dict, Tuple, Optional, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

# ================================
# CURRICULUM STRATEGY INTERFACES
# ================================

class CurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies."""
    
    @abstractmethod
    def get_difficulty(self, epoch: int, max_epochs: int) -> float:
        """Return difficulty level (0.0 = easiest, 1.0 = hardest) for given epoch."""
        pass
    
    @abstractmethod
    def should_include_sample(self, sample_difficulty: float, current_difficulty: float) -> bool:
        """Determine if a sample should be included given current curriculum difficulty."""
        pass


class ScaleStrategy(ABC):
    """Abstract base class for multi-scale training strategies."""
    
    @abstractmethod
    def get_scale(self, epoch: int, max_epochs: int, available_scales: List[float]) -> float:
        """Return the scale to use for the given epoch."""
        pass


# ================================
# CONCRETE CURRICULUM STRATEGIES
# ================================

class LinearCurriculum(CurriculumStrategy):
    """Linear progression from easy to hard."""
    
    def __init__(self, start_difficulty: float = 0.0, end_difficulty: float = 1.0):
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
    
    def get_difficulty(self, epoch: int, max_epochs: int) -> float:
        progress = min(1.0, epoch / max_epochs)
        return self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)
    
    def should_include_sample(self, sample_difficulty: float, current_difficulty: float) -> bool:
        # Include samples that are not harder than current difficulty level
        return sample_difficulty <= current_difficulty + 0.1  # Small buffer


class ExponentialCurriculum(CurriculumStrategy):
    """Exponential progression from easy to hard."""
    
    def __init__(self, growth_rate: float = 3.0):
        self.growth_rate = growth_rate
    
    def get_difficulty(self, epoch: int, max_epochs: int) -> float:
        progress = min(1.0, epoch / max_epochs)
        return 1.0 - np.exp(-self.growth_rate * progress)
    
    def should_include_sample(self, sample_difficulty: float, current_difficulty: float) -> bool:
        return sample_difficulty <= current_difficulty + 0.15


class StepCurriculum(CurriculumStrategy):
    """Step-wise curriculum with discrete difficulty levels."""
    
    def __init__(self, steps: List[Tuple[float, float]]):
        """
        Args:
            steps: List of (epoch_fraction, difficulty) tuples
            E.g., [(0.3, 0.2), (0.6, 0.5), (1.0, 1.0)]
        """
        self.steps = sorted(steps, key=lambda x: x[0])
    
    def get_difficulty(self, epoch: int, max_epochs: int) -> float:
        progress = epoch / max_epochs
        
        for epoch_fraction, difficulty in self.steps:
            if progress <= epoch_fraction:
                return difficulty
        
        return self.steps[-1][1]  # Return max difficulty
    
    def should_include_sample(self, sample_difficulty: float, current_difficulty: float) -> bool:
        return sample_difficulty <= current_difficulty + 0.1


class RandomCurriculum(CurriculumStrategy):
    """No curriculum - random sampling."""
    
    def get_difficulty(self, epoch: int, max_epochs: int) -> float:
        return 1.0  # Always max difficulty
    
    def should_include_sample(self, sample_difficulty: float, current_difficulty: float) -> bool:
        return True  # Include all samples


# ================================
# SCALE STRATEGIES
# ================================

class CurriculumScaleStrategy(ScaleStrategy):
    """Scale progression from small to large (curriculum)."""
    
    def get_scale(self, epoch: int, max_epochs: int, available_scales: List[float]) -> float:
        progress = min(1.0, epoch / max_epochs)
        scale_idx = int(progress * (len(available_scales) - 1))
        return available_scales[scale_idx]


class RandomScaleStrategy(ScaleStrategy):
    """Random scale selection."""
    
    def get_scale(self, epoch: int, max_epochs: int, available_scales: List[float]) -> float:
        return np.random.choice(available_scales)


class CyclicScaleStrategy(ScaleStrategy):
    """Cyclic progression through scales."""
    
    def __init__(self, cycle_length: int = 5):
        self.cycle_length = cycle_length
    
    def get_scale(self, epoch: int, max_epochs: int, available_scales: List[float]) -> float:
        scale_idx = (epoch // self.cycle_length) % len(available_scales)
        return available_scales[scale_idx]


# ================================
# CURRICULUM MANAGER
# ================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    curriculum_strategy: CurriculumStrategy
    scale_strategy: ScaleStrategy
    max_epochs: int = 100
    warmup_epochs: int = 0
    log_frequency: int = 10


class CurriculumManager:
    """Manages curriculum learning state and progression."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_epoch = 0
        self.current_difficulty = 0.0
        self.current_scale = 1.0
        self.available_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
    def update_epoch(self, epoch: int) -> Dict[str, float]:
        """Update curriculum state for new epoch."""
        self.current_epoch = epoch
        
        # Apply warmup period
        effective_epoch = max(0, epoch - self.config.warmup_epochs)
        effective_max_epochs = self.config.max_epochs - self.config.warmup_epochs
        
        # Update difficulty
        if effective_max_epochs > 0:
            self.current_difficulty = self.config.curriculum_strategy.get_difficulty(
                effective_epoch, effective_max_epochs
            )
        else:
            self.current_difficulty = 1.0
        
        # Update scale
        self.current_scale = self.config.scale_strategy.get_scale(
            effective_epoch, effective_max_epochs, self.available_scales
        )
        
        return {
            'epoch': epoch,
            'difficulty': self.current_difficulty,
            'scale': self.current_scale,
            'effective_epoch': effective_epoch
        }
    
    def should_include_sample(self, sample_difficulty: float) -> bool:
        """Check if sample should be included based on current curriculum."""
        return self.config.curriculum_strategy.should_include_sample(
            sample_difficulty, self.current_difficulty
        )
    
    def get_current_scale(self) -> float:
        """Get current scale for multi-scale training."""
        return self.current_scale
    
    def get_state(self) -> Dict[str, Any]:
        """Get current curriculum state."""
        return {
            'epoch': self.current_epoch,
            'difficulty': self.current_difficulty,
            'scale': self.current_scale,
            'available_scales': self.available_scales
        }


# ================================
# CURRICULUM CALLBACK
# ================================

class CurriculumCallback(Callback):
    """PyTorch Lightning callback for managing curriculum learning."""
    
    def __init__(self, curriculum_manager: CurriculumManager):
        self.curriculum_manager = curriculum_manager
        
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update curriculum at the start of each training epoch."""
        # Update curriculum state
        curriculum_state = self.curriculum_manager.update_epoch(trainer.current_epoch)
        
        # Update datasets that support curriculum
        self._update_dataloaders(trainer, curriculum_state)
        
        # Log curriculum progress
        if trainer.current_epoch % self.curriculum_manager.config.log_frequency == 0:
            self._log_curriculum_state(trainer, curriculum_state)
    
    def _update_dataloaders(self, trainer: L.Trainer, curriculum_state: Dict[str, float]):
        """Update all dataloaders with curriculum state."""
        # Update training dataloader
        if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader:
            self._update_single_dataloader(trainer.train_dataloader, curriculum_state)
        
        # You can also update validation dataloader if needed
        # if hasattr(trainer, 'val_dataloaders') and trainer.val_dataloaders:
        #     for val_loader in trainer.val_dataloaders:
        #         self._update_single_dataloader(val_loader, curriculum_state)
    
    def _update_single_dataloader(self, dataloader, curriculum_state: Dict[str, float]):
        """Update a single dataloader with curriculum state."""
        # Method 1: Direct dataset update
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'update_curriculum'):
            dataloader.dataset.update_curriculum(curriculum_state)
        
        # Method 2: Sampler update
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'update_curriculum'):
            dataloader.sampler.update_curriculum(curriculum_state)
        
        # Method 3: Dataloader wrapper update
        if hasattr(dataloader, 'update_curriculum'):
            dataloader.update_curriculum(curriculum_state)
    
    def _log_curriculum_state(self, trainer: L.Trainer, curriculum_state: Dict[str, float]):
        """Log curriculum state to trainer logger."""
        if trainer.logger:
            log_dict = {
                'curriculum/difficulty': curriculum_state['difficulty'],
                'curriculum/scale': curriculum_state['scale'],
                'curriculum/epoch': curriculum_state['epoch']
            }
            trainer.logger.log_metrics(log_dict, step=trainer.global_step)



if __name__ == "__main__":
    from torchvision import transforms
    from .data import CurriculumDataModule
    
    # Example data
    train_paths = ["train1.jpg", "train2.jpg"] * 100
    train_anns = [
        {"objects": [{"bbox": [100, 100, 50, 50], "class_id": 0}]},
        {"objects": [{"bbox": [200, 200, 30, 30], "class_id": 1}]}
    ] * 100
    
    val_paths = ["val1.jpg", "val2.jpg"] * 20
    val_anns = [
        {"objects": [{"bbox": [110, 110, 45, 45], "class_id": 0}]},
        {"objects": [{"bbox": [190, 190, 35, 35], "class_id": 1}]}
    ] * 20
    
    # 1. Create curriculum strategies
    curriculum_strategy = LinearCurriculum(start_difficulty=0.0, end_difficulty=1.0)
    scale_strategy = CurriculumScaleStrategy()
    
    # 2. Create curriculum configuration
    curriculum_config = CurriculumConfig(
        curriculum_strategy=curriculum_strategy,
        scale_strategy=scale_strategy,
        max_epochs=100,
        warmup_epochs=5,
        log_frequency=10
    )
    
    # 3. Create curriculum manager
    curriculum_manager = CurriculumManager(curriculum_config)
    
    # 4. Create curriculum callback
    curriculum_callback = CurriculumCallback(curriculum_manager)
    
    # 5. Create clean data module (no curriculum logic inside)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_module = CurriculumDataModule(
        train_image_paths=train_paths,
        train_annotations=train_anns,
        val_image_paths=val_paths,
        val_annotations=val_anns,
        batch_size=32,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # 6. Create trainer with curriculum callback
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[curriculum_callback],  # This handles all curriculum logic
        accelerator='auto',
        devices='auto'
    )
    
    print("Clean separated curriculum architecture:")
    print("✅ CurriculumStrategy - defines how difficulty progresses")
    print("✅ ScaleStrategy - defines how scales change")
    print("✅ CurriculumManager - manages state")
    print("✅ CurriculumCallback - integrates with Lightning")
    print("✅ CurriculumAwareDataset - responds to curriculum updates")
    print("✅ CleanMultiScaleDataModule - pure data loading")
    
    # Test curriculum progression
    print(f"\nCurriculum progression example:")
    for epoch in [0, 25, 50, 75, 99]:
        state = curriculum_manager.update_epoch(epoch)
        print(f"Epoch {epoch:2d}: difficulty={state['difficulty']:.3f}, scale={state['scale']:.3f}")