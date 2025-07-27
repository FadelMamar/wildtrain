"""
Curriculum Learning Callback for PyTorch Lightning.

This module provides a callback that automatically manages curriculum learning
during training, including epoch updates and logging.
"""

from typing import Optional, Dict, Any
import lightning as L
from .mixins import CurriculumDataModuleMixin


class CurriculumCallback(L.Callback):
    """
    PyTorch Lightning callback for managing curriculum learning.
    
    This callback automatically:
    1. Updates curriculum state at the start of each epoch
    2. Logs curriculum metrics to the trainer logger
    3. Handles curriculum setup when training begins
    
    Usage:
        # In your trainer
        if datamodule.is_curriculum_enabled():
            callbacks.append(CurriculumCallback(datamodule))
    """
    
    def __init__(self, datamodule: CurriculumDataModuleMixin):
        """
        Initialize curriculum callback.
        
        Args:
            datamodule: Data module that implements CurriculumDataModuleMixin
        """
        self.datamodule = datamodule
        self.log_frequency = 1  # Log every epoch by default
        
        # Get log frequency from config if available
        if hasattr(datamodule, 'curriculum_config') and datamodule.curriculum_config:
            self.log_frequency = datamodule.curriculum_config.log_frequency
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Setup curriculum when training starts."""
        if self.datamodule.is_curriculum_enabled():
            # Setup curriculum with max epochs
            self.datamodule.setup_curriculum(trainer.max_epochs)
            
            # Log initial curriculum state
            if trainer.logger:
                initial_state = self.datamodule.get_curriculum_state()
                if initial_state:
                    trainer.logger.log_metrics({
                        'curriculum/initial_difficulty': initial_state.get('difficulty', 0.0),
                        'curriculum/initial_scale': initial_state.get('scale', 1.0),
                        'curriculum/available_scales': len(initial_state.get('available_scales', [1.0])),
                    }, step=0)
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update curriculum at the start of each training epoch."""
        if not self.datamodule.is_curriculum_enabled():
            return
            
        # Update curriculum state
        curriculum_state = self.datamodule.update_curriculum_epoch(trainer.current_epoch)
        
        if curriculum_state and trainer.logger:
            # Log curriculum metrics
            self._log_curriculum_metrics(trainer, curriculum_state)
    
    def _log_curriculum_metrics(self, trainer: L.Trainer, curriculum_state: Dict[str, Any]) -> None:
        """Log curriculum metrics to trainer logger."""
        # Always log basic metrics
        metrics = {
            'curriculum/difficulty': curriculum_state.get('difficulty', 0.0),
            'curriculum/scale': curriculum_state.get('scale', 1.0),
            'curriculum/epoch': curriculum_state.get('epoch', 0),
        }
        
        # Log detailed metrics based on frequency
        if trainer.current_epoch % self.log_frequency == 0:
            # Add detailed metrics
            available_scales = curriculum_state.get('available_scales', [])
            if available_scales:
                metrics.update({
                    'curriculum/num_scales': len(available_scales),
                    'curriculum/min_scale': min(available_scales),
                    'curriculum/max_scale': max(available_scales),
                })
            
            # Add curriculum type information
            if hasattr(self.datamodule, 'curriculum_config') and self.datamodule.curriculum_config:
                config = self.datamodule.curriculum_config
                metrics.update({
                    'curriculum/type': config.type,
                    'curriculum/difficulty_strategy': config.difficulty_strategy,
                    'curriculum/scale_strategy': config.scale_strategy,
                })
        
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Optional: Log curriculum summary at epoch end."""
        if not self.datamodule.is_curriculum_enabled():
            return
            
        # Log epoch summary if needed
        if trainer.logger and trainer.current_epoch % self.log_frequency == 0:
            current_state = self.datamodule.get_curriculum_state()
            if current_state:
                trainer.logger.log_metrics({
                    'curriculum/epoch_end_difficulty': current_state.get('difficulty', 0.0),
                    'curriculum/epoch_end_scale': current_state.get('scale', 1.0),
                }, step=trainer.global_step) 