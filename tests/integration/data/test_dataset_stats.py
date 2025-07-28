"""
Integration tests for dataset statistics computation functionality.

This test suite validates:
1. compute_dataset_stats function with mock dataset
2. Mean and standard deviation calculation
3. ClassificationDataModule integration
4. Batch processing functionality
5. Error handling for invalid datasets
"""

import pytest
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
import tempfile
import shutil

# Import WildTrain modules
from wildtrain.data.classification_datamodule import ClassificationDataModule, compute_dataset_stats


@pytest.mark.integration
@pytest.mark.data
class TestDatasetStats:
    """Test suite for dataset statistics computation functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self, mock_dataset_path):
        """Set up test data and configuration."""
        self.dataset_path = mock_dataset_path
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(prefix="wildtrain_dataset_stats_test_")
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compute_dataset_stats_standalone(self):
        """Test compute_dataset_stats function with mock dataset."""
        # Create a simple mock dataset
        class MockDataset:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                # Create random images
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        # Create mock dataset
        mock_dataset = MockDataset(num_samples=50)
        
        # Compute statistics
        mean, std = compute_dataset_stats(
            mock_dataset,
            batch_size=8,
            num_workers=0,
        )
        
        # Validate results
        assert isinstance(mean, torch.Tensor), "Mean should be a tensor"
        assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor"
        assert mean.shape == (3,), "Mean should have shape (3,) for RGB channels"
        assert std.shape == (3,), "Standard deviation should have shape (3,) for RGB channels"
        
        # Check that values are reasonable (standard normal distribution has mean ~0, std ~1)
        assert torch.all(mean >= -1) and torch.all(mean <= 1), "Mean values should be between -1 and 1"
        assert torch.all(std >= 0) and torch.all(std <= 2), "Standard deviation values should be between 0 and 2"
        
        print(f"Computed mean: {mean.tolist()}")
        print(f"Computed std: {std.tolist()}")
    
    def test_compute_dataset_stats_with_classification_datamodule(self):
        """Test compute_dataset_stats with ClassificationDataModule."""
        # Create data module
        data_module = ClassificationDataModule(
            root_data_directory=self.dataset_path,
            batch_size=4,
            transforms=None
        )
        
        # Setup the data module
        data_module.setup(stage="fit")
        
        # Compute statistics using the train dataset
        mean, std = compute_dataset_stats(
            data_module.train_dataset,
            batch_size=4,
            num_workers=0,
        )
        
        # Validate results
        assert isinstance(mean, torch.Tensor), "Mean should be a tensor"
        assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor"
        assert mean.shape == (3,), "Mean should have shape (3,) for RGB channels"
        assert std.shape == (3,), "Standard deviation should have shape (3,) for RGB channels"
        
        # Check that values are reasonable
        assert torch.all(mean >= 0) and torch.all(mean <= 1), "Mean values should be between 0 and 1"
        assert torch.all(std >= 0) and torch.all(std <= 1), "Standard deviation values should be between 0 and 1"
        
        print(f"Computed mean with ClassificationDataModule: {mean.tolist()}")
        print(f"Computed std with ClassificationDataModule: {std.tolist()}")
    
    def test_batch_processing_functionality(self):
        """Test batch processing functionality with different batch sizes."""
        # Create a simple mock dataset
        class MockDataset:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        mock_dataset = MockDataset(num_samples=50)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            mean, std = compute_dataset_stats(
                mock_dataset,
                batch_size=batch_size,
                num_workers=0,
            )
            
            # Validate results
            assert isinstance(mean, torch.Tensor), f"Mean should be a tensor for batch_size={batch_size}"
            assert isinstance(std, torch.Tensor), f"Standard deviation should be a tensor for batch_size={batch_size}"
            assert mean.shape == (3,), f"Mean should have shape (3,) for batch_size={batch_size}"
            assert std.shape == (3,), f"Standard deviation should have shape (3,) for batch_size={batch_size}"
            
            print(f"Batch size {batch_size}: mean={mean.tolist()}, std={std.tolist()}")
    
    def test_error_handling_invalid_dataset(self):
        """Test error handling for invalid datasets."""
        # Test with empty dataset
        class EmptyDataset:
            def __init__(self):
                self.images = torch.empty(0, 3, 224, 224)
                self.labels = torch.empty(0)
            
            def __len__(self):
                return 0
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        empty_dataset = EmptyDataset()
        
        # Should handle empty dataset gracefully
        try:
            mean, std = compute_dataset_stats(
                empty_dataset,
                batch_size=4,
                num_workers=0,
            )
            # If it doesn't raise an error, the result should be reasonable
            assert isinstance(mean, torch.Tensor), "Mean should be a tensor even for empty dataset"
            assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor even for empty dataset"
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass
        
        # Test with dataset that returns invalid data
        class InvalidDataset:
            def __init__(self):
                self.num_samples = 10
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Return invalid data (not a tensor)
                return "invalid", "invalid"
        
        invalid_dataset = InvalidDataset()
        
        # Should handle invalid data gracefully
        try:
            mean, std = compute_dataset_stats(
                invalid_dataset,
                batch_size=4,
                num_workers=0,
            )
            # If it doesn't raise an error, the result should be reasonable
            assert isinstance(mean, torch.Tensor), "Mean should be a tensor even for invalid dataset"
            assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor even for invalid dataset"
        except (TypeError, RuntimeError):
            # This is also acceptable behavior
            pass
    
    def test_different_image_sizes(self):
        """Test compute_dataset_stats with different image sizes."""
        # Create datasets with different image sizes
        image_sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
        
        for height, width in image_sizes:
            class MockDataset:
                def __init__(self, num_samples=20, height=224, width=224):
                    self.num_samples = num_samples
                    self.images = torch.randn(num_samples, 3, height, width)
                    self.labels = torch.randint(0, 3, (num_samples,))
                
                def __len__(self):
                    return self.num_samples
                
                def __getitem__(self, idx):
                    return self.images[idx], self.labels[idx]
            
            mock_dataset = MockDataset(num_samples=20, height=height, width=width)
            
            # Compute statistics
            mean, std = compute_dataset_stats(
                mock_dataset,
                batch_size=4,
                num_workers=0,
            )
            
            # Validate results
            assert isinstance(mean, torch.Tensor), f"Mean should be a tensor for size {height}x{width}"
            assert isinstance(std, torch.Tensor), f"Standard deviation should be a tensor for size {height}x{width}"
            assert mean.shape == (3,), f"Mean should have shape (3,) for size {height}x{width}"
            assert std.shape == (3,), f"Standard deviation should have shape (3,) for size {height}x{width}"
            
            print(f"Size {height}x{width}: mean={mean.tolist()}, std={std.tolist()}")
    
    def test_consistency_across_runs(self):
        """Test that statistics are consistent across multiple runs."""
        # Create a mock dataset
        class MockDataset:
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        mock_dataset = MockDataset(num_samples=50)
        
        # Compute statistics multiple times
        results = []
        for i in range(3):
            mean, std = compute_dataset_stats(
                mock_dataset,
                batch_size=8,
                num_workers=0,
            )
            results.append((mean, std))
        
        # Check that results are consistent
        for i in range(1, len(results)):
            mean_1, std_1 = results[i-1]
            mean_2, std_2 = results[i]
            
            # Results should be identical (deterministic)
            assert torch.allclose(mean_1, mean_2), f"Mean should be consistent across runs {i-1} and {i}"
            assert torch.allclose(std_1, std_2), f"Standard deviation should be consistent across runs {i-1} and {i}"
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create a larger mock dataset
        class LargeMockDataset:
            def __init__(self, num_samples=1000):
                self.num_samples = num_samples
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        large_dataset = LargeMockDataset(num_samples=500)
        
        # Time the computation
        import time
        start_time = time.time()
        
        mean, std = compute_dataset_stats(
            large_dataset,
            batch_size=32,
            num_workers=0,
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Validate results
        assert isinstance(mean, torch.Tensor), "Mean should be a tensor"
        assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor"
        assert mean.shape == (3,), "Mean should have shape (3,)"
        assert std.shape == (3,), "Standard deviation should have shape (3,)"
        
        # Check performance (should complete within reasonable time)
        assert processing_time < 30.0, f"Processing should be fast (took {processing_time:.2f}s)"
        
        print(f"Processed {len(large_dataset)} samples in {processing_time:.2f}s")
        print(f"Mean: {mean.tolist()}")
        print(f"Std: {std.tolist()}")
    
    def test_transforms_impact(self):
        """Test how transforms affect the computed statistics."""
        # Create a mock dataset
        class MockDataset:
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        mock_dataset = MockDataset(num_samples=50)
        
        # Compute statistics without transforms
        mean_no_transform, std_no_transform = compute_dataset_stats(
            mock_dataset,
            batch_size=8,
            num_workers=0,
        )
        
        # Create dataset with transforms
        class TransformedDataset:
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset
                self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                image, label = self.base_dataset[idx]
                image = self.transform(image)
                return image, label
        
        transformed_dataset = TransformedDataset(mock_dataset)
        
        # Compute statistics with transforms
        mean_with_transform, std_with_transform = compute_dataset_stats(
            transformed_dataset,
            batch_size=8,
            num_workers=0,
        )
        
        # Validate that transforms affect the statistics
        assert not torch.allclose(mean_no_transform, mean_with_transform), \
            "Transforms should affect the mean"
        assert not torch.allclose(std_no_transform, std_with_transform), \
            "Transforms should affect the standard deviation"
        
        print(f"Without transforms: mean={mean_no_transform.tolist()}, std={std_no_transform.tolist()}")
        print(f"With transforms: mean={mean_with_transform.tolist()}, std={std_with_transform.tolist()}")
    
    def test_edge_cases(self):
        """Test edge cases for dataset statistics computation."""
        # Test with single sample
        class SingleSampleDataset:
            def __init__(self):
                self.images = torch.randn(1, 3, 224, 224)
                self.labels = torch.randint(0, 3, (1,))
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        single_dataset = SingleSampleDataset()
        
        mean, std = compute_dataset_stats(
            single_dataset,
            batch_size=1,
            num_workers=0,
        )
        
        # Validate results
        assert isinstance(mean, torch.Tensor), "Mean should be a tensor for single sample"
        assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor for single sample"
        assert mean.shape == (3,), "Mean should have shape (3,) for single sample"
        assert std.shape == (3,), "Standard deviation should have shape (3,) for single sample"
        
        # Test with very small batch size
        class SmallDataset:
            def __init__(self, num_samples=10):
                self.num_samples = num_samples
                self.images = torch.randn(num_samples, 3, 224, 224)
                self.labels = torch.randint(0, 3, (num_samples,))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        small_dataset = SmallDataset(num_samples=10)
        
        mean, std = compute_dataset_stats(
            small_dataset,
            batch_size=1,  # Very small batch size
            num_workers=0,
        )
        
        # Validate results
        assert isinstance(mean, torch.Tensor), "Mean should be a tensor for small batch size"
        assert isinstance(std, torch.Tensor), "Standard deviation should be a tensor for small batch size"
        assert mean.shape == (3,), "Mean should have shape (3,) for small batch size"
        assert std.shape == (3,), "Standard deviation should have shape (3,) for small batch size" 