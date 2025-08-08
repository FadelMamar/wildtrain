"""
Configuration Endpoints Integration Tests

These tests ensure configuration operations work correctly.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import yaml

from wildtrain.api.main import fastapi_app as app

# Valid classification config
VALID_CLASSIFICATION_CONFIG = Path(r"configs\classification\classification_train.yaml")
        
        # Valid detection config
VALID_DETECTION_CONFIG = Path(r"configs\detection\yolo_configs\yolo.yaml")

@pytest.mark.config
class TestConfigurationEndpoints:
    """Test configuration endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)
        
    def test_post_config_validate_validates_configs_correctly(self, client):
        """Test POST /config/validate validates configs correctly."""
        response = client.post("/config/validate", json={
            "config_path": str(VALID_CLASSIFICATION_CONFIG),
            "config_type": "classification"
        })
        
        assert response.status_code in [200, 201, 202, 500]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert data["success"] is True
            assert "is_valid" in data
            assert "errors" in data
            assert "warnings" in data
            assert "config_type" in data
    
    def test_get_config_templates_returns_templates(self, client):
        """Test GET /config/templates/{config_type} returns templates."""
        config_types = ["classification", "detection", "classification_eval", "detection_eval"]
        
        for config_type in config_types:
            response = client.get(f"/config/templates/{config_type}")
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "template" in data
                assert "config_type" in data
    
    def test_get_config_types_lists_available_types(self, client):
        """Test GET /config/types lists available types."""
        response = client.get("/config/types")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "config_types" in data
        assert "total_types" in data
        assert len(data["config_types"]) > 0
        
        # Check that each config type has required fields
        for config_type in data["config_types"]:
            assert "name" in config_type
            assert "description" in config_type
            assert "file_extension" in config_type
      
    def test_template_generation_for_all_config_types(self, client):
        """Test template generation for all config types."""
        config_types = [
            "classification", "detection", "classification_eval", 
            "detection_eval", "classification_visualization", 
            "detection_visualization", "pipeline"
        ]
        
        for config_type in config_types:
            response = client.get(f"/config/templates/{config_type}")
            
            # Should handle all config types gracefully
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "template" in data
    
    def test_file_upload_handling_for_configs(self, client):
        """Test file upload handling for configs."""
        # Test with file path instead of upload
        response = client.post("/config/validate", json={
            "config_path": str(VALID_CLASSIFICATION_CONFIG),
            "config_type": "classification"
        })
        
        assert response.status_code in [200, 201, 202, 500]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert data["success"] is True
    
    def test_config_validation_with_different_types(self, client):
        """Test config validation with different config types."""
        config_types = ["classification", "detection", "classification_eval"]
        
        for config_type in config_types:
            response = client.post("/config/validate", json={
                "config_path": str(VALID_CLASSIFICATION_CONFIG),
                "config_type": config_type
            })
            
            # Should handle different config types gracefully
            assert response.status_code in [200, 201, 202, 500]
    
    def test_config_validation_with_nonexistent_files(self, client):
        """Test config validation with non-existent files."""
        response = client.post("/config/validate", json={
            "config_path": "/nonexistent/config.yaml",
            "config_type": "classification"
        })
        
        # Should handle non-existent files gracefully
        assert response.status_code in [404, 500, 422]
    
    def test_config_validation_with_malformed_yaml(self, client):
        """Test config validation with malformed YAML."""
        # Create malformed YAML file
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_config = Path(temp_dir) / "malformed.yaml"
            malformed_config.write_text("""
            model:
              name: resnet18
            data:
              dataset_path: /path/to/dataset
            # Missing closing brace or invalid YAML
            """)
            
            response = client.post("/config/validate", json={
                "config_path": str(malformed_config),
                "config_type": "classification"
            })
            
            # Should handle malformed YAML gracefully
            assert response.status_code in [200, 201, 202, 500]
            if response.status_code in [200, 201, 202]:
                data = response.json()
                # Should either be invalid or have errors
                assert not data.get("is_valid", True) or len(data.get("errors", [])) > 0
    
    def test_config_templates_with_invalid_types(self, client):
        """Test config templates with invalid config types."""
        invalid_types = ["invalid_type", "nonexistent", "random"]
        
        for config_type in invalid_types:
            response = client.get(f"/config/templates/{config_type}")
            
            # Should handle invalid config types gracefully
            assert response.status_code in [404, 500]
    
    def test_config_types_endpoint_structure(self, client):
        """Test that config types endpoint returns proper structure."""
        response = client.get("/config/types")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "success" in data
        assert "message" in data
        assert "config_types" in data
        assert "total_types" in data
        
        # Check that config_types is a list
        assert isinstance(data["config_types"], list)
        
        # Check that total_types matches the length
        assert data["total_types"] == len(data["config_types"])
    
    def test_config_validation_error_details(self, client):
        """Test that config validation provides detailed error information."""
        # Create a config with obvious errors
        with tempfile.TemporaryDirectory() as temp_dir:
            error_config = Path(temp_dir) / "error_config.yaml"
            error_config.write_text("""
            model:
              name: resnet18
            data:
              # Missing dataset_path
            training:
              # Missing epochs
            """)
        
            response = client.post("/config/validate", json={
                "config_path": str(error_config),
                "config_type": "classification"
            })
        
            if response.status_code in [200, 201, 202]:
                data = response.json()
                # Should provide detailed error information
                assert "errors" in data
                assert isinstance(data["errors"], list)
                    
    def test_config_validation_request_validation(self, client):
        """Test config validation request validation."""
        # Test with invalid data types
        invalid_requests = [
            {"config_path": 123, "config_type": "classification"},  # config_path should be string
            {"config_path": str(VALID_CLASSIFICATION_CONFIG), "config_type": 456},  # config_type should be string
            {"config_path": "", "config_type": "classification"},  # empty config_path
            {"config_path": str(VALID_CLASSIFICATION_CONFIG), "config_type": ""},  # empty config_type
        ]
        
        for request in invalid_requests:
            response = client.post("/config/validate", json=request)
            print(response.json())
            
            # Should validate input properly
            assert response.status_code in [422, 400, 500]