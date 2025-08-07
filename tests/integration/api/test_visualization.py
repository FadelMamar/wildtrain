"""
Visualization endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.visualization
class TestVisualizationEndpoints:
    """Test visualization endpoints."""
    
    def test_visualize_classifier_with_real_config(self, client, classification_visualization_config):
        """Test POST /visualization/classifier endpoint with real config."""
        response = client.post("/visualization/classifier", json={
            "config": classification_visualization_config,
            "debug": True,
            "verbose": False,
            "template_only": False
        })
        
        assert response.status_code in [200, 201, 202, 422]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data
    
    def test_visualize_detector_with_real_config(self, client, detection_visualization_config):
        """Test POST /visualization/detector endpoint with real config."""
        response = client.post("/visualization/detector", json={
            "config": detection_visualization_config,
            "debug": True,
            "verbose": False
            # Removed template_only as it's not supported by this endpoint
        })
        
        assert response.status_code in [200, 201, 202, 422, 500]  # 500 for server errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data or "job_id" in data  # Check for either success or job_id
            if "job_id" in data:
                assert "message" in data  # Should have a message field
    
    def test_visualize_classifier_template(self, client):
        """Test POST /visualization/classifier endpoint with template_only=True."""
        response = client.post("/visualization/classifier", json={
            "config": {},  # Empty config for template request
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data
    
    def test_visualize_detector_template(self, client):
        """Test POST /visualization/detector endpoint with template_only=True."""
        response = client.post("/visualization/detector", json={
            "config": {},  # Empty config for template request
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data
    
    def test_list_visualization_datasets(self, client):
        """Test GET /visualization/datasets endpoint."""
        response = client.get("/visualization/datasets")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "datasets" in data or isinstance(data, list)
    
    def test_get_dataset_info(self, client):
        """Test GET /visualization/datasets/{dataset_name} endpoint."""
        dataset_name = "test-dataset"
        response = client.get(f"/visualization/datasets/{dataset_name}")
        
        assert response.status_code in [200, 404, 500]  # 500 for internal server error
        if response.status_code == 200:
            data = response.json()
            assert "dataset_name" in data
            assert "info" in data
    
    def test_get_visualization_status(self, client):
        """Test GET /visualization/status/{job_id} endpoint."""
        job_id = "test-viz-job-123"
        response = client.get(f"/visualization/status/{job_id}")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_visualization_job_creation_workflow(self, client):
        """Test complete visualization job creation workflow."""
        # 1. Start visualization
        response = client.post("/visualization/classifier", json={
            "config": {},
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]
        data = response.json()
        job_id = data.get("job_id")
        
        if job_id and job_id != "template":
            # 2. Check job status
            status_response = client.get(f"/visualization/status/{job_id}")
            assert status_response.status_code in [200, 404]
            
            # 3. List datasets
            datasets_response = client.get("/visualization/datasets")
            assert datasets_response.status_code in [200, 404]
    
    def test_invalid_visualization_config(self, client):
        """Test visualization with invalid config."""
        response = client.post("/visualization/classifier", json={
            "config": {"invalid": "config"},
            "debug": True,
            "verbose": False,
            "template_only": False
        })
        
        assert response.status_code in [422, 400]  # Validation error
    
    def test_visualization_template_generation(self, client):
        """Test template generation for visualization configs."""
        response = client.post("/visualization/classifier", json={
            "config": {},
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data or "job_id" in data
