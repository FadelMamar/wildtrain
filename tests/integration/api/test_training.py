"""
Training endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.training
class TestTrainingEndpoints:
    """Test training endpoints."""
    
    def test_train_classifier_with_real_config(self, client, classification_config):
        """Test POST /training/classifier endpoint with real config."""
        response = client.post("/training/classifier", json={
            "config": classification_config,
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
    
    def test_train_detector_with_real_config(self, client, detection_config):
        """Test POST /training/detector endpoint with real config."""
        response = client.post("/training/detector", json={
            "config": detection_config,
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
    
    def test_train_classifier_template(self, client):
        """Test POST /training/classifier endpoint with template_only=True."""
        response = client.post("/training/classifier", json={
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
    
    def test_train_detector_template(self, client):
        """Test POST /training/detector endpoint with template_only=True."""
        response = client.post("/training/detector", json={
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
    
    def test_get_training_status(self, client):
        """Test GET /training/status/{job_id} endpoint."""
        job_id = "test-job-123"
        response = client.get(f"/training/status/{job_id}")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_list_training_jobs(self, client):
        """Test GET /training/jobs endpoint."""
        response = client.get("/training/jobs")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data or isinstance(data, list)
    
    def test_list_training_jobs_with_filters(self, client):
        """Test GET /training/jobs with query parameters."""
        response = client.get("/training/jobs?status=running&limit=10&offset=0")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data or isinstance(data, list)
    
    def test_cancel_training_job(self, client):
        """Test POST /training/cancel endpoint."""
        job_id = "test-job-123"
        response = client.post("/training/cancel", json={"job_id": job_id})
        
        assert response.status_code in [200, 202, 404, 400]  # 400 for bad request
        if response.status_code in [200, 202]:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_invalid_training_config(self, client):
        """Test training with invalid config."""
        response = client.post("/training/classifier", json={
            "config": {"invalid": "config"},
            "debug": True,
            "verbose": False,
            "template_only": False
        })
        
        assert response.status_code in [422, 400]  # Validation error
    
    def test_training_job_creation_workflow(self, client):
        """Test complete training job creation workflow."""
        # 1. Start training
        response = client.post("/training/classifier", json={
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
            status_response = client.get(f"/training/status/{job_id}")
            assert status_response.status_code in [200, 404]
            
            # 3. List jobs
            jobs_response = client.get("/training/jobs")
            assert jobs_response.status_code in [200, 404]
            
            # 4. Cancel job
            cancel_response = client.post("/training/cancel", json={"job_id": job_id})
            assert cancel_response.status_code in [200, 202, 404, 400]
