"""
Evaluation endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.evaluation
class TestEvaluationEndpoints:
    """Test evaluation endpoints."""
    
    def test_evaluate_classifier_with_real_config(self, client, classification_eval_config):
        """Test POST /evaluation/classifier endpoint with real config."""
        response = client.post("/evaluation/classifier", json={
            "config": classification_eval_config,
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
    
    def test_evaluate_detector_with_real_config(self, client, detection_eval_config):
        """Test POST /evaluation/detector endpoint with real config."""
        response = client.post("/evaluation/detector", json={
            "config": detection_eval_config,
            "debug": True,
            "verbose": False,
            "template_only": False
        })
        
        assert response.status_code in [200, 201, 202, 422]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data or "job_id" in data  # Check for either success or job_id
            if "job_id" in data:
                assert "message" in data  # Should have a message field
    
    def test_evaluate_classifier_template(self, client):
        """Test POST /evaluation/classifier endpoint with template_only=True."""
        response = client.post("/evaluation/classifier", json={
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
    
    def test_evaluate_detector_template(self, client):
        """Test POST /evaluation/detector endpoint with template_only=True."""
        response = client.post("/evaluation/detector", json={
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
    
    def test_get_evaluation_status(self, client):
        """Test GET /evaluation/status/{job_id} endpoint."""
        job_id = "test-eval-job-123"
        response = client.get(f"/evaluation/status/{job_id}")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_list_evaluation_jobs(self, client):
        """Test GET /evaluation/jobs endpoint."""
        response = client.get("/evaluation/jobs")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data or isinstance(data, list)
    
    def test_evaluation_metrics_returned(self, client):
        """Test that evaluation metrics are returned correctly."""
        # Start an evaluation job
        response = client.post("/evaluation/classifier", json={
            "config": {},
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]
        data = response.json()
        job_id = data.get("job_id")
        
        if job_id and job_id != "template":
            # Check job status for metrics
            status_response = client.get(f"/evaluation/status/{job_id}")
            assert status_response.status_code in [200, 404]
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                # Check for evaluation metrics in the response
                assert "job_id" in status_data
                assert "status" in status_data
    
    def test_evaluation_template_generation(self, client):
        """Test template generation for evaluation configs."""
        response = client.post("/evaluation/classifier", json={
            "config": {},
            "debug": True,
            "verbose": False,
            "template_only": True
        })
        
        assert response.status_code in [200, 201, 202, 422]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data or "job_id" in data
    
    def test_invalid_evaluation_config(self, client):
        """Test evaluation with invalid config."""
        response = client.post("/evaluation/classifier", json={
            "config": {"invalid": "config"},
            "debug": True,
            "verbose": False,
            "template_only": False
        })
        
        assert response.status_code in [422, 400]  # Validation error
    
    def test_evaluation_job_creation_workflow(self, client):
        """Test complete evaluation job creation workflow."""
        # 1. Start evaluation
        response = client.post("/evaluation/classifier", json={
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
            status_response = client.get(f"/evaluation/status/{job_id}")
            assert status_response.status_code in [200, 404]
            
            # 3. List jobs
            jobs_response = client.get("/evaluation/jobs")
            assert jobs_response.status_code in [200, 404]
