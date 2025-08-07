"""Configuration endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..models.requests import ConfigValidationRequest, FileUploadRequest
from ..models.responses import (
    ConfigValidationResponse,
    TemplateResponse,
    FileUploadResponse
)
from ..utils.file_handling import save_config_file
from ..dependencies import get_logger

logger = get_logger("config")

router = APIRouter()


@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a configuration file."""
    
    try:
        # TODO: Implement actual configuration validation using WildTrain CLI
        logger.info(f"Validating configuration file: {request.config_path}")
        
        # Simulate validation
        import time
        time.sleep(0.5)
        
        # For now, assume valid if file exists
        config_path = Path(request.config_path)
        is_valid = config_path.exists()
        
        errors = []
        warnings = []
        
        if not is_valid:
            errors.append(f"Configuration file not found: {request.config_path}")
        else:
            # Add some example warnings
            warnings.append("Consider using GPU for faster training")
            warnings.append("Batch size might be too large for your GPU memory")
        
        return ConfigValidationResponse(
            success=True,
            message="Configuration validation completed",
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            config_type=request.config_type
        )
        
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/{config_type}", response_model=TemplateResponse)
async def get_config_template(config_type: str) -> TemplateResponse:
    """Get configuration template."""
    
    try:
        # TODO: Implement actual template generation using WildTrain CLI
        logger.info(f"Generating template for config type: {config_type}")
        
        # Simulate template generation
        templates = ...
        
        template = templates.get(config_type, "# Configuration template for " + config_type)
        
        return TemplateResponse(
            success=True,
            message=f"Template generated for {config_type}",
            template=template,
            config_type=config_type
        )
        
    except Exception as e:
        logger.error(f"Failed to generate template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=FileUploadResponse)
async def upload_config_file(
    file: UploadFile = File(...),
    file_type: str = "config"
) -> FileUploadResponse:
    """Upload a configuration file."""
    
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith(('.yaml', '.yml', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Only YAML and JSON files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Save file
        file_path = await save_config_file(content, file.filename or "config.yaml")
        
        return FileUploadResponse(
            success=True,
            message="Configuration file uploaded successfully",
            file_path=str(file_path),
            file_size=len(content),
            file_type=file_type
        )
        
    except Exception as e:
        logger.error(f"Failed to upload configuration file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_config_types() -> Dict[str, Any]:
    """Get available configuration types."""
    
    config_types = [
        {
            "name": "classification",
            "description": "Classification training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "detection",
            "description": "Detection training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "classification_eval",
            "description": "Classification evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "detection_eval",
            "description": "Detection evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "classification_visualization",
            "description": "Classification visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "detection_visualization",
            "description": "Detection visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": "pipeline",
            "description": "Pipeline configuration",
            "file_extension": ".yaml"
        }
    ]
    
    return {
        "success": True,
        "message": "Configuration types retrieved",
        "config_types": config_types,
        "total_types": len(config_types)
    }
