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
from ...cli.config_loader import ConfigType

logger = get_logger("config")

router = APIRouter()


@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a configuration file."""
    
    try:
        logger.info(f"Validating configuration file: {request.config_path}")
        from ...cli import validate_config
        try:
            validate_config(request.config_path, request.config_type)
            is_valid = True
            errors = []
            warnings = []
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            is_valid = False
            errors = [str(e)]
            warnings = []
        
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



@router.get("/types")
async def get_config_types() -> Dict[str, Any]:
    """Get available configuration types."""
    
    config_types = [
        {
            "name": ConfigType.CLASSIFICATION,
            "description": "Classification training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION,
            "description": "Detection training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.CLASSIFICATION_EVAL,
            "description": "Classification evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION_EVAL,
            "description": "Detection evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.CLASSIFICATION_VISUALIZATION,
            "description": "Classification visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION_VISUALIZATION,
            "description": "Detection visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.PIPELINE,
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
