"""FastAPI application entry point for WildTrain API."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import Dict, Any

from .routers import training, evaluation, pipeline, visualization, dataset, config
from .utils.error_handling import WildTrainAPIException, wildtrain_exception_handler
from .dependencies import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="WildTrain API",
        description="REST API for WildTrain - Modular Computer Vision Framework",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom exception handler
    app.add_exception_handler(WildTrainAPIException, wildtrain_exception_handler)
    
    # Include routers
    app.include_router(training.router, prefix="/training", tags=["Training"])
    app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
    app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
    app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
    app.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
    app.include_router(config.router, prefix="/config", tags=["Configuration"])
    
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "message": "WildTrain API",
            "version": "0.1.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unexpected errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(exc) if get_settings().debug else "An unexpected error occurred"
            }
        )
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
