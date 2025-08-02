"""
Main FastAPI application entry point for Stroke Lab Backend

This module creates and configures the FastAPI application with all routes,
middleware, and startup/shutdown event handlers.
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time

from app.core.config import get_settings, setup_logging
from app.db.database import init_database, health_check
from app.api.endpoints import app as api_app

# Setup logging
logger = setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Stroke Lab Backend...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        init_database()
        
        # Verify database connection
        db_status = health_check()
        if db_status["status"] == "healthy":
            logger.info("✅ Database connection verified")
        else:
            logger.error("❌ Database connection failed")
            raise Exception("Database initialization failed")
        
        # Setup directories
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directories: {settings.MODELS_DIR}, {settings.DATASETS_DIR}")
        
        logger.info(f"🎯 Stroke Lab Backend v{settings.APP_VERSION} started successfully!")
        logger.info(f"🌐 API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Stroke Lab Backend...")
    logger.info("✅ Shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Stroke Lab Backend",
        description="""
        Advanced backend for the Stroke Invariant Research Laboratory.
        
        ## Features
        
        * **Dataset Management**: Import and process handwritten character datasets
        * **Image-to-Stroke Conversion**: Convert images to time-series stroke data
        * **Machine Learning Training**: Train models on stroke invariant features
        * **Real-time Analytics**: Real-time stroke analysis and feature extraction
        * **Model Export**: Export trained models for deployment
        
        ## Getting Started
        
        1. Check system health: `GET /api/health`
        2. Import dataset: `POST /api/dataset/import`
        3. Start training: `POST /api/train`
        4. Monitor progress: `GET /api/train/{session_id}/status`
        5. Export model: `GET /api/models/{model_id}/export`
        
        ## Mathematical Framework
        
        The system implements advanced mathematical analysis including:
        - Affine differential geometry
        - Topological data analysis (TDA)
        - Path signature calculations
        - Spectral analysis with wavelets
        - Persistent homology
        
        """,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.DEBUG
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add custom exception handlers
    setup_exception_handlers(app)
    
    # Include API routes from endpoints.py
    app.mount("/", api_app)
    
    return app


def setup_middleware(app: FastAPI):
    """Setup middleware for the FastAPI application."""
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    # Gzip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"📤 {request.method} {request.url.path} "
            f"→ {response.status_code} ({process_time:.3f}s)"
        )
        
        return response


def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers."""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions with custom formatting."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "status_code": exc.status_code,
                    "message": exc.detail,
                    "path": str(request.url.path),
                    "timestamp": time.time()
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc} - {request.url}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "status_code": 422,
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "path": str(request.url.path),
                    "timestamp": time.time()
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc} - {request.url}", exc_info=True)
        
        if settings.DEBUG:
            # Include exception details in debug mode
            error_detail = str(exc)
        else:
            # Generic error message in production
            error_detail = "An internal server error occurred"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "status_code": 500,
                    "message": error_detail,
                    "path": str(request.url.path),
                    "timestamp": time.time()
                }
            }
        )


# Create the main application instance
app = create_app()


# Additional route for API information
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🧠 Stroke Lab Backend API",
        "version": settings.APP_VERSION,
        "status": "running",
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/api/health",
            "info": "/api/info",
            "dataset_import": "/api/dataset/import",
            "training": "/api/train",
            "models": "/api/models"
        },
        "features": [
            "Image-to-stroke conversion",
            "Mathematical invariant analysis",
            "Machine learning training",
            "Real-time stroke processing",
            "Model export and deployment"
        ]
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon endpoint to prevent 404s."""
    return JSONResponse(content={"message": "No favicon available"}, status_code=404)


# Health check endpoint (additional to the one in endpoints.py)
@app.get("/health", include_in_schema=False)
async def simple_health_check():
    """Simple health check endpoint."""
    try:
        db_status = health_check()
        status = "healthy" if db_status["status"] == "healthy" else "unhealthy"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "version": settings.APP_VERSION
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# CLI entry point
def main():
    """
    Main entry point for running the application.
    Can be called directly or via command line.
    """
    logger.info(f"🚀 Starting Stroke Lab Backend on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS
    )


if __name__ == "__main__":
    main()
