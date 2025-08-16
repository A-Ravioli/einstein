"""
Global error handlers for FastAPI application
"""

import traceback
from typing import Union
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, DataError
from pydantic import ValidationError

from app.core.exceptions import (
    EinsteinException,
    WorkflowValidationError,
    WorkflowExecutionError,
    PlatformNotAvailableError,
    PlatformAuthenticationError,
    JobNotFoundError,
    FileNotFoundError,
    ConfigurationError,
    ResourceLimitError
)
from loguru import logger


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: dict = None,
    request_id: str = None
) -> JSONResponse:
    """Create standardized error response"""
    error_response = {
        "error": {
            "code": error_code,
            "message": message,
            "status_code": status_code
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    if request_id:
        error_response["error"]["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def einstein_exception_handler(request: Request, exc: EinsteinException) -> JSONResponse:
    """Handle custom Einstein exceptions"""
    logger.error(f"Einstein exception: {exc.message} | Details: {exc.details}")
    
    # Map error codes to HTTP status codes
    status_code_map = {
        "WORKFLOW_VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "WORKFLOW_EXECUTION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "PLATFORM_NOT_AVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
        "PLATFORM_AUTH_ERROR": status.HTTP_401_UNAUTHORIZED,
        "JOB_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "FILE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RESOURCE_LIMIT_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
    }
    
    status_code = status_code_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return create_error_response(
        status_code=status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        request_id=getattr(request.state, "request_id", None)
    )


async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]) -> JSONResponse:
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error: {exc}")
    
    error_details = []
    if hasattr(exc, 'errors'):
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", ""),
                "type": error.get("type", "")
            })
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message="Input validation failed",
        details={"validation_errors": error_details},
        request_id=getattr(request.state, "request_id", None)
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return create_error_response(
        status_code=exc.status_code,
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        request_id=getattr(request.state, "request_id", None)
    )


async def database_exception_handler(request: Request, exc: Union[IntegrityError, DataError]) -> JSONResponse:
    """Handle database-related exceptions"""
    logger.error(f"Database exception: {exc}")
    
    if isinstance(exc, IntegrityError):
        error_code = "DATABASE_INTEGRITY_ERROR"
        message = "Data integrity constraint violation"
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, DataError):
        error_code = "DATABASE_DATA_ERROR"
        message = "Invalid data format or type"
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        error_code = "DATABASE_ERROR"
        message = "Database operation failed"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return create_error_response(
        status_code=status_code,
        error_code=error_code,
        message=message,
        details={"database_error": str(exc.orig) if hasattr(exc, 'orig') else str(exc)},
        request_id=getattr(request.state, "request_id", None)
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Don't expose internal error details in production
    if hasattr(request.app.state, "environment") and request.app.state.environment == "production":
        message = "An internal server error occurred"
        details = None
    else:
        message = str(exc)
        details = {"traceback": traceback.format_exc()}
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_SERVER_ERROR",
        message=message,
        details=details,
        request_id=getattr(request.state, "request_id", None)
    )


def setup_error_handlers(app):
    """Setup all error handlers for the FastAPI app"""
    
    # Custom Einstein exceptions
    app.add_exception_handler(EinsteinException, einstein_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    
    # Database exceptions
    app.add_exception_handler(IntegrityError, database_exception_handler)
    app.add_exception_handler(DataError, database_exception_handler)
    
    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)
