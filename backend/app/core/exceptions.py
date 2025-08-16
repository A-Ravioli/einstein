"""
Custom exception classes for Einstein workflow builder
"""

from typing import Any, Dict, Optional


class EinsteinException(Exception):
    """Base exception class for Einstein-specific errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None, 
        details: Dict[str, Any] = None
    ):
        self.message = message
        self.error_code = error_code or "EINSTEIN_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class WorkflowException(EinsteinException):
    """Workflow-related exceptions"""
    pass


class WorkflowValidationError(WorkflowException):
    """Workflow validation errors"""
    
    def __init__(self, validation_errors: list, workflow_id: int = None):
        self.validation_errors = validation_errors
        self.workflow_id = workflow_id
        message = f"Workflow validation failed: {'; '.join(validation_errors)}"
        super().__init__(
            message=message,
            error_code="WORKFLOW_VALIDATION_ERROR",
            details={"validation_errors": validation_errors, "workflow_id": workflow_id}
        )


class WorkflowExecutionError(WorkflowException):
    """Workflow execution errors"""
    
    def __init__(self, message: str, workflow_id: int = None, platform: str = None):
        self.workflow_id = workflow_id
        self.platform = platform
        super().__init__(
            message=message,
            error_code="WORKFLOW_EXECUTION_ERROR",
            details={"workflow_id": workflow_id, "platform": platform}
        )


class PlatformException(EinsteinException):
    """Platform integration exceptions"""
    pass


class PlatformNotAvailableError(PlatformException):
    """Platform is not available or not responding"""
    
    def __init__(self, platform: str, reason: str = None):
        self.platform = platform
        self.reason = reason
        message = f"Platform '{platform}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            error_code="PLATFORM_NOT_AVAILABLE",
            details={"platform": platform, "reason": reason}
        )


class PlatformAuthenticationError(PlatformException):
    """Platform authentication errors"""
    
    def __init__(self, platform: str, reason: str = None):
        self.platform = platform
        self.reason = reason
        message = f"Authentication failed for platform '{platform}'"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            error_code="PLATFORM_AUTH_ERROR",
            details={"platform": platform, "reason": reason}
        )


class PlatformTemporaryError(PlatformException):
    """Temporary platform errors that can be retried"""
    
    def __init__(self, platform: str, message: str, retry_after: int = None):
        self.platform = platform
        self.retry_after = retry_after
        super().__init__(
            message=message,
            error_code="PLATFORM_TEMPORARY_ERROR",
            details={"platform": platform, "retry_after": retry_after}
        )


class PlatformPermanentError(PlatformException):
    """Permanent platform errors that should not be retried"""
    
    def __init__(self, platform: str, message: str):
        self.platform = platform
        super().__init__(
            message=message,
            error_code="PLATFORM_PERMANENT_ERROR",
            details={"platform": platform}
        )


class JobException(EinsteinException):
    """Job execution exceptions"""
    pass


class JobNotFoundError(JobException):
    """Job not found"""
    
    def __init__(self, job_id: int):
        self.job_id = job_id
        super().__init__(
            message=f"Job {job_id} not found",
            error_code="JOB_NOT_FOUND",
            details={"job_id": job_id}
        )


class JobExecutionError(JobException):
    """Job execution errors"""
    
    def __init__(self, message: str, job_id: int = None, platform: str = None):
        self.job_id = job_id
        self.platform = platform
        super().__init__(
            message=message,
            error_code="JOB_EXECUTION_ERROR",
            details={"job_id": job_id, "platform": platform}
        )


class FileException(EinsteinException):
    """File management exceptions"""
    pass


class FileNotFoundError(FileException):
    """File not found"""
    
    def __init__(self, file_id: int = None, file_path: str = None):
        self.file_id = file_id
        self.file_path = file_path
        
        if file_id:
            message = f"File {file_id} not found"
        else:
            message = f"File not found: {file_path}"
        
        super().__init__(
            message=message,
            error_code="FILE_NOT_FOUND",
            details={"file_id": file_id, "file_path": file_path}
        )


class FileUploadError(FileException):
    """File upload errors"""
    
    def __init__(self, message: str, filename: str = None):
        self.filename = filename
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            details={"filename": filename}
        )


class FileStorageError(FileException):
    """File storage backend errors"""
    
    def __init__(self, message: str, backend: str = None, operation: str = None):
        self.backend = backend
        self.operation = operation
        super().__init__(
            message=message,
            error_code="FILE_STORAGE_ERROR",
            details={"backend": backend, "operation": operation}
        )


class ConfigurationError(EinsteinException):
    """Configuration errors"""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )


class ResourceLimitError(EinsteinException):
    """Resource limit exceeded errors"""
    
    def __init__(self, message: str, resource_type: str = None, limit: Any = None, current: Any = None):
        self.resource_type = resource_type
        self.limit = limit
        self.current = current
        super().__init__(
            message=message,
            error_code="RESOURCE_LIMIT_ERROR",
            details={"resource_type": resource_type, "limit": limit, "current": current}
        )


class TemplateException(EinsteinException):
    """Template-related exceptions"""
    pass


class TemplateNotFoundError(TemplateException):
    """Template not found"""
    
    def __init__(self, template_id: int):
        self.template_id = template_id
        super().__init__(
            message=f"Template {template_id} not found",
            error_code="TEMPLATE_NOT_FOUND",
            details={"template_id": template_id}
        )


class TemplateParameterError(TemplateException):
    """Template parameter errors"""
    
    def __init__(self, message: str, template_id: int = None, parameter: str = None):
        self.template_id = template_id
        self.parameter = parameter
        super().__init__(
            message=message,
            error_code="TEMPLATE_PARAMETER_ERROR",
            details={"template_id": template_id, "parameter": parameter}
        )
