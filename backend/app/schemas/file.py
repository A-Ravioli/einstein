"""
File management Pydantic schemas
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class WorkflowFileBase(BaseModel):
    """Base workflow file schema"""
    filename: str = Field(..., min_length=1, max_length=512)
    file_type: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_public: bool = Field(default=False)


class WorkflowFileCreate(WorkflowFileBase):
    """Schema for creating a workflow file"""
    workflow_id: Optional[int] = None
    job_id: Optional[int] = None
    file_role: Optional[str] = Field(None, max_length=50)
    storage_backend: str = Field(default="s3", max_length=50)


class WorkflowFileUpdate(BaseModel):
    """Schema for updating a workflow file"""
    filename: Optional[str] = Field(None, min_length=1, max_length=512)
    file_type: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    expires_at: Optional[datetime] = None


class WorkflowFileResponse(WorkflowFileBase):
    """Schema for workflow file responses"""
    id: int
    original_filename: str
    workflow_id: Optional[int] = None
    job_id: Optional[int] = None
    mime_type: Optional[str] = None
    storage_path: str
    storage_backend: str
    file_size: Optional[int] = None
    checksum_md5: Optional[str] = None
    file_role: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """Schema for file list responses"""
    files: List[WorkflowFileResponse]
    total: int
    skip: int
    limit: int


class FileUploadResponse(BaseModel):
    """Schema for file upload responses"""
    file_id: int
    filename: str
    file_size: int
    upload_url: Optional[str] = None  # For direct S3 uploads
    file_type: Optional[str] = None
    message: str = "File uploaded successfully"


class FileDownloadResponse(BaseModel):
    """Schema for file download responses"""
    download_url: str
    filename: str
    file_size: Optional[int] = None
    expires_at: Optional[datetime] = None
    content_type: Optional[str] = None


class FileShareCreate(BaseModel):
    """Schema for creating file shares"""
    file_id: int
    expires_at: Optional[datetime] = None
    max_downloads: Optional[int] = Field(None, gt=0)
    password: Optional[str] = Field(None, min_length=4)
    allowed_ips: List[str] = Field(default_factory=list)


class FileShareResponse(BaseModel):
    """Schema for file share responses"""
    id: int
    file_id: int
    share_token: str
    expires_at: Optional[datetime] = None
    download_count: int
    max_downloads: Optional[int] = None
    created_at: datetime
    last_accessed: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class FileMetadata(BaseModel):
    """Schema for file metadata"""
    file_type: str
    size_bytes: int
    checksum_md5: str
    content_type: str
    encoding: Optional[str] = None
    scientific_metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkFileOperation(BaseModel):
    """Schema for bulk file operations"""
    file_ids: List[int] = Field(..., min_items=1)
    operation: str = Field(..., regex="^(delete|archive|share|move)$")
    parameters: Dict[str, Any] = Field(default_factory=dict)
