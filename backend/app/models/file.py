"""
File management database models
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class WorkflowFile(Base):
    """File associated with workflows"""
    __tablename__ = "workflow_files"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Workflow relationship (optional - files can exist without workflows)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=True)
    workflow = relationship("Workflow", back_populates="files")
    
    # Job relationship (optional - for output files)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    
    # File metadata
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_type = Column(String(100))  # e.g., "fasta", "pdb", "sdf", "csv"
    mime_type = Column(String(255))
    
    # Storage details
    storage_path = Column(String(1024), nullable=False)  # S3 key or file path
    storage_backend = Column(String(50), default="s3")  # "s3", "local", "gcs"
    file_size = Column(BigInteger)  # Size in bytes
    checksum_md5 = Column(String(32))
    
    # File purpose and context
    file_role = Column(String(50))  # "input", "output", "intermediate", "result"
    description = Column(Text)
    metadata = Column(JSON, default=dict)  # Additional file metadata
    
    # Access control
    is_public = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True))  # For temporary files
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<WorkflowFile(id={self.id}, filename='{self.filename}', file_type='{self.file_type}')>"


class FileShare(Base):
    """File sharing records"""
    __tablename__ = "file_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # File relationship
    file_id = Column(Integer, ForeignKey("workflow_files.id"), nullable=False)
    
    # Sharing details
    share_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True))
    download_count = Column(Integer, default=0)
    max_downloads = Column(Integer)  # Optional download limit
    
    # Access control
    password_hash = Column(String(255))  # Optional password protection
    allowed_ips = Column(JSON, default=list)  # Optional IP restrictions
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<FileShare(id={self.id}, file_id={self.file_id}, share_token='{self.share_token}')>"
