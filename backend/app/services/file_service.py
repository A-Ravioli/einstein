"""
File service layer
"""

import hashlib
import os
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.file import WorkflowFile, FileShare
from app.schemas.file import WorkflowFileResponse, FileUploadResponse
from app.services.file_storage_service import FileStorageService
from app.core.config import settings
from loguru import logger


class FileService:
    """Service for file management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage_service = FileStorageService()
    
    async def upload_file(
        self, 
        file: UploadFile, 
        workflow_id: Optional[int] = None,
        job_id: Optional[int] = None,
        file_role: str = "input"
    ) -> FileUploadResponse:
        """Upload a file"""
        try:
            # Generate storage path
            storage_path = f"files/{workflow_id or 'standalone'}/{file.filename}"
            
            # Detect scientific file type
            file_type = self.storage_service.detect_scientific_file_type(file.filename)
            
            # Upload to storage backend
            storage_result = await self.storage_service.upload_file(
                file=file,
                file_path=storage_path,
                metadata={
                    "workflow_id": str(workflow_id) if workflow_id else None,
                    "job_id": str(job_id) if job_id else None,
                    "file_role": file_role,
                    "file_type": file_type
                }
            )
            
            # Create file record in database
            file_record = WorkflowFile(
                filename=file.filename,
                original_filename=file.filename,
                workflow_id=workflow_id,
                job_id=job_id,
                file_type=file_type,
                mime_type=storage_result["mime_type"],
                storage_path=storage_result["storage_path"],
                storage_backend=storage_result["storage_backend"],
                file_size=storage_result["file_size"],
                checksum_md5=storage_result["md5_hash"],
                file_role=file_role
            )
            
            self.db.add(file_record)
            await self.db.commit()
            await self.db.refresh(file_record)
            
            logger.info(f"Uploaded file: {file_record.id} ({file.filename})")
            
            return FileUploadResponse(
                file_id=file_record.id,
                filename=file.filename,
                file_size=storage_result["file_size"],
                file_type=file_type
            )
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def get_download_url(self, file_id: int, expires_in: int = 3600) -> Optional[str]:
        """Get download URL for a file"""
        try:
            query = select(WorkflowFile).where(WorkflowFile.id == file_id)
            result = await self.db.execute(query)
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                return None
            
            # Generate signed download URL from storage service
            download_url = await self.storage_service.generate_download_url(
                file_record.storage_path,
                expires_in=expires_in
            )
            
            logger.info(f"Generated download URL for file: {file_id}")
            return download_url
        
        except Exception as e:
            logger.error(f"Error generating download URL for file {file_id}: {e}")
            raise
    
    async def list_files(
        self,
        workflow_id: Optional[int] = None,
        file_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WorkflowFileResponse]:
        """List files with optional filtering"""
        try:
            query = select(WorkflowFile).offset(skip).limit(limit)
            
            filters = []
            if workflow_id:
                filters.append(WorkflowFile.workflow_id == workflow_id)
            if file_type:
                filters.append(WorkflowFile.file_type == file_type)
            
            if filters:
                query = query.where(and_(*filters))
            
            result = await self.db.execute(query)
            files = result.scalars().all()
            
            return [WorkflowFileResponse.from_orm(file) for file in files]
        
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise
    
    async def delete_file(self, file_id: int) -> bool:
        """Delete a file"""
        try:
            query = select(WorkflowFile).where(WorkflowFile.id == file_id)
            result = await self.db.execute(query)
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                return False
            
            # Delete from storage backend
            storage_deleted = await self.storage_service.delete_file(file_record.storage_path)
            if not storage_deleted:
                logger.warning(f"Failed to delete file from storage: {file_record.storage_path}")
            
            # Delete from database
            await self.db.delete(file_record)
            await self.db.commit()
            
            logger.info(f"Deleted file: {file_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting file {file_id}: {e}")
            raise
    
    async def get_file_content(self, file_id: int) -> Optional[bytes]:
        """Get file content"""
        try:
            query = select(WorkflowFile).where(WorkflowFile.id == file_id)
            result = await self.db.execute(query)
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                return None
            
            # Download from storage backend
            content = await self.storage_service.download_file(file_record.storage_path)
            
            logger.info(f"Retrieved content for file: {file_id}")
            return content
        
        except Exception as e:
            logger.error(f"Error getting file content for {file_id}: {e}")
            raise
