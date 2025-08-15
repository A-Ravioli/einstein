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
from app.core.config import settings
from loguru import logger


class FileService:
    """Service for file management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def upload_file(
        self, 
        file: UploadFile, 
        workflow_id: Optional[int] = None,
        job_id: Optional[int] = None,
        file_role: str = "input"
    ) -> FileUploadResponse:
        """Upload a file"""
        try:
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Calculate MD5 checksum
            md5_hash = hashlib.md5(content).hexdigest()
            
            # Generate storage path
            storage_path = f"files/{workflow_id or 'standalone'}/{file.filename}"
            
            # Detect file type from extension
            file_type = self._detect_file_type(file.filename)
            
            # Create file record
            file_record = WorkflowFile(
                filename=file.filename,
                original_filename=file.filename,
                workflow_id=workflow_id,
                job_id=job_id,
                file_type=file_type,
                mime_type=file.content_type,
                storage_path=storage_path,
                storage_backend=settings.S3_BUCKET and "s3" or "local",
                file_size=file_size,
                checksum_md5=md5_hash,
                file_role=file_role
            )
            
            self.db.add(file_record)
            await self.db.commit()
            await self.db.refresh(file_record)
            
            # TODO: Upload to actual storage (S3 or local filesystem)
            # For now, just simulate the upload
            
            logger.info(f"Uploaded file: {file_record.id} ({file.filename})")
            
            return FileUploadResponse(
                file_id=file_record.id,
                filename=file.filename,
                file_size=file_size,
                file_type=file_type
            )
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def get_download_url(self, file_id: int) -> Optional[str]:
        """Get download URL for a file"""
        try:
            query = select(WorkflowFile).where(WorkflowFile.id == file_id)
            result = await self.db.execute(query)
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                return None
            
            # TODO: Generate actual signed URL for S3 or local file access
            # For now, return a placeholder URL
            download_url = f"/api/v1/files/{file_id}/download"
            
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
            
            # TODO: Delete from actual storage (S3 or local filesystem)
            
            await self.db.delete(file_record)
            await self.db.commit()
            
            logger.info(f"Deleted file: {file_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting file {file_id}: {e}")
            raise
    
    def _detect_file_type(self, filename: str) -> Optional[str]:
        """Detect file type from filename extension"""
        extension = os.path.splitext(filename)[1].lower()
        
        file_type_map = {
            '.fasta': 'fasta',
            '.fa': 'fasta',
            '.fas': 'fasta',
            '.pdb': 'pdb',
            '.cif': 'cif',
            '.sdf': 'sdf',
            '.mol': 'mol',
            '.mol2': 'mol2',
            '.xyz': 'xyz',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'text',
            '.json': 'json',
            '.xml': 'xml',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.pdf': 'pdf',
            '.zip': 'archive',
            '.tar.gz': 'archive',
            '.tar': 'archive'
        }
        
        return file_type_map.get(extension, 'unknown')
