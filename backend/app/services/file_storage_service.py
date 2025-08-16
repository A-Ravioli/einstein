"""
Enhanced file storage service with S3 and local storage support
"""

import os
import hashlib
import mimetypes
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.core.config import settings
from loguru import logger


class FileStorageService:
    """Service for handling file storage operations"""
    
    def __init__(self):
        self.storage_backend = settings.S3_BUCKET and "s3" or "local"
        
        if self.storage_backend == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            self.bucket_name = settings.S3_BUCKET
        else:
            # Local storage configuration
            self.local_storage_path = Path("data/files")
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(
        self, 
        file: UploadFile, 
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload a file to the configured storage backend"""
        try:
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Calculate checksums
            md5_hash = hashlib.md5(content).hexdigest()
            sha256_hash = hashlib.sha256(content).hexdigest()
            
            # Detect MIME type
            mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
            
            # Prepare metadata
            file_metadata = {
                "original_filename": file.filename,
                "file_size": file_size,
                "md5_hash": md5_hash,
                "sha256_hash": sha256_hash,
                "mime_type": mime_type,
                "upload_timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            if self.storage_backend == "s3":
                result = await self._upload_to_s3(content, file_path, file_metadata)
            else:
                result = await self._upload_to_local(content, file_path, file_metadata)
            
            logger.info(f"Uploaded file: {file.filename} to {file_path}")
            
            return {
                **result,
                "storage_backend": self.storage_backend,
                "file_size": file_size,
                "md5_hash": md5_hash,
                "sha256_hash": sha256_hash,
                "mime_type": mime_type
            }
        
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            raise
    
    async def _upload_to_s3(
        self, 
        content: bytes, 
        file_path: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload file to S3"""
        try:
            # Convert metadata to string values for S3
            s3_metadata = {k: str(v) for k, v in metadata.items()}
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=content,
                Metadata=s3_metadata,
                ContentType=metadata.get("mime_type", "application/octet-stream")
            )
            
            return {
                "storage_path": file_path,
                "storage_url": f"s3://{self.bucket_name}/{file_path}"
            }
        
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise
    
    async def _upload_to_local(
        self, 
        content: bytes, 
        file_path: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload file to local storage"""
        try:
            local_path = self.local_storage_path / file_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(local_path, 'wb') as f:
                f.write(content)
            
            # Write metadata file
            metadata_path = local_path.with_suffix(local_path.suffix + '.meta')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "storage_path": str(local_path),
                "storage_url": f"file://{local_path.absolute()}"
            }
        
        except Exception as e:
            logger.error(f"Local storage upload error: {e}")
            raise
    
    async def download_file(self, file_path: str) -> bytes:
        """Download file content"""
        try:
            if self.storage_backend == "s3":
                return await self._download_from_s3(file_path)
            else:
                return await self._download_from_local(file_path)
        
        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {e}")
            raise
    
    async def _download_from_s3(self, file_path: str) -> bytes:
        """Download file from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return response['Body'].read()
        
        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            raise
    
    async def _download_from_local(self, file_path: str) -> bytes:
        """Download file from local storage"""
        try:
            local_path = Path(file_path)
            if not local_path.is_absolute():
                local_path = self.local_storage_path / file_path
            
            with open(local_path, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Local storage download error: {e}")
            raise
    
    async def generate_download_url(
        self, 
        file_path: str, 
        expires_in: int = 3600
    ) -> str:
        """Generate a signed download URL"""
        try:
            if self.storage_backend == "s3":
                return self._generate_s3_download_url(file_path, expires_in)
            else:
                return self._generate_local_download_url(file_path, expires_in)
        
        except Exception as e:
            logger.error(f"Error generating download URL for {file_path}: {e}")
            raise
    
    def _generate_s3_download_url(self, file_path: str, expires_in: int) -> str:
        """Generate S3 presigned URL"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_path},
                ExpiresIn=expires_in
            )
            return url
        
        except ClientError as e:
            logger.error(f"S3 presigned URL generation error: {e}")
            raise
    
    def _generate_local_download_url(self, file_path: str, expires_in: int) -> str:
        """Generate local file access URL"""
        # In a real implementation, you might use a file serving endpoint
        # For now, return a placeholder URL
        return f"/api/v1/files/download/{file_path}"
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            if self.storage_backend == "s3":
                return await self._delete_from_s3(file_path)
            else:
                return await self._delete_from_local(file_path)
        
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise
    
    async def _delete_from_s3(self, file_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        
        except ClientError as e:
            logger.error(f"S3 delete error: {e}")
            return False
    
    async def _delete_from_local(self, file_path: str) -> bool:
        """Delete file from local storage"""
        try:
            local_path = Path(file_path)
            if not local_path.is_absolute():
                local_path = self.local_storage_path / file_path
            
            if local_path.exists():
                local_path.unlink()
                
                # Also delete metadata file if it exists
                metadata_path = local_path.with_suffix(local_path.suffix + '.meta')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                return True
            return False
        
        except Exception as e:
            logger.error(f"Local storage delete error: {e}")
            return False
    
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata and information"""
        try:
            if self.storage_backend == "s3":
                return await self._get_s3_file_info(file_path)
            else:
                return await self._get_local_file_info(file_path)
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    async def _get_s3_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file info from S3"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            
            return {
                "file_size": response.get('ContentLength'),
                "last_modified": response.get('LastModified'),
                "etag": response.get('ETag', '').strip('"'),
                "content_type": response.get('ContentType'),
                "metadata": response.get('Metadata', {})
            }
        
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            logger.error(f"S3 head object error: {e}")
            raise
    
    async def _get_local_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file info from local storage"""
        try:
            local_path = Path(file_path)
            if not local_path.is_absolute():
                local_path = self.local_storage_path / file_path
            
            if not local_path.exists():
                return None
            
            stat = local_path.stat()
            
            # Try to read metadata file
            metadata = {}
            metadata_path = local_path.with_suffix(local_path.suffix + '.meta')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return {
                "file_size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime),
                "content_type": mimetypes.guess_type(str(local_path))[0],
                "metadata": metadata
            }
        
        except Exception as e:
            logger.error(f"Local storage file info error: {e}")
            return None
    
    def detect_scientific_file_type(self, filename: str, content: bytes = None) -> str:
        """Detect scientific file type based on filename and content"""
        extension = Path(filename).suffix.lower()
        
        # Common scientific file formats
        scientific_formats = {
            '.fasta': 'fasta',
            '.fa': 'fasta',
            '.fas': 'fasta',
            '.fna': 'fasta',
            '.pdb': 'pdb',
            '.cif': 'cif',
            '.sdf': 'sdf',
            '.mol': 'mol',
            '.mol2': 'mol2',
            '.xyz': 'xyz',
            '.gro': 'gromacs',
            '.xtc': 'gromacs_trajectory',
            '.dcd': 'trajectory',
            '.nc': 'netcdf',
            '.h5': 'hdf5',
            '.mtz': 'mtz',
            '.ccp4': 'ccp4',
            '.dx': 'opendx',
            '.cube': 'gaussian_cube'
        }
        
        detected_type = scientific_formats.get(extension, 'unknown')
        
        # Content-based detection for ambiguous cases
        if content and detected_type == 'unknown':
            content_str = content[:1000].decode('utf-8', errors='ignore').upper()
            
            if content_str.startswith('>'):
                detected_type = 'fasta'
            elif 'HEADER' in content_str and 'PDB' in content_str:
                detected_type = 'pdb'
            elif 'data_' in content_str:
                detected_type = 'cif'
        
        return detected_type
