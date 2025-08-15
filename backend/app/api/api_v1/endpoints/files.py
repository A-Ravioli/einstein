"""
File management endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.file_service import FileService

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    workflow_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Upload a file"""
    service = FileService(db)
    file_record = await service.upload_file(file, workflow_id)
    return file_record


@router.get("/{file_id}/download")
async def download_file(
    file_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get download URL for a file"""
    service = FileService(db)
    download_url = await service.get_download_url(file_id)
    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    return {"download_url": download_url}


@router.get("/")
async def list_files(
    workflow_id: int = None,
    file_type: str = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List files with optional filtering"""
    service = FileService(db)
    files = await service.list_files(
        workflow_id=workflow_id,
        file_type=file_type,
        skip=skip,
        limit=limit
    )
    return files


@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a file"""
    service = FileService(db)
    success = await service.delete_file(file_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    return {"message": "File deleted successfully"}