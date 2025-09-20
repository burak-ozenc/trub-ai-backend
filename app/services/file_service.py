import os
import uuid
from typing import BinaryIO
from fastapi import UploadFile
from app.config import settings
from app.core.exceptions import FileProcessingError

class FileService:
    """Service for handling file operations"""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.max_file_size = settings.MAX_FILE_SIZE

    async def save_uploaded_file(self, uploaded_file: UploadFile) -> str:
        """
        Save uploaded file to disk
        
        Args:
            uploaded_file: FastAPI UploadFile object
            
        Returns:
            Full path to saved file
        """
        try:
            # Generate filename
            file_name = self._generate_filename(uploaded_file.filename)
            file_path = os.path.join(self.upload_dir, file_name)

            # Check file size
            content = await uploaded_file.read()
            if len(content) > self.max_file_size:
                raise FileProcessingError(f"File too large: {len(content)} bytes (max: {self.max_file_size})")

            # Save file
            with open(file_path, "wb") as f:
                f.write(content)

            return file_path

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"Failed to save file: {str(e)}")

    def _generate_filename(self, original_filename: str = None) -> str:
        """
        Generate unique filename
        
        Args:
            original_filename: Original filename from upload
            
        Returns:
            Generated unique filename
        """
        if original_filename and original_filename != "blob":
            # Extract extension from original filename
            name, ext = os.path.splitext(original_filename)
            if not ext:
                ext = ".wav"  # Default extension
            return f"{name}_{uuid.uuid4().hex}{ext}"
        else:
            # Generate completely new filename
            return f"audio_{uuid.uuid4().hex}.wav"

    def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up temporary file
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to cleanup file {file_path}: {e}")
            return False