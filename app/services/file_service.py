import os
import uuid
from typing import Optional
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

    async def save_audio_file(self, audio_data: bytes, filename: Optional[str] = None) -> str:
        """
        Save audio file to disk
        
        Args:
            audio_data: Audio file bytes
            filename: Optional filename (will generate if not provided)
            
        Returns:
            Full path to saved file
        """
        try:
            # Generate or clean filename
            if not filename:
                filename = self._generate_filename()
            else:
                filename = self._generate_filename(filename)

            file_path = os.path.join(self.upload_dir, filename)

            # Check file size
            if len(audio_data) > self.max_file_size:
                raise FileProcessingError(f"File too large: {len(audio_data)} bytes (max: {self.max_file_size})")

            # Save file
            with open(file_path, "wb") as f:
                f.write(audio_data)

            return file_path

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"Failed to save audio file: {str(e)}")

    def get_audio_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve audio file from disk
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio file bytes or None if not found
        """
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading audio file {file_path}: {e}")
            return None

    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists
        
        Args:
            file_path: Path to file
            
        Returns:
            True if exists, False otherwise
        """
        return os.path.exists(file_path)

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