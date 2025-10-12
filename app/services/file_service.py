import os
import uuid
import tempfile
from typing import Optional
from fastapi import UploadFile
from app.config import settings
from app.core.exceptions import FileProcessingError

# Cloudinary import (optional - only if enabled)
if settings.USE_CLOUDINARY:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api

    # Configure Cloudinary
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True
    )

class FileService:
    """Service for handling file operations with Cloudinary support"""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.max_file_size = settings.MAX_FILE_SIZE
        self.use_cloudinary = settings.USE_CLOUDINARY

    async def save_uploaded_file(self, uploaded_file: UploadFile) -> str:
        """
        Save uploaded file to Cloudinary or local disk
        
        Args:
            uploaded_file: FastAPI UploadFile object
            
        Returns:
            Cloudinary URL or local file path
        """
        try:
            # Check file size
            content = await uploaded_file.read()
            if len(content) > self.max_file_size:
                raise FileProcessingError(f"File too large: {len(content)} bytes (max: {self.max_file_size})")

            # Generate filename
            file_name = self._generate_filename(uploaded_file.filename)

            if self.use_cloudinary:
                # Upload to Cloudinary
                return await self._upload_to_cloudinary(content, file_name)
            else:
                # Save locally
                file_path = os.path.join(self.upload_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(content)
                return file_path

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"Failed to save file: {str(e)}")

    async def save_audio_file(self, audio_data: bytes, filename: Optional[str] = None) -> str:
        """
        Save audio file to Cloudinary or local disk
        
        Args:
            audio_data: Audio file bytes
            filename: Optional filename (will generate if not provided)
            
        Returns:
            Cloudinary URL or local file path
        """
        try:
            # Check file size
            if len(audio_data) > self.max_file_size:
                raise FileProcessingError(f"File too large: {len(audio_data)} bytes (max: {self.max_file_size})")

            # Generate or clean filename
            if not filename:
                filename = self._generate_filename()
            else:
                filename = self._generate_filename(filename)

            if self.use_cloudinary:
                # Upload to Cloudinary
                return await self._upload_to_cloudinary(audio_data, filename)
            else:
                # Save locally
                file_path = os.path.join(self.upload_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                return file_path

        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"Failed to save audio file: {str(e)}")

    async def _upload_to_cloudinary(self, file_data: bytes, filename: str) -> str:
        """
        Upload file to Cloudinary
        
        Args:
            file_data: File bytes
            filename: Filename for identification
            
        Returns:
            Cloudinary public URL
        """
        try:
            # Create temporary file (Cloudinary SDK needs file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name

            try:
                # Upload to Cloudinary
                result = cloudinary.uploader.upload(
                    temp_path,
                    public_id=os.path.splitext(filename)[0],  # Use filename without extension
                    resource_type="video",  # Cloudinary treats audio as video
                    overwrite=False,
                    unique_filename=True
                )

                # Return secure URL
                return result['secure_url']

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            raise FileProcessingError(f"Failed to upload to Cloudinary: {str(e)}")

    def get_audio_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve audio file from Cloudinary URL or local disk
        
        Args:
            file_path: Cloudinary URL or local file path
            
        Returns:
            Audio file bytes or None if not found
        """
        try:
            # Check if it's a URL (Cloudinary)
            if file_path.startswith(('http://', 'https://')):
                import requests
                response = requests.get(file_path)
                if response.status_code == 200:
                    return response.content
                return None
            else:
                # Local file
                if not os.path.exists(file_path):
                    return None
                with open(file_path, "rb") as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading audio file {file_path}: {e}")
            return None

    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists (Cloudinary URL or local)
        
        Args:
            file_path: Cloudinary URL or local file path
            
        Returns:
            True if exists, False otherwise
        """
        try:
            # Check if it's a URL (Cloudinary)
            if file_path.startswith(('http://', 'https://')):
                import requests
                response = requests.head(file_path)
                return response.status_code == 200
            else:
                # Local file
                return os.path.exists(file_path)
        except:
            return False

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
        Clean up file from Cloudinary or local disk
        
        Args:
            file_path: Cloudinary URL or local file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if it's a Cloudinary URL
            if file_path.startswith(('http://', 'https://')) and self.use_cloudinary:
                # Extract public_id from Cloudinary URL
                # URL format: https://res.cloudinary.com/{cloud_name}/video/upload/v{version}/{folder}/{public_id}.wav
                parts = file_path.split('/')
                if len(parts) > 0:
                    # Get public_id with folder
                    public_id_with_ext = '/'.join(parts[-2:])  # folder/filename.wav
                    public_id = os.path.splitext(public_id_with_ext)[0]  # Remove extension

                    # Delete from Cloudinary
                    cloudinary.uploader.destroy(
                        public_id,
                        resource_type="video"
                    )
                    return True
            else:
                # Local file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
            return False
        except Exception as e:
            print(f"Warning: Failed to cleanup file {file_path}: {e}")
            return False