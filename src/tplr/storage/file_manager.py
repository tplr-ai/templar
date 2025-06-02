# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import uuid
from typing import Optional

import tplr

# Constants
LOCAL_TMP_DIR = "/tmp/local_store"


class FileManager:
    """Manages local file operations and cleanup"""

    def __init__(self, base_temp_dir: str, uid: Optional[str] = None):
        """Initialize with base temporary directory"""
        self.base_temp_dir = base_temp_dir
        self.uid = uid

        # Create base temp directory
        os.makedirs(self.base_temp_dir, exist_ok=True)

        # Create uid-specific temp directory if uid provided
        if self.uid:
            self.uid_temp_dir = os.path.join(self.base_temp_dir, f"templar_{self.uid}")
            os.makedirs(self.uid_temp_dir, exist_ok=True)
        else:
            self.uid_temp_dir = self.base_temp_dir

    def create_temp_file(self, prefix: str, suffix: str = ".pt") -> str:
        """Create a temporary file and return its path"""
        filename = f"{prefix}_{uuid.uuid4().hex}{suffix}"
        return os.path.join(self.uid_temp_dir, filename)

    def create_temp_dir(self, name: str) -> str:
        """Create a temporary directory and return its path"""
        dir_path = os.path.join(self.uid_temp_dir, name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def delete_file(self, file_path: str) -> bool:
        """Delete a file safely"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            tplr.logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def delete_directory(self, dir_path: str) -> bool:
        """Safely remove a directory and all its contents"""
        try:
            if not os.path.exists(dir_path):
                return True

            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dir_path)
            return True
        except Exception as e:
            tplr.logger.error(f"Error deleting directory {dir_path}: {e}")
            return False

    async def cleanup_local_data(
        self, uid: str, current_window: int, stale_retention: int
    ) -> None:
        """Clean up stale local data for a given uid."""
        user_dir = os.path.join(LOCAL_TMP_DIR, str(uid))
        if not os.path.exists(user_dir):
            return

        min_allowed_window = current_window - stale_retention
        for wdir in os.listdir(user_dir):
            if wdir.isdigit():
                w = int(wdir)
                if w < min_allowed_window:
                    old_path = os.path.join(user_dir, wdir)
                    tplr.logger.debug(f"Removing stale local directory: {old_path}")
                    try:
                        self.delete_directory(old_path)
                    except Exception as e:
                        tplr.logger.debug(
                            f"Error removing stale directory {old_path}: {e}"
                        )

    async def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary files older than max_age_hours"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for root, dirs, files in os.walk(self.uid_temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            tplr.logger.debug(f"Removing old temp file: {file_path}")
                            self.delete_file(file_path)
                    except Exception as e:
                        tplr.logger.debug(
                            f"Error checking/removing temp file {file_path}: {e}"
                        )

        except Exception as e:
            tplr.logger.error(f"Error during temp file cleanup: {e}")

    def get_local_storage_path(self, uid: str, window: int, filename: str) -> str:
        """Get the local storage path for a specific uid/window/filename"""
        return os.path.join(LOCAL_TMP_DIR, str(uid), str(window), filename)

    def ensure_directory_exists(self, path: str) -> None:
        """Ensure a directory exists, creating it if necessary"""
        os.makedirs(path, exist_ok=True)

    def get_temp_dir(self) -> str:
        """Get the temporary directory for this instance"""
        return self.uid_temp_dir
