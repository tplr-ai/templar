import os
import pytest
import tempfile
import shutil
import time
import stat
from unittest.mock import patch
from src.tplr.storage.file_manager import FileManager, LOCAL_TMP_DIR


class TestFileManager:
    """
    Test cases for FileManager class covering:
    - Basic functionality
    - Edge cases and error conditions
    - Permission issues
    - Concurrency scenarios
    - Invalid inputs
    - File system edge cases
    """

    @pytest.fixture
    def temp_base_dir(self):
        """Create a temporary base directory for testing"""
        # Create isolated temp directory for each test
        temp_dir = tempfile.mkdtemp(prefix="test_file_manager_")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def file_manager(self, temp_base_dir):
        """Create FileManager instance with temp directory"""
        return FileManager(temp_base_dir)

    @pytest.fixture
    def file_manager_with_uid(self, temp_base_dir):
        """Create FileManager instance with uid"""
        return FileManager(temp_base_dir, uid=12345)

    # === __init__ method tests ===

    def test_init_creates_base_directory(self, temp_base_dir):
        """Test: FileManager creates base directory if it doesn't exist"""
        non_existent_dir = os.path.join(temp_base_dir, "non_existent")
        fm = FileManager(non_existent_dir)
        assert os.path.exists(non_existent_dir)
        assert fm.base_temp_dir == non_existent_dir

    def test_init_with_existing_directory(self, temp_base_dir):
        """Test: FileManager works with existing directory"""
        fm = FileManager(temp_base_dir)
        assert fm.base_temp_dir == temp_base_dir
        assert fm.uid is None
        assert fm.uid_temp_dir == temp_base_dir

    def test_init_with_uid_creates_uid_directory(self, temp_base_dir):
        """Test: FileManager creates uid-specific directory when uid provided"""
        uid = 54321
        fm = FileManager(temp_base_dir, uid=uid)
        expected_uid_dir = os.path.join(temp_base_dir, f"templar_{uid}")

        assert fm.uid == uid
        assert fm.uid_temp_dir == expected_uid_dir
        assert os.path.exists(expected_uid_dir)

    def test_init_with_zero_uid(self, temp_base_dir):
        """Test: FileManager handles uid=0 correctly"""
        fm = FileManager(temp_base_dir, uid=0)
        expected_uid_dir = os.path.join(temp_base_dir, "templar_0")
        assert fm.uid_temp_dir == expected_uid_dir
        assert os.path.exists(expected_uid_dir)

    def test_init_with_negative_uid(self, temp_base_dir):
        """Test: FileManager handles negative uid"""
        fm = FileManager(temp_base_dir, uid=-1)
        expected_uid_dir = os.path.join(temp_base_dir, "templar_-1")
        assert fm.uid_temp_dir == expected_uid_dir

    def test_init_with_very_long_path(self, temp_base_dir):
        """Test: FileManager handles very long paths"""
        # Create nested directory structure
        long_path = temp_base_dir
        for i in range(20):  # Create deeply nested path
            long_path = os.path.join(long_path, f"very_long_directory_name_{i}" * 5)

        FileManager(long_path)
        assert os.path.exists(long_path)

    def test_init_permission_denied(self, temp_base_dir):
        """Test: FileManager handles permission denied on directory creation"""
        # TODO: Test with read-only parent directory
        # This test would need root privileges or specific test environment setup
        pass

    def test_init_with_unicode_path(self, temp_base_dir):
        """Test: FileManager handles Unicode characters in path"""
        unicode_path = os.path.join(temp_base_dir, "测试目录_العربية_🚀")
        FileManager(unicode_path)
        assert os.path.exists(unicode_path)

    # === create_temp_file method tests ===

    def test_create_temp_file_basic(self, file_manager):
        """Test: Basic temp file creation"""
        file_path = file_manager.create_temp_file("test_prefix")

        assert file_path.startswith(file_manager.uid_temp_dir)
        assert "test_prefix" in os.path.basename(file_path)
        assert file_path.endswith(".pt")
        assert len(os.path.basename(file_path)) > len("test_prefix.pt")  # Has UUID

    def test_create_temp_file_custom_suffix(self, file_manager):
        """Test: Temp file creation with custom suffix"""
        file_path = file_manager.create_temp_file("test", suffix=".json")
        assert file_path.endswith(".json")

    def test_create_temp_file_empty_suffix(self, file_manager):
        """Test: Temp file creation with empty suffix"""
        file_path = file_manager.create_temp_file("test", suffix="")
        assert not file_path.endswith(".pt")
        assert "test_" in os.path.basename(file_path)

    def test_create_temp_file_uniqueness(self, file_manager):
        """Test: Multiple temp files have unique names"""
        files = [file_manager.create_temp_file("test") for _ in range(10)]
        assert len(set(files)) == 10  # All unique

    def test_create_temp_file_special_characters_prefix(self, file_manager):
        """Test: Temp file creation with special characters in prefix"""
        special_chars = "test-file_name.with@special#chars"
        file_path = file_manager.create_temp_file(special_chars)
        assert special_chars in os.path.basename(file_path)

    def test_create_temp_file_unicode_prefix(self, file_manager):
        """Test: Temp file creation with Unicode prefix"""
        unicode_prefix = "测试文件_العربية_🚀"
        file_path = file_manager.create_temp_file(unicode_prefix)
        assert unicode_prefix in os.path.basename(file_path)

    def test_create_temp_file_very_long_prefix(self, file_manager):
        """Test: Temp file creation with very long prefix"""
        long_prefix = "x" * 200
        file_path = file_manager.create_temp_file(long_prefix)
        # TODO: Check filesystem limitations
        assert long_prefix in os.path.basename(file_path)

    def test_create_temp_file_empty_prefix(self, file_manager):
        """Test: Temp file creation with empty prefix"""
        file_path = file_manager.create_temp_file("")
        assert os.path.basename(file_path).startswith("_")

    # === create_temp_dir method tests ===

    def test_create_temp_dir_basic(self, file_manager):
        """Test: Basic temp directory creation"""
        dir_path = file_manager.create_temp_dir("test_dir")

        assert os.path.exists(dir_path)
        assert os.path.isdir(dir_path)
        assert dir_path == os.path.join(file_manager.uid_temp_dir, "test_dir")

    def test_create_temp_dir_existing(self, file_manager):
        """Test: Creating temp directory that already exists"""
        dir_name = "existing_dir"
        dir_path1 = file_manager.create_temp_dir(dir_name)
        dir_path2 = file_manager.create_temp_dir(dir_name)

        assert dir_path1 == dir_path2
        assert os.path.exists(dir_path1)

    def test_create_temp_dir_nested_path(self, file_manager):
        """Test: Creating nested directory structure"""
        nested_name = "parent/child/grandchild"
        dir_path = file_manager.create_temp_dir(nested_name)

        assert os.path.exists(dir_path)
        assert os.path.isdir(dir_path)

    def test_create_temp_dir_special_characters(self, file_manager):
        """Test: Creating directory with special characters"""
        special_name = "test-dir_name.with@special#chars"
        dir_path = file_manager.create_temp_dir(special_name)
        assert os.path.exists(dir_path)

    def test_create_temp_dir_unicode(self, file_manager):
        """Test: Creating directory with Unicode name"""
        unicode_name = "测试目录_العربية_🚀"
        dir_path = file_manager.create_temp_dir(unicode_name)
        assert os.path.exists(dir_path)

    def test_create_temp_dir_permission_denied(self, file_manager):
        """Test: Creating directory when permission denied"""
        # TODO: Test with read-only parent directory
        pass

    # === delete_file method tests ===

    def test_delete_file_existing(self, file_manager, temp_base_dir):
        """Test: Deleting existing file"""
        test_file = os.path.join(temp_base_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = file_manager.delete_file(test_file)
        assert result is True
        assert not os.path.exists(test_file)

    def test_delete_file_non_existent(self, file_manager, temp_base_dir):
        """Test: Deleting non-existent file"""
        non_existent = os.path.join(temp_base_dir, "non_existent.txt")
        result = file_manager.delete_file(non_existent)
        assert result is False

    def test_delete_file_permission_denied(self, file_manager, temp_base_dir):
        """Test: Deleting file with permission denied"""
        test_file = os.path.join(temp_base_dir, "readonly_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Make file read-only
        os.chmod(test_file, stat.S_IRUSR)

        # On some systems, you can still delete read-only files
        # TODO: Create more robust permission test
        file_manager.delete_file(test_file)
        # Cleanup regardless
        try:
            os.chmod(test_file, stat.S_IWRITE | stat.S_IREAD)
            os.remove(test_file)
        except OSError:
            pass

    def test_delete_file_directory_instead_of_file(self, file_manager, temp_base_dir):
        """Test: Trying to delete directory with delete_file method"""
        test_dir = os.path.join(temp_base_dir, "test_dir")
        os.makedirs(test_dir)

        result = file_manager.delete_file(test_dir)
        assert result is False
        assert os.path.exists(test_dir)

    @patch("tplr.logger")
    def test_delete_file_logs_error(self, mock_logger, file_manager):
        """Test: delete_file logs errors appropriately"""
        with patch("os.remove", side_effect=Exception("Test error")):
            with patch("os.path.exists", return_value=True):
                result = file_manager.delete_file("some_file")
                assert result is False
                mock_logger.error.assert_called()

    def test_delete_file_symlink(self, file_manager, temp_base_dir):
        """Test: Deleting symbolic link"""
        test_file = os.path.join(temp_base_dir, "real_file.txt")
        test_link = os.path.join(temp_base_dir, "link_file.txt")

        with open(test_file, "w") as f:
            f.write("content")
        os.symlink(test_file, test_link)

        result = file_manager.delete_file(test_link)
        assert result is True
        assert not os.path.exists(test_link)
        assert os.path.exists(test_file)  # Original file should remain

    # === delete_directory method tests ===

    def test_delete_directory_basic(self, file_manager, temp_base_dir):
        """Test: Basic directory deletion"""
        test_dir = os.path.join(temp_base_dir, "test_delete_dir")
        os.makedirs(test_dir)

        # Add some files
        with open(os.path.join(test_dir, "file1.txt"), "w") as f:
            f.write("content1")

        result = file_manager.delete_directory(test_dir)
        assert result is True
        assert not os.path.exists(test_dir)

    def test_delete_directory_with_subdirs(self, file_manager, temp_base_dir):
        """Test: Deleting directory with subdirectories"""
        test_dir = os.path.join(temp_base_dir, "parent_dir")
        sub_dir = os.path.join(test_dir, "sub_dir")
        os.makedirs(sub_dir)

        # Add files in both directories
        with open(os.path.join(test_dir, "parent_file.txt"), "w") as f:
            f.write("parent content")
        with open(os.path.join(sub_dir, "sub_file.txt"), "w") as f:
            f.write("sub content")

        result = file_manager.delete_directory(test_dir)
        assert result is True
        assert not os.path.exists(test_dir)

    def test_delete_directory_non_existent(self, file_manager, temp_base_dir):
        """Test: Deleting non-existent directory"""
        non_existent = os.path.join(temp_base_dir, "non_existent_dir")
        result = file_manager.delete_directory(non_existent)
        assert result is True  # Method returns True for non-existent dirs

    def test_delete_directory_with_hidden_files(self, file_manager, temp_base_dir):
        """Test: Deleting directory with hidden files"""
        test_dir = os.path.join(temp_base_dir, "hidden_test_dir")
        os.makedirs(test_dir)

        # Create hidden file
        with open(os.path.join(test_dir, ".hidden_file"), "w") as f:
            f.write("hidden content")

        result = file_manager.delete_directory(test_dir)
        assert result is True
        assert not os.path.exists(test_dir)

    @patch("tplr.logger")
    def test_delete_directory_logs_error(self, mock_logger, file_manager):
        """Test: delete_directory logs errors appropriately"""
        with patch("os.path.exists", return_value=True):
            with patch("os.walk", side_effect=Exception("Test error")):
                result = file_manager.delete_directory("some_dir")
                assert result is False
                mock_logger.error.assert_called()

    def test_delete_directory_deep_nesting(self, file_manager, temp_base_dir):
        """Test: Deleting deeply nested directory structure"""
        # Create deep nesting
        current_dir = temp_base_dir
        for i in range(20):
            current_dir = os.path.join(current_dir, f"level_{i}")
        os.makedirs(current_dir)

        # Add file at deepest level
        with open(os.path.join(current_dir, "deep_file.txt"), "w") as f:
            f.write("deep content")

        # Delete from top level
        top_level = os.path.join(temp_base_dir, "level_0")
        result = file_manager.delete_directory(top_level)
        assert result is True
        assert not os.path.exists(top_level)

    # === cleanup_local_data method tests ===

    @pytest.mark.asyncio
    async def test_cleanup_local_data_basic(self, file_manager):
        """Test: Basic cleanup of stale local data"""
        uid = "test_uid"
        current_window = 100
        retention = 10

        # Create test directory structure
        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # Create old and new window directories
        old_window_dir = os.path.join(user_dir, "85")  # Should be deleted
        new_window_dir = os.path.join(user_dir, "95")  # Should be kept
        current_window_dir = os.path.join(user_dir, "100")  # Should be kept

        for wdir in [old_window_dir, new_window_dir, current_window_dir]:
            os.makedirs(wdir, exist_ok=True)
            with open(os.path.join(wdir, "test.txt"), "w") as f:
                f.write("test")

        await file_manager.cleanup_local_data(uid, current_window, retention)

        assert not os.path.exists(old_window_dir)
        assert os.path.exists(new_window_dir)
        assert os.path.exists(current_window_dir)

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cleanup_local_data_non_existent_user(self, file_manager):
        """Test: Cleanup with non-existent user directory"""
        uid = "non_existent_uid"
        await file_manager.cleanup_local_data(uid, 100, 10)
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_cleanup_local_data_non_numeric_dirs(self, file_manager):
        """Test: Cleanup ignores non-numeric directory names"""
        uid = "test_uid_2"
        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # Create directories with non-numeric names
        non_numeric_dirs = ["logs", "cache", "temp", "85abc", "not_a_number"]
        for dirname in non_numeric_dirs:
            dirpath = os.path.join(user_dir, dirname)
            os.makedirs(dirpath)
            with open(os.path.join(dirpath, "test.txt"), "w") as f:
                f.write("test")

        await file_manager.cleanup_local_data(uid, 100, 10)

        # All non-numeric directories should still exist
        for dirname in non_numeric_dirs:
            assert os.path.exists(os.path.join(user_dir, dirname))

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cleanup_local_data_boundary_conditions(self, file_manager):
        """Test: Cleanup boundary conditions for window retention"""
        uid = "boundary_test_uid"
        current_window = 50
        retention = 5

        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # min_allowed_window = 50 - 5 = 45
        # So windows < 45 should be deleted
        test_windows = [40, 44, 45, 46, 50, 55]  # 40,44 deleted; rest kept

        for window in test_windows:
            wdir = os.path.join(user_dir, str(window))
            os.makedirs(wdir)
            with open(os.path.join(wdir, "test.txt"), "w") as f:
                f.write("test")

        await file_manager.cleanup_local_data(uid, current_window, retention)

        # Check results
        assert not os.path.exists(os.path.join(user_dir, "40"))
        assert not os.path.exists(os.path.join(user_dir, "44"))
        assert os.path.exists(os.path.join(user_dir, "45"))
        assert os.path.exists(os.path.join(user_dir, "46"))
        assert os.path.exists(os.path.join(user_dir, "50"))
        assert os.path.exists(os.path.join(user_dir, "55"))

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cleanup_local_data_zero_retention(self, file_manager):
        """Test: Cleanup with zero retention period"""
        uid = "zero_retention_uid"
        current_window = 10
        retention = 0

        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # All windows < 10 should be deleted
        for window in [5, 8, 9, 10, 11]:
            wdir = os.path.join(user_dir, str(window))
            os.makedirs(wdir)

        await file_manager.cleanup_local_data(uid, current_window, retention)

        assert not os.path.exists(os.path.join(user_dir, "5"))
        assert not os.path.exists(os.path.join(user_dir, "8"))
        assert not os.path.exists(os.path.join(user_dir, "9"))
        assert os.path.exists(os.path.join(user_dir, "10"))
        assert os.path.exists(os.path.join(user_dir, "11"))

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cleanup_local_data_negative_windows(self, file_manager):
        """Test: Cleanup handles negative window numbers"""
        uid = "negative_window_uid"
        current_window = 0
        retention = 5

        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # Create directories with negative window numbers
        for window in [-10, -5, -1, 0, 1]:
            wdir = os.path.join(user_dir, str(window))
            os.makedirs(wdir, exist_ok=True)

        await file_manager.cleanup_local_data(uid, current_window, retention)

        # min_allowed = 0 - 5 = -5, so windows < -5 should be deleted
        assert not os.path.exists(os.path.join(user_dir, "-10"))
        assert os.path.exists(os.path.join(user_dir, "-5"))
        assert os.path.exists(os.path.join(user_dir, "-1"))
        assert os.path.exists(os.path.join(user_dir, "0"))
        assert os.path.exists(os.path.join(user_dir, "1"))

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_cleanup_local_data_error_handling(self, mock_logger, file_manager):
        """Test: Cleanup handles errors gracefully"""
        uid = "error_test_uid"
        user_dir = os.path.join(LOCAL_TMP_DIR, uid)
        os.makedirs(user_dir, exist_ok=True)

        # Create a directory we'll fail to delete
        problem_dir = os.path.join(user_dir, "1")
        os.makedirs(problem_dir)

        # Mock delete_directory to fail
        with patch.object(
            file_manager, "delete_directory", side_effect=Exception("Delete failed")
        ):
            await file_manager.cleanup_local_data(uid, 100, 10)
            # Should log the error but not crash
            mock_logger.debug.assert_called()

        # Cleanup
        shutil.rmtree(user_dir, ignore_errors=True)

    # === cleanup_temp_files method tests ===

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_basic(self, file_manager, temp_base_dir):
        """Test: Basic temp file cleanup by age"""
        # Create old and new files
        old_file = os.path.join(file_manager.uid_temp_dir, "old_file.txt")
        new_file = os.path.join(file_manager.uid_temp_dir, "new_file.txt")

        with open(old_file, "w") as f:
            f.write("old content")
        with open(new_file, "w") as f:
            f.write("new content")

        # Make old file appear old (25 hours ago)
        old_time = time.time() - (25 * 3600)
        os.utime(old_file, (old_time, old_time))

        await file_manager.cleanup_temp_files(max_age_hours=24)

        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_no_old_files(self, file_manager):
        """Test: Cleanup when no old files exist"""
        # Create only new files
        new_file = os.path.join(file_manager.uid_temp_dir, "new_file.txt")
        with open(new_file, "w") as f:
            f.write("new content")

        await file_manager.cleanup_temp_files(max_age_hours=1)
        assert os.path.exists(new_file)

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_empty_directory(self, file_manager):
        """Test: Cleanup in empty directory"""
        await file_manager.cleanup_temp_files(max_age_hours=24)
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_nested_directories(self, file_manager):
        """Test: Cleanup files in nested directory structure"""
        # Create nested structure
        sub_dir = os.path.join(file_manager.uid_temp_dir, "subdir")
        os.makedirs(sub_dir)

        old_nested_file = os.path.join(sub_dir, "old_nested.txt")
        new_nested_file = os.path.join(sub_dir, "new_nested.txt")

        with open(old_nested_file, "w") as f:
            f.write("old nested")
        with open(new_nested_file, "w") as f:
            f.write("new nested")

        # Make one file old
        old_time = time.time() - (25 * 3600)
        os.utime(old_nested_file, (old_time, old_time))

        await file_manager.cleanup_temp_files(max_age_hours=24)

        assert not os.path.exists(old_nested_file)
        assert os.path.exists(new_nested_file)

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_custom_max_age(self, file_manager):
        """Test: Cleanup with different max_age_hours values"""
        test_file = os.path.join(file_manager.uid_temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Make file 2 hours old
        old_time = time.time() - (2 * 3600)
        os.utime(test_file, (old_time, old_time))

        # Should not be deleted with max_age=3
        await file_manager.cleanup_temp_files(max_age_hours=3)
        assert os.path.exists(test_file)

        # Should be deleted with max_age=1
        await file_manager.cleanup_temp_files(max_age_hours=1)
        assert not os.path.exists(test_file)

    @pytest.mark.asyncio
    async def test_cleanup_temp_files_zero_max_age(self, file_manager):
        """Test: Cleanup with zero max age (delete all files)"""
        test_file = os.path.join(file_manager.uid_temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        await file_manager.cleanup_temp_files(max_age_hours=0)
        assert not os.path.exists(test_file)

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_cleanup_temp_files_error_handling(self, mock_logger, file_manager):
        """Test: Cleanup handles file access errors gracefully"""
        # Create test file
        test_file = os.path.join(file_manager.uid_temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Mock os.walk to raise exception
        with patch("os.walk", side_effect=Exception("Walk failed")):
            await file_manager.cleanup_temp_files()
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_cleanup_temp_files_individual_file_errors(
        self, mock_logger, file_manager
    ):
        """Test: Cleanup handles individual file errors gracefully"""
        test_file = os.path.join(file_manager.uid_temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Mock getmtime to fail for specific file
        with patch("os.path.getmtime", side_effect=Exception("Access denied")):
            await file_manager.cleanup_temp_files()
            mock_logger.debug.assert_called()

    # === get_local_storage_path method tests ===

    def test_get_local_storage_path_basic(self, file_manager):
        """Test: Basic local storage path generation"""
        uid = "12345"
        window = 100
        filename = "test_file.pt"

        expected_path = os.path.join(LOCAL_TMP_DIR, uid, str(window), filename)
        result = file_manager.get_local_storage_path(uid, window, filename)

        assert result == expected_path

    def test_get_local_storage_path_special_characters(self, file_manager):
        """Test: Local storage path with special characters"""
        uid = "user@domain.com"
        window = -5
        filename = "file with spaces & symbols!.json"

        expected_path = os.path.join(LOCAL_TMP_DIR, uid, str(window), filename)
        result = file_manager.get_local_storage_path(uid, window, filename)

        assert result == expected_path

    def test_get_local_storage_path_unicode(self, file_manager):
        """Test: Local storage path with Unicode characters"""
        uid = "用户_123"
        window = 42
        filename = "测试文件.txt"

        expected_path = os.path.join(LOCAL_TMP_DIR, uid, str(window), filename)
        result = file_manager.get_local_storage_path(uid, window, filename)

        assert result == expected_path

    def test_get_local_storage_path_edge_cases(self, file_manager):
        """Test: Local storage path edge cases"""
        # Empty strings
        result1 = file_manager.get_local_storage_path("", 0, "")
        assert result1 == os.path.join(LOCAL_TMP_DIR, "", "0", "")

        # Very large window number
        result2 = file_manager.get_local_storage_path("uid", 999999999, "file.txt")
        assert "999999999" in result2

    # === ensure_directory_exists method tests ===

    def test_ensure_directory_exists_new_directory(self, file_manager, temp_base_dir):
        """Test: Creating new directory"""
        new_dir = os.path.join(temp_base_dir, "new_directory")
        file_manager.ensure_directory_exists(new_dir)

        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)

    def test_ensure_directory_exists_existing_directory(
        self, file_manager, temp_base_dir
    ):
        """Test: Ensuring existing directory exists"""
        existing_dir = os.path.join(temp_base_dir, "existing_dir")
        os.makedirs(existing_dir)

        # Should not raise error
        file_manager.ensure_directory_exists(existing_dir)
        assert os.path.exists(existing_dir)

    def test_ensure_directory_exists_nested_path(self, file_manager, temp_base_dir):
        """Test: Creating nested directory structure"""
        nested_path = os.path.join(temp_base_dir, "level1", "level2", "level3")
        file_manager.ensure_directory_exists(nested_path)

        assert os.path.exists(nested_path)
        assert os.path.isdir(nested_path)

    def test_ensure_directory_exists_unicode_path(self, file_manager, temp_base_dir):
        """Test: Creating directory with Unicode name"""
        unicode_dir = os.path.join(temp_base_dir, "unicode_测试_🚀")
        file_manager.ensure_directory_exists(unicode_dir)

        assert os.path.exists(unicode_dir)

    # === get_temp_dir method tests ===

    def test_get_temp_dir_no_uid(self, file_manager):
        """Test: Get temp directory when no uid provided"""
        result = file_manager.get_temp_dir()
        assert result == file_manager.base_temp_dir
        assert result == file_manager.uid_temp_dir

    def test_get_temp_dir_with_uid(self, file_manager_with_uid):
        """Test: Get temp directory when uid provided"""
        result = file_manager_with_uid.get_temp_dir()
        expected = os.path.join(file_manager_with_uid.base_temp_dir, "templar_12345")
        assert result == expected
        assert result == file_manager_with_uid.uid_temp_dir

    # === Integration and concurrent access tests ===

    def test_concurrent_file_operations(self, file_manager):
        """Test: Multiple file operations don't interfere"""
        # Create multiple temp files and directories simultaneously
        files = []
        dirs = []

        for i in range(10):
            file_path = file_manager.create_temp_file(f"concurrent_{i}")
            dir_path = file_manager.create_temp_dir(f"concurrent_dir_{i}")
            files.append(file_path)
            dirs.append(dir_path)

        # Verify all were created uniquely
        assert len(set(files)) == 10
        assert len(set(dirs)) == 10

        # Verify all directories exist
        for dir_path in dirs:
            assert os.path.exists(dir_path)

    def test_file_manager_isolation(self, temp_base_dir):
        """Test: Multiple FileManager instances are properly isolated"""
        fm1 = FileManager(temp_base_dir, uid=1)
        fm2 = FileManager(temp_base_dir, uid=2)

        # Create files in each
        file1 = fm1.create_temp_file("test")
        file2 = fm2.create_temp_file("test")

        # Should be in different directories
        assert os.path.dirname(file1) != os.path.dirname(file2)
        assert "templar_1" in file1
        assert "templar_2" in file2

    @pytest.mark.asyncio
    async def test_cleanup_operations_dont_interfere(self, file_manager, temp_base_dir):
        """Test: Cleanup operations don't interfere with active file operations"""
        # Create some files
        active_file = file_manager.create_temp_file("active")
        with open(active_file, "w") as f:
            f.write("active content")

        # Create old file for cleanup
        old_file = os.path.join(file_manager.uid_temp_dir, "old_file.txt")
        with open(old_file, "w") as f:
            f.write("old content")

        # Make old file actually old
        old_time = time.time() - (25 * 3600)
        os.utime(old_file, (old_time, old_time))

        # Run cleanup
        await file_manager.cleanup_temp_files(max_age_hours=24)

        # Active file should remain, old file should be gone
        assert os.path.exists(active_file)
        assert not os.path.exists(old_file)

    # === Performance and stress tests ===

    def test_large_number_of_files(self, file_manager):
        """Test: Creating large number of temp files"""
        # TODO: Stress test with many files
        # This test might be resource intensive
        files = []
        for i in range(100):
            file_path = file_manager.create_temp_file(f"stress_{i}")
            files.append(file_path)

        # All should be unique
        assert len(set(files)) == 100

    @pytest.mark.asyncio
    async def test_cleanup_large_directory_structure(self, file_manager):
        """Test: Cleanup with large directory structures"""
        # TODO: Create large nested structure and test cleanup performance
        # This test might be resource intensive
        pass

    # === Parametrized tests for various input types ===

    @pytest.mark.parametrize(
        "uid_input",
        [None, 0, 1, -1, 999999, "string_uid", "special@chars", "unicode_用户"],
    )
    def test_file_manager_with_various_uid_types(self, temp_base_dir, uid_input):
        """Test: FileManager with various uid input types"""
        fm = FileManager(temp_base_dir, uid=uid_input)

        if uid_input is None:
            assert fm.uid_temp_dir == temp_base_dir
        else:
            expected_dir = os.path.join(temp_base_dir, f"templar_{uid_input}")
            assert fm.uid_temp_dir == expected_dir
            assert os.path.exists(expected_dir)

    @pytest.mark.parametrize(
        "prefix,suffix",
        [
            ("test", ".pt"),
            ("", ""),
            ("special@chars", ".json"),
            ("unicode_测试", ".txt"),
            ("very_long_" + "x" * 100, ".log"),
            ("with spaces", ".data"),
            ("with.dots.and-dashes_underscores", ".bin"),
        ],
    )
    def test_create_temp_file_various_inputs(self, file_manager, prefix, suffix):
        """Test: create_temp_file with various prefix/suffix combinations"""
        file_path = file_manager.create_temp_file(prefix, suffix)

        assert prefix in os.path.basename(file_path) or prefix == ""
        assert file_path.endswith(suffix) if suffix else True
        assert file_path.startswith(file_manager.uid_temp_dir)

    @pytest.mark.parametrize("hours", [0, 1, 24, 168, 8760, 0.5, 100000])
    @pytest.mark.asyncio
    async def test_cleanup_temp_files_various_ages(self, file_manager, hours):
        """Test: cleanup_temp_files with various max_age_hours values"""
        # Create test file
        test_file = os.path.join(file_manager.uid_temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Make file appear old (but leave some buffer for edge cases)
        # Use 8759 hours to avoid floating point precision issues with 8760
        old_hours = 8759
        very_old_time = time.time() - (old_hours * 3600)
        os.utime(test_file, (very_old_time, very_old_time))

        await file_manager.cleanup_temp_files(max_age_hours=hours)

        # File should be deleted when file_age > max_age_seconds
        # File is 8759 hours old
        if hours < old_hours:
            # File is older than max_age, so it should be deleted
            assert not os.path.exists(test_file)
        else:
            # File is not older than max_age, so it should remain
            assert os.path.exists(test_file)

    # === Mock and patch tests for error conditions ===

    @patch("os.makedirs")
    def test_init_makedirs_failure(self, mock_makedirs, temp_base_dir):
        """Test: FileManager handles makedirs failure during init"""
        mock_makedirs.side_effect = OSError("Permission denied")

        with pytest.raises(OSError):
            FileManager(temp_base_dir)

    @patch("uuid.uuid4")
    def test_create_temp_file_uuid_failure(self, mock_uuid, file_manager):
        """Test: create_temp_file handles UUID generation failure"""
        mock_uuid.side_effect = Exception("UUID generation failed")

        with pytest.raises(Exception):
            file_manager.create_temp_file("test")

    @patch("os.walk")
    def test_delete_directory_walk_failure(self, mock_walk, file_manager):
        """Test: delete_directory handles os.walk failure"""
        mock_walk.side_effect = Exception("Walk failed")

        with patch("os.path.exists", return_value=True):
            result = file_manager.delete_directory("some_dir")
            assert result is False
