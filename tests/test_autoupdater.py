# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors
# type: ignore

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import git
import pytest
import asyncio

# Import the AutoUpdate class
from templar.autoupdate import AutoUpdate


@pytest.mark.asyncio
async def test_autoupdate_version_check_no_update():
    """Test that no update occurs when local and remote versions are the same."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a temporary Git repository
        repo = git.Repo.init(temp_dir)

        # Create the initial __init__.py with version 1.0.0
        templar_init_path = os.path.join(temp_dir, "src", "templar")
        os.makedirs(templar_init_path)
        with open(os.path.join(templar_init_path, "__init__.py"), "w") as f:
            f.write('__version__ = "1.0.0"')

        repo.index.add(["src/templar/__init__.py"])
        repo.index.commit("Initial commit")

        # Mock the AutoUpdate class to use this temporary repo
        with patch("templar.autoupdate.git.Repo") as mock_repo:
            mock_repo.return_value = repo

            # Mock get_remote_version to return the same version asynchronously
            async def mock_get_remote_version(self):
                return "1.0.0"

            with patch.object(
                AutoUpdate, "get_remote_version", new=mock_get_remote_version
            ):
                # Create a mock 'templar' module
                mock_templar = MagicMock()
                mock_templar.__version__ = "1.0.0"

                # Inject the mock_templar into sys.modules
                with patch.dict("sys.modules", {"templar": mock_templar}):
                    autoupdater = AutoUpdate()
                    # Run the asynchronous check_version_updated method
                    updated = await autoupdater.check_version_updated()
                    assert (
                        not updated
                    ), "Should not attempt to update when versions are the same"


@pytest.mark.asyncio
async def test_autoupdate_version_check_update_available():
    """Test that an update is detected when remote version is higher."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a temporary Git repository
        repo = git.Repo.init(temp_dir)

        # Create the initial __init__.py with version 1.0.0
        templar_init_path = os.path.join(temp_dir, "src", "templar")
        os.makedirs(templar_init_path)
        with open(os.path.join(templar_init_path, "__init__.py"), "w") as f:
            f.write('__version__ = "1.0.0"')

        repo.index.add(["src/templar/__init__.py"])
        repo.index.commit("Initial commit")

        # Mock the AutoUpdate class to use this temporary repo
        with patch("templar.autoupdate.git.Repo") as mock_repo:
            mock_repo.return_value = repo

            # Mock get_remote_version to return a higher version asynchronously
            async def mock_get_remote_version(self):
                return "1.1.0"

            with patch.object(
                AutoUpdate, "get_remote_version", new=mock_get_remote_version
            ):
                autoupdater = AutoUpdate()
                # Run the asynchronous check_version_updated method
                updated = await autoupdater.check_version_updated()
                assert updated, "Should detect that an update is available"


def test_autoupdate_attempt_update():
    """Test that attempt_update pulls the latest changes when the repo is clean."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a temporary Git repository
        repo = git.Repo.init(temp_dir)

        # Create the initial commit
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write("Initial content")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Create a remote named 'origin'
        origin = repo.create_remote("origin", url="git@github.com:user/repo.git")

        # Mock the Repo object
        with patch("templar.autoupdate.git.Repo") as mock_repo:
            mock_repo.return_value = repo
            autoupdater = AutoUpdate()

            # Mock 'is_detached' property
            with patch.object(
                type(repo.head), "is_detached", new_callable=PropertyMock
            ) as mock_is_detached:
                mock_is_detached.return_value = False

                # Mock 'active_branch' property
                with patch.object(
                    type(repo), "active_branch", new_callable=PropertyMock
                ) as mock_active_branch:
                    mock_branch = MagicMock()
                    mock_branch.name = "main"
                    mock_active_branch.return_value = mock_branch

                    # Mock 'is_dirty' method
                    with patch.object(repo, "is_dirty", return_value=False):
                        # Mock 'pull' method on git.Remote
                        with patch.object(
                            git.Remote, "pull", return_value=None
                        ) as mock_pull:
                            success = autoupdater.attempt_update()
                            assert success, "Update should succeed when repo is clean"
                            mock_pull.assert_called_once()


def test_autoupdate_restart_app():
    """Test that the application restarts appropriately."""
    autoupdater = AutoUpdate(process_name="test_process")

    with patch("templar.autoupdate.os.execv") as mock_execv:
        with patch("templar.autoupdate.subprocess.check_call") as mock_check_call:
            # Mock PM2 environment
            with patch.dict(os.environ, {"PM2_HOME": "/path/to/pm2"}):
                # Catch the SystemExit exception
                with pytest.raises(SystemExit) as excinfo:
                    autoupdater.restart_app()
                assert excinfo.value.code == 1, "Should exit with code 1"
                mock_check_call.assert_called_with(["pm2", "restart", "test_process"])
                mock_execv.assert_not_called()


def test_autoupdate_attempt_package_update():
    """Test that the package update process is triggered."""
    autoupdater = AutoUpdate()

    with patch("templar.autoupdate.subprocess.check_call") as mock_check_call:
        autoupdater.attempt_package_update()

        mock_check_call.assert_called_with(
            ["uv", "sync", "--extra", "all"],
            timeout=300,
        )
