# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import git
import pytest
import asyncio
import subprocess
import json

# Import the AutoUpdate class
from templar.autoupdate import AutoUpdate, TARGET_BRANCH


@pytest.mark.asyncio
async def test_autoupdate_cleanup_old_versions():
    """Test cleanup of old versions."""
    autoupdater = AutoUpdate()

    with patch("templar.autoupdate.BUCKET_SECRETS", {"bucket_name": "test-bucket"}):
        with patch("templar.autoupdate.delete_old_version_files") as mock_cleanup:
            # Mock the templar.__version__ to a known value
            with patch("templar.__version__", "0.1.29"):
                await autoupdater.cleanup_old_versions()
                mock_cleanup.assert_called_with("test-bucket", "0.1.29")


def test_autoupdate_run_method():
    """Test the run method's loop."""
    autoupdater = AutoUpdate()

    with patch.object(autoupdater, "try_update") as mock_try_update:
        # Create a side effect function to stop the loop after first iteration
        def side_effect_sleep(duration):
            raise KeyboardInterrupt  # Breaking the infinite loop

        with patch("time.sleep", side_effect=side_effect_sleep) as mock_sleep:
            try:
                autoupdater.run()
            except KeyboardInterrupt:
                pass  # Expected to break the loop

            mock_try_update.assert_called_once()
            mock_sleep.assert_called_once_with(60)


def test_autoupdate_restart_app_pm2_no_process_name_failure():
    """Test that the application handles failure to get PM2 process name."""
    autoupdater = AutoUpdate()

    # Mock get_pm2_process_name to return None
    with patch.object(autoupdater, "get_pm2_process_name", return_value=None):
        with patch("templar.autoupdate.logger") as mock_logger:
            with patch("templar.autoupdate.sys.exit", side_effect=SystemExit):
                # Mock PM2 environment
                with patch.dict(os.environ, {"PM2_HOME": "/path/to/pm2"}):
                    with pytest.raises(SystemExit):
                        autoupdater.restart_app()
                    # Check the correct info message is logged
                    mock_logger.info.assert_any_call(
                        "PM2 process name not found. Performing regular restart using subprocess.Popen"
                    )


def test_autoupdate_restart_app_pm2_success():
    """Test that the application restarts successfully in a PM2 environment."""
    autoupdater = AutoUpdate()

    mock_pm2_process_name = "test_process"
    # Mock get_pm2_process_name to return a test process name
    with patch.object(
        autoupdater, "get_pm2_process_name", return_value=mock_pm2_process_name
    ):
        with patch("templar.autoupdate.subprocess.run") as mock_run:
            with patch("templar.autoupdate.sys.exit", side_effect=SystemExit):
                # Mock PM2 environment
                with patch.dict(os.environ, {"PM2_HOME": "/path/to/pm2"}):
                    with pytest.raises(SystemExit):
                        autoupdater.restart_app()
                    mock_run.assert_called_with(
                        ["pm2", "restart", mock_pm2_process_name], check=True
                    )


@pytest.mark.asyncio
async def test_autoupdate_check_version_updated():
    """Test that check_version_updated works correctly when update is needed."""
    autoupdater = AutoUpdate()

    # Mock get_remote_version to return a higher version
    async def mock_get_remote_version(self):
        return "0.2.0"

    with patch.object(AutoUpdate, "get_remote_version", new=mock_get_remote_version):
        # Mock templar.__version__ to a lower version
        with patch("templar.__version__", "0.1.0"):
            is_updated = await autoupdater.check_version_updated()
            assert is_updated is True


@pytest.mark.asyncio
async def test_autoupdate_check_version_not_updated():
    """Test that check_version_updated works correctly when no update is needed."""
    autoupdater = AutoUpdate()

    # Mock get_remote_version to return the same version
    async def mock_get_remote_version(self):
        return "0.1.0"

    with patch.object(AutoUpdate, "get_remote_version", new=mock_get_remote_version):
        # Mock templar.__version__ to the same version
        with patch("templar.__version__", "0.1.0"):
            is_updated = await autoupdater.check_version_updated()
            assert is_updated is False


def test_autoupdate_attempt_update_success():
    """Test that attempt_update succeeds when repo is clean."""
    autoupdater = AutoUpdate()

    # Patch 'is_detached' property
    with patch.object(
        type(autoupdater.repo.head), "is_detached", new_callable=PropertyMock
    ) as mock_is_detached:
        mock_is_detached.return_value = False

        # Patch 'active_branch' property to return a mock branch
        mock_branch = MagicMock()
        mock_branch.name = TARGET_BRANCH
        with patch.object(
            type(autoupdater.repo), "active_branch", new_callable=PropertyMock
        ) as mock_active_branch:
            mock_active_branch.return_value = mock_branch

            with patch.object(autoupdater.repo, "is_dirty", return_value=False):
                # Mock the 'remote' method to return a mock 'origin'
                mock_origin = MagicMock()
                mock_origin.fetch.return_value = None  # Simulate successful fetch
                with patch.object(autoupdater.repo, "remote", return_value=mock_origin):
                    # Mock 'git' attribute
                    mock_git = MagicMock()
                    autoupdater.repo.git = mock_git

                    # Mock commits for local and remote to match
                    mock_commit = MagicMock()
                    mock_commit.hexsha = "abcdef"
                    with patch.object(
                        autoupdater.repo, "commit", return_value=mock_commit
                    ):
                        result = autoupdater.attempt_update()
                        mock_origin.fetch.assert_called_once()
                        mock_git.reset.assert_called_with(
                            "--hard", f"origin/{TARGET_BRANCH}"
                        )
                        assert result is True


def test_autoupdate_attempt_update_dirty_repo():
    """Test that attempt_update fails when repo is dirty."""
    autoupdater = AutoUpdate()

    with patch.object(autoupdater.repo, "is_dirty", return_value=True):
        with patch("templar.autoupdate.logger") as mock_logger:
            result = autoupdater.attempt_update()
            mock_logger.error.assert_called_with(
                "Repository has uncommitted changes or untracked files. Cannot update."
            )
            assert result is False


def test_autoupdate_attempt_update_pull_failure():
    """Test that attempt_update handles fetch failure."""
    autoupdater = AutoUpdate()

    # Patch 'is_detached' property
    with patch.object(
        type(autoupdater.repo.head), "is_detached", new_callable=PropertyMock
    ) as mock_is_detached:
        mock_is_detached.return_value = False

        # Patch 'active_branch' property
        mock_branch = MagicMock()
        mock_branch.name = TARGET_BRANCH
        with patch.object(
            type(autoupdater.repo), "active_branch", new_callable=PropertyMock
        ) as mock_active_branch:
            mock_active_branch.return_value = mock_branch

            with patch.object(autoupdater.repo, "is_dirty", return_value=False):
                mock_origin = MagicMock()
                mock_origin.fetch.side_effect = git.exc.GitCommandError(
                    "fetch", "Failed to fetch"
                )
                with patch.object(autoupdater.repo, "remote", return_value=mock_origin):
                    # Mock 'git' attribute
                    mock_git = MagicMock()
                    autoupdater.repo.git = mock_git

                    with patch("templar.autoupdate.logger") as mock_logger:
                        result = autoupdater.attempt_update()
                        # Confirm that an error was logged
                        mock_logger.error.assert_called_once()
                        error_msg = mock_logger.error.call_args[0][0]
                        assert error_msg.startswith("Git command failed:")
                        assert "Failed to fetch" in error_msg
                        assert result is False


def test_autoupdate_attempt_package_update():
    """Test that attempt_package_update calls the correct subprocess."""
    autoupdater = AutoUpdate()

    with patch("templar.autoupdate.subprocess.check_call") as mock_check_call:
        autoupdater.attempt_package_update()
        mock_check_call.assert_called_with(
            ["uv", "sync", "--extra", "all"],
            timeout=300,
        )


def test_autoupdate_attempt_package_update_failure():
    """Test that package update handles failures gracefully."""
    autoupdater = AutoUpdate()

    with patch(
        "templar.autoupdate.subprocess.check_call",
        side_effect=subprocess.CalledProcessError(1, "uv"),
    ):
        with patch("templar.autoupdate.logger") as mock_logger:
            autoupdater.attempt_package_update()
            mock_logger.exception.assert_called_once()


def test_autoupdate_get_pm2_process_name():
    """Test getting the PM2 process name."""
    autoupdater = AutoUpdate()

    # Mock os.getpid()
    with patch("os.getpid", return_value=12345):
        # Mock subprocess.run()
        mock_pm2_output = json.dumps(
            [
                {"name": "test_process", "pid": 12345},
                {"name": "other_process", "pid": 67890},
            ]
        )

        mock_completed_process = subprocess.CompletedProcess(
            args=["pm2", "jlist"],
            returncode=0,
            stdout=mock_pm2_output,
        )

        with patch("subprocess.run", return_value=mock_completed_process):
            process_name = autoupdater.get_pm2_process_name()
            assert process_name == "test_process"
