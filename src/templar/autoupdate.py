# Global imports
import asyncio
import aiohttp
from packaging import version
import git
import os
import subprocess
import sys
import threading
import time
import json

# Local imports
from .config import BUCKET_SECRETS
from .comms import delete_old_version_files
from .logging import logger


TARGET_BRANCH = "main"


class AutoUpdate(threading.Thread):
    """
    Automatic update utility for templar neurons.
    """

    def __init__(self):
        super().__init__()
        self.daemon = True  # Ensure thread exits when main program exits
        try:
            self.repo = git.Repo(search_parent_directories=True)
        except Exception as e:
            logger.exception("Failed to initialize the repository", exc_info=e)
            sys.exit(1)  # Terminate the thread/application

    async def get_remote_version(self):
        """
        Asynchronously fetch the remote version string from a remote HTTP endpoint.
        """
        try:
            # url = "https://raw.githubusercontent.com/tplr-ai/templar/main/src/templar/__init__.py"
            url = "https://raw.githubusercontent.com/distributedstatemachine/templar/main/src/templar/__init__.py"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    content = await response.text()

            for line in content.split("\n"):
                if line.startswith("__version__"):
                    version_info = line.split("=")[1].strip().strip(" \"'")
                    return version_info

            logger.error("Version string not found in remote __init__.py")
            return None

        except Exception as e:
            logger.exception(
                "Failed to get remote version for version check", exc_info=e
            )
            return None

    async def check_version_updated(self):
        """
        Asynchronously compares local and remote versions and returns True if the remote version is higher.
        """
        remote_version = await self.get_remote_version()
        if not remote_version:
            logger.error("Failed to get remote version, skipping version check")
            return False

        # Reload the version from __init__.py to get the latest version after updates
        try:
            import templar
            from importlib import reload

            reload(templar)
            local_version = templar.__version__
        except Exception as e:
            logger.error(f"Failed to reload local version: {e}")
            # Fallback to imported version
            local_version = templar.__version__

        local_version_obj = version.parse(local_version)
        remote_version_obj = version.parse(remote_version)
        logger.info(
            f"Version check - remote_version: {remote_version}, local_version: {local_version}"
        )

        if remote_version_obj > local_version_obj:
            logger.info(
                f"Remote version ({remote_version}) is higher "
                f"than local version ({local_version}), automatically updating..."
            )
            return True
        return False

    def attempt_update(self):
        """
        Attempts to pull the latest changes from the remote repository.
        """
        if self.repo.is_dirty():
            logger.error(
                "Current changeset is dirty. Please commit changes, discard changes, or update manually."
            )
            return False
        try:
            origin = self.repo.remote(name="origin")
            origin.pull(TARGET_BRANCH, ff_only=True)
            logger.info("Successfully pulled latest changes from remote")
            return True
        except git.exc.GitCommandError as e:
            # Handle merge conflicts
            logger.error(
                "Automatic update failed due to conflicts. Attempting to handle merge conflicts.",
                exc_info=e,
            )
            try:
                self.handle_merge_conflicts()
                return True
            except Exception as e:
                logger.exception(
                    "Failed to resolve merge conflicts, automatic update cannot proceed. Please manually pull and update.",
                    exc_info=e,
                )
                return False
        except Exception as e:
            logger.exception("Failed to pull latest changes from remote", exc_info=e)
            return False

    def handle_merge_conflicts(self):
        """
        Attempt to automatically resolve any merge conflicts that may have arisen.
        """
        try:
            self.repo.git.reset("--merge")
            origin = self.repo.remotes.origin
            current_branch = self.repo.active_branch
            origin.pull(current_branch.name)

            for item in self.repo.index.diff(None):
                file_path = item.a_path
                logger.info(f"Resolving conflict in file: {file_path}")
                self.repo.git.checkout("--theirs", file_path)
            self.repo.index.commit("Resolved merge conflicts automatically")
            logger.info("Merge conflicts resolved, repository updated to remote state.")
            logger.info("✅ Successfully updated")
            return True
        except git.GitCommandError as e:
            logger.exception(
                "Failed to resolve merge conflicts, automatic update cannot proceed. Please manually pull and update.",
                exc_info=e,
            )
            return False

    def attempt_package_update(self):
        """
        Synchronize dependencies using 'uv sync --extra all'.
        """
        logger.info("Attempting to update packages using 'uv sync --extra all'...")

        try:
            uv_executable = "uv"
            # TODO: Allow specifying the path to 'uv' if it's not in PATH

            subprocess.check_call(
                [uv_executable, "sync", "--extra", "all"],
                timeout=300,
            )
            logger.info("Successfully updated packages using 'uv sync --extra all'.")
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to synchronize dependencies with uv", exc_info=e)
        except FileNotFoundError:
            logger.error(
                "uv executable not found. Please ensure 'uv' is installed and in PATH."
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during package synchronization", exc_info=e
            )

    async def cleanup_old_versions(self):
        """
        Cleans up old version slices from the S3 bucket.
        """
        from templar import __version__

        logger.info(
            f"Cleaning up old versions from bucket {BUCKET_SECRETS['bucket_name']}"
        )
        await delete_old_version_files(BUCKET_SECRETS["bucket_name"], __version__)

    def try_update(self):
        """
        Automatic update entrypoint method.
        """
        # if self.repo.head.is_detached or self.repo.active_branch.name != TARGET_BRANCH:
        #     logger.info("Not on the target branch, skipping auto-update")
        #     return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            is_updated = loop.run_until_complete(self.check_version_updated())
            if not is_updated:
                return

            if not self.attempt_update():
                return

            # Synchronize dependencies
            self.attempt_package_update()

            # Clean up old versions from the bucket
            loop.run_until_complete(self.cleanup_old_versions())

            # Restart application
            self.restart_app()
        except Exception as e:
            logger.exception("Exception during autoupdate process", exc_info=e)
        finally:
            loop.close()

    def get_pm2_process_name(self):
        """
        Attempt to find the current process's PM2 name by using `pm2 jlist` and matching the current PID.
        """
        current_pid = os.getpid()
        try:
            result = subprocess.run(
                ["pm2", "jlist"], check=True, capture_output=True, text=True
            )
            pm2_data = json.loads(result.stdout)
        except Exception as e:
            logger.error(f"Error running `pm2 jlist`: {e}")
            return None

        for proc in pm2_data:
            if proc.get("pid") == current_pid:
                return proc.get("name")

        return None

    def restart_app(self):
        """Restarts the current application appropriately based on the runtime environment."""
        logger.info("Restarting application...")
        # Check for PM2 environment
        if "PM2_HOME" in os.environ:
            pm2_name = self.get_pm2_process_name()
            if not pm2_name:
                logger.warning("Could not determine PM2 process name. Restart aborted.")
                sys.exit(1)
            # PM2 will restart the process if we exit
            logger.info(f"Detected PM2 environment. Restarting process '{pm2_name}'")
            try:
                subprocess.check_call(["pm2", "restart", pm2_name])
                time.sleep(5)  # Give PM2 time to restart the process
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                logger.exception("PM2 restart failed.", exc_info=e)
                sys.exit(1)
        else:
            # Regular restart
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                logger.exception("Failed to restart application.", exc_info=e)
                sys.exit(1)

    def run(self):
        """Thread run method to periodically check for updates."""
        while True:
            try:
                logger.info("Running autoupdate")
                self.try_update()
            except Exception as e:
                logger.exception("Exception during autoupdate check", exc_info=e)
            time.sleep(60)
