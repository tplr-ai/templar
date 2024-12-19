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

# Local imports
from .comms import delete_old_version_files
from .logging import logger


TARGET_BRANCH = "main"


class AutoUpdate(threading.Thread):
    """
    Automatic update utility for templar neurons.
    """

    def __init__(self, process_name=None, bucket_name=None):
        super().__init__()
        self.process_name = process_name
        self.bucket_name = bucket_name
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
            url = "https://raw.githubusercontent.com/tplr-ai/templar/main/src/templar/__init__.py"

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
        Attempt to update the repository by pulling the latest changes from the remote repository.
        """
        try:
            origin = self.repo.remotes.origin

            if self.repo.is_dirty(untracked_files=False):
                logger.error(
                    "Current changeset is dirty. Please commit changes, discard changes, or update manually."
                )
                return False
            try:
                logger.debug("Attempting to pull the latest changes...")
                origin.pull(
                    TARGET_BRANCH, kill_after_timeout=10, rebase=True
                )  # Invalid argument
                logger.debug("Successfully pulled the latest changes")
                return True
            except git.GitCommandError as e:
                logger.exception(
                    "Automatic update failed due to conflicts. Attempting to handle merge conflicts.",
                    exc_info=e,
                )
                return self.handle_merge_conflicts()
        except Exception as e:
            logger.exception(
                "Automatic update failed. Manually pull the latest changes and update.",
                exc_info=e,
            )

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
            # Optionally, specify the path to 'uv' if it's not in PATH

            subprocess.check_call(
                [uv_executable, "sync", "--extra", "all"],
                timeout=300,
            )
            logger.info("Successfully updated packages using 'uv sync --extra all'.")
        except Exception as e:
            logger.exception("Failed to synchronize dependencies with uv", exc_info=e)

    async def cleanup_old_versions(self):
        """
        Cleans up old version slices from the S3 bucket.
        """
        from templar import __version__

        bucket_name = self.bucket_name
        logger.info(f"Cleaning up old versions from bucket {bucket_name}")
        await delete_old_version_files(bucket_name, __version__)

    def try_update(self):
        """
        Automatic update entrypoint method.
        """
        if self.repo.head.is_detached or self.repo.active_branch.name != TARGET_BRANCH:
            logger.info("Not on the target branch, skipping auto-update")
            return

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

    def restart_app(self):
        """Restarts the current application appropriately based on the runtime environment."""
        logger.info("Restarting application...")
        # Check for PM2 environment
        if "PM2_HOME" in os.environ:
            if not self.process_name:
                logger.error("PM2 environment detected but process_name not provided")
                sys.exit(1)
            # PM2 will restart the process if we exit
            logger.info(
                f"Detected PM2 environment. Restarting process: {self.process_name}"
            )
            subprocess.check_call(["pm2", "restart", self.process_name])
            time.sleep(5)  # Give PM2 time to restart the process
            sys.exit(1)
        # TODO: Not tested
        # elif os.getenv("RUNNING_IN_DOCKER") == "true" or os.path.exists('/.dockerenv'):
        #     # In Docker, it's better to exit and let the orchestrator handle restarts
        #     logger.info("Detected Docker environment. Exiting for Docker to restart the container.")
        #     sys.exit(0)
        else:
            # Regular restart
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def run(self):
        """Thread run method to periodically check for updates."""
        while True:
            try:
                logger.info("Running autoupdate")
                self.try_update()
            except Exception as e:
                logger.exception("Exception during autoupdate check", exc_info=e)
            time.sleep(60)  # Sleep for 15 mins
