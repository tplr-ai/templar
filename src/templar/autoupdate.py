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

        local_version = self.get_local_version()
        if not local_version:
            logger.error("Failed to get local version, skipping version check")
            return False

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
        Attempts to update the local repository to match the remote.
        """
        if self.repo.head.is_detached:
            logger.error("Repository is in a detached HEAD state. Cannot update.")
            return False

        if self.repo.is_dirty(untracked_files=True):
            logger.error(
                "Repository has uncommitted changes or untracked files. Cannot update."
            )
            return False

        try:
            origin = self.repo.remote(name="origin")
            # Fetch latest changes from remote
            origin.fetch()
            # Get the current branch
            current_branch = self.repo.active_branch
            if current_branch.name != TARGET_BRANCH:
                logger.error(
                    f"Current branch ({current_branch.name}) is not the target branch ({TARGET_BRANCH}). Cannot update."
                )
                return False

            # Reset local branch to the remote branch
            remote_ref = f"origin/{TARGET_BRANCH}"
            logger.info(
                f"Resetting local branch '{current_branch.name}' to '{remote_ref}'"
            )
            self.repo.git.reset("--hard", remote_ref)
            logger.info("Successfully reset to the latest commit from remote.")

            # Verify that local and remote commits match
            local_commit = self.repo.commit(current_branch)
            remote_commit = self.repo.commit(remote_ref)
            if local_commit.hexsha != remote_commit.hexsha:
                logger.error(
                    "Local commit does not match remote commit after reset. Rolling back."
                )
                self.repo.git.reset("--hard", "HEAD@{1}")  # Reset to previous HEAD
                return False

            return True
        except git.exc.GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            # Rollback on failure
            self.repo.git.reset("--hard", "HEAD@{1}")
            return False
        except Exception as e:
            logger.exception("Failed to update repository.", exc_info=e)
            return False
        except git.exc.GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            return False
        except Exception as e:
            logger.exception("Failed to update repository.", exc_info=e)
            return False

    def handle_merge_conflicts(self):
        """
        Attempt to automatically resolve any merge conflicts that may have arisen.
        """
        try:
            self.repo.git.reset("--merge")
            origin = self.repo.remote(name="origin")
            current_branch = self.repo.active_branch.name
            origin.pull(current_branch)

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
                "Failed to resolve merge conflicts. Please manually pull and update.",
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

        if self.repo.head.is_detached or self.repo.active_branch.name != TARGET_BRANCH:
            logger.info("Not on the target branch, skipping auto-update")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("Checking for updates...")
            # Check if remote version is newer
            is_update_needed = loop.run_until_complete(self.check_version_updated())
            if not is_update_needed:
                logger.info("Local version is up to date. No updates needed.")
                return

            logger.info("Attempting auto update")
            # Attempt to update code
            update_applied = self.attempt_update()
            if not update_applied:
                logger.info("No updates were applied. Continuing without restart.")
                return

            # Now read the local version
            local_version = self.get_local_version()
            logger.info(f"Local version after update: {local_version}")

            # Synchronize dependencies
            self.attempt_package_update()

            # Clean up old versions from the bucket
            loop.run_until_complete(self.cleanup_old_versions())

            # Restart application
            logger.info("Attempting to restart the application...")
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
        pm2_name = self.get_pm2_process_name()
        if pm2_name:
            logger.info(
                f"Detected PM2 environment. Restarting PM2 process '{pm2_name}'..."
            )
            try:
                subprocess.run(["pm2", "restart", pm2_name], check=True)
                logger.info(f"Successfully restarted PM2 process '{pm2_name}'.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Failed to restart PM2 process '{pm2_name}': {e}")
                sys.exit(1)
        else:
            try:
                logger.info(
                    "PM2 process name not found. Performing regular restart using subprocess.Popen"
                )
                subprocess.Popen([sys.executable] + sys.argv)
                logger.info("New process started. Exiting current process.")
                sys.exit(0)
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

    def get_local_version(self):
        """
        Reads the local __version__ from the __init__.py file.
        """
        try:
            init_py_path = os.path.join(os.path.dirname(__file__), "__init__.py")
            with open(init_py_path, "r") as f:
                content = f.read()
            for line in content.split("\n"):
                if line.startswith("__version__"):
                    local_version = line.split("=")[1].strip().strip(" \"'")
                    return local_version
            logger.error("Could not find __version__ in local __init__.py")
            return None
        except Exception as e:
            logger.exception("Failed to read local version", exc_info=e)
            return None
