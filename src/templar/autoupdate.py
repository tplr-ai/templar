import os
import subprocess
import sys
import git
import hashlib
from packaging import version
from typing import Optional

# Import local modules
from . import logger

# Import the local version
from .__init__ import __version__

TARGET_BRANCH = "main"


class AutoUpdate:
    """
    Automatic update utility for templar neurons.
    """

    def __init__(self):
        try:
            self.repo = git.Repo(search_parent_directories=True)
            self.update_requirements = False
        except Exception as e:
            logger.exception("Failed to initialize the repository", exc_info=e)

    def get_remote_version(self) -> Optional[str]:
        """
        Fetch the remote version string from the src/templar/__init__.py file in the repository.
        """
        try:
            # Perform a git fetch to ensure we have the latest remote information
            self.repo.remotes.origin.fetch(timeout=5)

            # Check if the requirements.txt file has changed
            local_requirements_path = os.path.join(
                self.repo.working_tree_dir, "requirements.txt"
            )
            with open(local_requirements_path, "r", encoding="utf-8") as file:
                local_requirements_hash = hashlib.sha256(
                    file.read().encode("utf-8")
                ).hexdigest()

            requirements_blob = (
                self.repo.remote().refs[TARGET_BRANCH].commit.tree / "requirements.txt"
            )
            remote_requirements_content = requirements_blob.data_stream.read().decode(
                "utf-8"
            )
            remote_requirements_hash = hashlib.sha256(
                remote_requirements_content.encode("utf-8")
            ).hexdigest()
            self.update_requirements = (
                local_requirements_hash != remote_requirements_hash
            )

            # Get version number from remote __init__.py
            init_blob = (
                self.repo.remote().refs[TARGET_BRANCH].commit.tree
                / "src"
                / "templar"
                / "__init__.py"
            )
            lines = init_blob.data_stream.read().decode("utf-8").split("\n")

            for line in lines:
                if line.startswith("__version__"):
                    version_info = line.split("=")[1].strip().strip(' "')
                    return version_info
        except Exception as e:
            logger.exception("Failed to get remote version for version check", exc_info=e)
            return None

    def check_version_updated(self):
        """
        Compares local and remote versions and returns True if the remote version is higher.
        """
        remote_version = self.get_remote_version()
        if not remote_version:
            logger.error("Failed to get remote version, skipping version check")
            return False

        local_version = __version__

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
                logger.info("Attempting to pull the latest changes...")
                origin.pull(timeout=10)
                logger.info("Successfully pulled the latest changes")
                return True
            except git.GitCommandError as e:
                logger.exception(
                    "Automatic update failed due to conflicts. Attempting to handle merge conflicts...",
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
            logger.info(
                "Merge conflicts resolved, repository updated to remote state."
            )
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
        Attempt to update the packages by installing the requirements from the requirements.txt file.
        """
        logger.info("Attempting to update packages...")

        try:
            repo = git.Repo(search_parent_directories=True)
            repo_path: str = repo.working_tree_dir or ""

            requirements_path = os.path.join(repo_path, "requirements.txt")

            python_executable = sys.executable
            subprocess.check_call(
                [python_executable, "-m", "pip", "install", "-r", requirements_path],
                timeout=60,
            )
            logger.info("Successfully updated packages.")
        except Exception as e:
            logger.exception("Failed to update requirements", exc_info=e)

    def try_update(self):
        """
        Automatic update entrypoint method.
        """
        if self.repo.head.is_detached or self.repo.active_branch.name != TARGET_BRANCH:
            logger.debug("Not on the target branch, skipping auto-update")
            return

        if not self.check_version_updated():
            return

        if not self.attempt_update():
            return

        if self.update_requirements:
            self.attempt_package_update()

        restart_app()

def restart_app():
    """Restarts the current application."""
    os.execv(sys.executable, [sys.executable] + sys.argv)