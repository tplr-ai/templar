# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

import os
import re
import sys
import json
import requests
import subprocess

import tplr as tplr

GITHUB_RAW_URL = "https://raw.githubusercontent.com/unconst/tplr/refs/heads/main/run.py"

def get_pm2_process_name():
    """
    Attempt to find the current process's PM2 name by using `pm2 jlist` and matching the current PID.
    """
    # Get current PID
    current_pid = os.getpid()
    
    # Run `pm2 jlist` to get the JSON list of PM2 processes
    try:
        result = subprocess.run(["pm2", "jlist"], check=True, capture_output=True, text=True)
        pm2_data = json.loads(result.stdout)
    except Exception as e:
        tplr.logger.error(f"Error running `pm2 jlist`: {e}")
        return None
    
    # Find the process with the matching pid
    for proc in pm2_data:
        if proc.get("pid") == current_pid:
            return proc.get("name")
    
    return None


def get_remote_spec_version():
    """
    Fetch the remote run.py from GitHub and extract the SPEC_VERSION value.
    """
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        content = response.text
    except Exception as e:
        tplr.logger.error(f"Failed to fetch remote run.py: {e}")
        return None
    
    # Regex to find `SPEC_VERSION = <number>`
    match = re.search(r"^\s*SPEC_VERSION\s*=\s*(\d+)", content, re.MULTILINE)
    if match:
        return int(match.group(1))
    else:
        return None


def update_and_restart():
    """
    Perform a git pull to update the local repository and then restart the process via PM2.
    """
    pm2_name = get_pm2_process_name()
    if not pm2_name:
        tplr.logger.warning("Could not determine PM2 process name. Restart aborted.")
        return
    
    # Pull latest changes
    try:
        subprocess.run(["git", "pull", "origin", "master"], check=True)
    except Exception as e:
        tplr.logger.error(f"Failed to pull latest changes from git: {e}")
        return
    
    # Restart the PM2 process
    tplr.logger.info(f"Restarting PM2 process '{pm2_name}'...")
    try:
        subprocess.run(["pm2", "restart", pm2_name], check=True)
    except Exception as e:
        tplr.logger.error(f"Failed to restart PM2 process: {e}")
        
def optionally_auto_update( SPEC_VERSION ):
    """
    Checks for optional update and restarts.
    """
    # Check remote SPEC_VERSION
    remote_version = get_remote_spec_version()
    if remote_version is None:
        tplr.logger.error("Could not determine remote SPEC_VERSION. Continuing with current version.")
    else:
        if remote_version > SPEC_VERSION:
            tplr.logger.info(f"Remote SPEC_VERSION ({remote_version}) is greater than local ({SPEC_VERSION}). Updating and restarting...")
            update_and_restart()
            sys.exit(0)
        else:
            tplr.logger.info(f"Local SPEC_VERSION ({SPEC_VERSION}) is up-to-date or newer (Remote: {remote_version}).")