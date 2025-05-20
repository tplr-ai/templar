#!/usr/bin/env bash

set +e

trap 'echo "Interrupted. Exiting."; exit 0' SIGINT SIGTERM

log() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1"
}

MAX_FAILURES=10
failure_count=0

ARGS=("$@")

while true; do
  log "Starting hfdownloader.py (attempt $(($failure_count + 1)))"
  
  python ./hfdownloader.py "${ARGS[@]}"
  
  exit_code=$?
  log "hfdownloader.py exited with code ${exit_code}"
  
  if [ $exit_code -eq 0 ]; then
    log "Process completed successfully. Exiting."
    break
  fi
  
  if [ $exit_code -eq 130 ]; then
    log "Process was interrupted by user. Exiting."
    exit $exit_code
  fi
  
  ((failure_count++))
  
  if [ $failure_count -ge $MAX_FAILURES ]; then
    log "Maximum number of consecutive failures ($MAX_FAILURES) reached. Giving up."
    exit 1
  fi
  
  backoff=$((5 * 2 ** (failure_count - 1)))
  backoff=$((backoff > 300 ? 300 : backoff))
  
  log "Restart attempt $failure_count/$MAX_FAILURES. Waiting $backoff seconds before retrying..."
  sleep $backoff
done

log "Download completed successfully"
exit 0