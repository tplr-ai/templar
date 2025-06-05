#!/bin/bash
# Watchtower management script for Subtensor nodes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="/opt/subtensor/docker-compose.yml"
LOGFILE="/opt/subtensor/logs/watchtower-manage.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

show_help() {
    cat << EOF
Watchtower Management Script for Subtensor Nodes

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    status              Show Watchtower and container status
    start               Start Watchtower service
    stop                Stop Watchtower service
    restart             Restart Watchtower service
    monitor             Switch to monitor-only mode (no updates)
    update              Switch to update mode (default)
    force-update        Force immediate update check
    logs                Show Watchtower logs
    health              Check health of all containers
    switch-group        Switch deployment group (blue/green)

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output

Examples:
    $0 status           # Show current status
    $0 monitor          # Switch to monitor-only mode
    $0 force-update     # Force immediate update check
    $0 switch-group green  # Switch to green deployment group

EOF
}

check_watchtower_status() {
    if docker ps --filter "name=watchtower" --format "table {{.Names}}\t{{.Status}}" | grep -q watchtower; then
        log "Watchtower is running"
        docker ps --filter "name=watchtower" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
        return 0
    else
        log "Watchtower is not running"
        return 1
    fi
}

check_container_health() {
    log "Checking health of all Subtensor containers"
    
    for i in {0..2}; do
        container="subtensor-$i"
        if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
            status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-health-check")
            log "Container $container: $status"
            
            # Check if RPC is responding
            rpc_port=$((9933 + i))
            if curl -s -f "http://localhost:$rpc_port/health" >/dev/null 2>&1; then
                log "  RPC health check: PASS"
            else
                log "  RPC health check: FAIL"
            fi
        else
            log "Container $container: NOT RUNNING"
        fi
    done
}

switch_deployment_group() {
    local new_group="$1"
    
    if [[ "$new_group" != "blue" && "$new_group" != "green" ]]; then
        log "ERROR: Invalid deployment group. Use 'blue' or 'green'"
        exit 1
    fi
    
    log "Switching deployment group to: $new_group"
    
    # Update the group variable in the compose file
    sed -i "s/subtensor.deployment.group=.*/subtensor.deployment.group=$new_group/" "$COMPOSE_FILE"
    
    # Restart Watchtower to pick up new labels
    docker-compose -f "$COMPOSE_FILE" restart watchtower
    
    log "Deployment group switched to: $new_group"
}

force_update() {
    log "Forcing immediate update check"
    
    if check_watchtower_status; then
        # Send USR1 signal to trigger immediate check
        docker kill --signal=USR1 watchtower
        log "Update signal sent to Watchtower"
    else
        log "ERROR: Watchtower is not running"
        exit 1
    fi
}

set_monitor_mode() {
    local monitor_only="$1"
    
    log "Setting monitor mode to: $monitor_only"
    
    # Update environment variable
    if [[ "$monitor_only" == "true" ]]; then
        docker-compose -f "$COMPOSE_FILE" exec watchtower sh -c 'export WATCHTOWER_MONITOR_ONLY=true'
        log "Watchtower set to monitor-only mode"
    else
        docker-compose -f "$COMPOSE_FILE" exec watchtower sh -c 'export WATCHTOWER_MONITOR_ONLY=false'
        log "Watchtower set to update mode"
    fi
    
    # Restart to apply changes
    docker-compose -f "$COMPOSE_FILE" restart watchtower
}

main() {
    case "${1:-help}" in
        status)
            check_watchtower_status
            echo
            check_container_health
            ;;
        start)
            log "Starting Watchtower"
            docker-compose -f "$COMPOSE_FILE" up -d watchtower
            ;;
        stop)
            log "Stopping Watchtower"
            docker-compose -f "$COMPOSE_FILE" stop watchtower
            ;;
        restart)
            log "Restarting Watchtower"
            docker-compose -f "$COMPOSE_FILE" restart watchtower
            ;;
        monitor)
            set_monitor_mode "true"
            ;;
        update)
            set_monitor_mode "false"
            ;;
        force-update)
            force_update
            ;;
        logs)
            docker-compose -f "$COMPOSE_FILE" logs -f watchtower
            ;;
        health)
            check_container_health
            ;;
        switch-group)
            if [[ -z "${2:-}" ]]; then
                log "ERROR: Please specify deployment group (blue or green)"
                exit 1
            fi
            switch_deployment_group "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log "ERROR: Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"