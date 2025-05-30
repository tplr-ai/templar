#!/bin/bash
# Simple Ubuntu I/O and Network Performance Tuning Script
# Run with: sudo ./tune_os.sh

set -xe

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "Run with sudo: sudo ./tune_os.sh"
   exit 1
fi

echo "Setting file descriptor limits..."
if ! grep -qF "* soft nofile 1048576" /etc/security/limits.conf; then
    cat >> /etc/security/limits.conf << 'EOF'

# Performance tuning
* soft nofile 1048576
* hard nofile 1048576
* soft nproc 1048576
* hard nproc 1048576
EOF

ulimit -n 1048576 2>/dev/null || true

echo "Applying kernel optimizations..."

# Network performance
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.core.rmem_default=262144
sysctl -w net.core.wmem_default=262144
sysctl -w net.core.somaxconn=65535
sysctl -w net.core.netdev_max_backlog=30000
sysctl -w net.ipv4.tcp_rmem="4096 65536 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sysctl -w net.ipv4.tcp_congestion_control=bbr
sysctl -w net.ipv4.tcp_fastopen=3
sysctl -w net.ipv4.tcp_max_syn_backlog=30000
sysctl -w net.ipv4.tcp_tw_reuse=1
sysctl -w net.ipv4.tcp_fin_timeout=10
sysctl -w net.ipv4.tcp_keepalive_time=60
sysctl -w net.ipv4.tcp_slow_start_after_idle=0

# I/O performance
sysctl -w vm.dirty_ratio=15
sysctl -w vm.dirty_background_ratio=5
sysctl -w vm.swappiness=1
sysctl -w vm.vfs_cache_pressure=50
sysctl -w fs.file-max=2097152
sysctl -w fs.inotify.max_user_watches=524288

echo "Optimizing network interfaces..."
for iface in $(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | cut -d'@' -f1 | grep -v lo); do
    if [[ ! "$iface" =~ ^(docker|br-|virbr|veth) ]]; then
        ethtool -G "$iface" rx 4096 tx 4096 2>/dev/null || true
        ethtool -K "$iface" gro on gso on tso on 2>/dev/null || true
    fi
done

echo "Optimizing I/O schedulers..."
for device in $(lsblk -d -o NAME | grep -E '^(sd|nvme|vd)'); do
    scheduler_path="/sys/block/$device/queue/scheduler"
    if [[ -f "$scheduler_path" ]]; then
        # Use mq-deadline for all devices (safe and performant)
        if grep -q "mq-deadline" "$scheduler_path"; then
            echo "mq-deadline" > "$scheduler_path" 2>/dev/null || true
        fi
        # Set read-ahead
        echo "4096" > "/sys/block/$device/queue/read_ahead_kb" 2>/dev/null || true
    fi
done

echo "Making settings persistent..."
cat > /etc/sysctl.d/99-performance.conf << 'EOF'
# Network performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 30000
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_max_syn_backlog = 30000
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_slow_start_after_idle = 0

# I/O performance
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.swappiness = 1
vm.vfs_cache_pressure = 50
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
EOF

sysctl --system

echo "Current status:"
echo "   File descriptors: $(ulimit -n)"
echo "   TCP congestion: $(cat /proc/sys/net/ipv4/tcp_congestion_control)"
echo "   Socket buffer max: $(cat /proc/sys/net/core/rmem_max)"
echo ""
