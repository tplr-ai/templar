#!/bin/sh
# Set permissions on /wal so that Loki can write to it.
chmod -R 777 /wal
# Execute Loki with the custom config file.
exec loki --config.file=/etc/loki/local-config.yaml