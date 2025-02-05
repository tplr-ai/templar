# ruff: noqa

import bittensor as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import numpy as np
import boto3
import botocore.config

sub = bt.subtensor()


def get_grads(uid):
    commit = sub.get_commitment(netuid=3, uid=uid)
    name = commit[:32]
    account_id = commit[:32]
    access_key_id = commit[32:64]
    secret_access_key = commit[64:]
    client_config = botocore.config.Config(
        signature_version="s3v4", max_pool_connections=256
    )
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com/"
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=client_config,
    )
    print(s3_client.get_bucket_location(Bucket=account_id))

    response = s3_client.list_objects_v2(Bucket=account_id)
    results = []

    if "Contents" in response:
        for item in response["Contents"]:
            key = item["Key"]
            last_modified = item["LastModified"]

            # Extract window, step and version if gradient file
            window = None
            step = None
            version = None
            if key.startswith("gradient-"):
                parts = key.split("-")
                if len(parts) >= 4:
                    window = int(parts[1])
                    step = int(parts[2])
                    version = parts[3].split(".pt")[0]

            results.append(
                {
                    "name": key,
                    "last_modified": last_modified,
                    "window": window,
                    "step": step,
                    "version": version,
                }
            )

    # Sort results by last_modified datetime
    results.sort(key=lambda x: x["last_modified"])
    return results


# List of UIDs to plot
uids = [250, 182, 211, 114, 127, 101, 128, 65, 243, 132, 251, 89, 107, 191, 66]

# Get results for each UID
all_results = []
for uid in uids:
    results = get_grads(uid)
    all_results.append(results)


# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Get current time and 24 hours ago in UTC
now = datetime.now(pytz.UTC)
day_ago = now - timedelta(days=10)

# Generate evenly spaced y-positions for each UID's section
section_height = 1.0 / len(uids)
colors = plt.cm.rainbow(np.linspace(0, 1, len(uids)))

# Plot vertical lines for each UID's results
for i, (uid_results, color) in enumerate(zip(all_results, colors)):
    y_min = i * section_height
    y_max = (i + 1) * section_height

    for result in uid_results:
        upload_time = result["last_modified"]
        # Ensure upload_time is timezone aware
        if not upload_time.tzinfo:
            upload_time = pytz.UTC.localize(upload_time)
        if upload_time >= day_ago:  # Only plot if within last 24 hours
            ax.vlines(
                x=upload_time,
                ymin=y_min,
                ymax=y_max,
                color=color,
                alpha=0.3,
                linewidth=1,
            )

# Set x-axis range to last 24 hours
ax.set_xlim(day_ago, now)

# Set y-axis range from 0 to 1
ax.set_ylim(0, 1)

# Format x-axis to show time nicely
plt.gcf().autofmt_xdate()

# Add labels and title
plt.xlabel("Upload Time")
plt.ylabel("UIDs")
plt.title("Timeline of File Uploads by UID (Last 24 Hours)")

# Add y-axis ticks with UID labels
y_ticks = [i * section_height + section_height / 2 for i in range(len(uids))]
ax.set_yticks(y_ticks)
ax.set_yticklabels(uids)

plt.tight_layout()
plt.show()
