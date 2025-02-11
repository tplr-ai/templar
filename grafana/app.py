import re
import sys
import os

import requests

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Now proceed with other imports
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import asyncio
from classes.grafana_tools import Grafana, WINDOW_OFFSET
import tplr
from tplr import __version__
from dotenv import load_dotenv
import wandb
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import models to register them with SQLAlchemy
from classes.models import *

def get_current_version_record():
    version = Version.get_last()
    return version

def get_tplr_version():
    url = "https://raw.githubusercontent.com/tplr-ai/templar/main/src/tplr/__init__.py"

    response = requests.get(url)

    if response.status_code == 200:
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', response.text)
        if match:
            return match.group(1)  # ✅ Extracts version number
        else:
            print("Version not found.")
    else:
        print("Failed to fetch file.")

def update_current_version(current_version, old_version_record, created_at, window_number):
    # Create a new version record
    new_window_info = Version(
        version=current_version,
        created_at=created_at,  # current timestamp
        window_number=window_number,
    )
    db.session.add(new_window_info)

    # Update old version
    if old_version_record:
        old_version_record.is_running = False

def insert_window(window_number, global_step, learning_rate):    
    api = wandb.Api()
    runs = api.runs(f"tplr/templar")
    run_id = "hvf4v9fp" # Run ID for V1
    for run in runs:
        if run.name == "V1" and run.state == "running":
            run_id = run.id
            break
    run = api.run(f"tplr/templar/{run_id}")
    history = run.history(pandas=False)
    tplr.logger.info(f"\nWandb run {run_id}")

    sync_window_number = window_number
    if history:
        eval_info = {}
        eval_info_detail = {}
        last_row = history[-1]  # Get the last row
        for key, value in last_row.items():
            if "latest/validator/network/window" in key:
                try:
                    sync_window_number= value
                    break

                except ValueError as e:
                    tplr.logger.error(f"Error parsing key: {key} - {e}")

    # Create a new WindowInfo record
    new_window_info = WindowInfo(
        window_number=window_number,
        sync_window_number=sync_window_number,
        window_time=datetime.utcnow(),  # current timestamp
        global_step=global_step,
        learning_rate=learning_rate
    )

    # Add and commit the new record
    db.session.add(new_window_info)
    db.session.flush()  # ✅ Assigns ID without committing

    return new_window_info.id

def insert_run_metadata(window_id, avg_window_duration, blocks_per_window, gradient_retention):    
    # Create a new run metadata record
    new_run_metadata = RunMetadata(
        window_id=window_id,
        avg_window_duration=avg_window_duration,
        blocks_per_window=blocks_per_window,
        gradient_retention=gradient_retention
    )

    # Add the new record to the session
    db.session.add(new_run_metadata)

def insert_active_miners(window_id, active_miners, error_miners, bad_miners, gather_miners, diff_miners):    
    tplr.logger.info(f"\n window_id: {window_id}")
    tplr.logger.info(f"\n active_miners: {active_miners}, error_miners: {error_miners}, bad_miners: {bad_miners}, gather_miners: {sorted(gather_miners)}, diff_miners: {sorted(diff_miners)}")
    # Create a new active miners record
    new_active_miners = ActiveMiners(
        window_id=window_id,
        active_miners=",".join(map(str, active_miners)),
        error_miners=",".join(map(str, error_miners)),
        bad_miners=",".join(map(str, bad_miners)),
        gather_miners=",".join(map(str, sorted(gather_miners))),
        diff_miners=",".join(map(str, sorted(diff_miners))),
    )

    # Add the new record to the session
    db.session.add(new_active_miners)

def sync_neurons(metagraph_info):
    for uid, stake in enumerate(metagraph_info["stake"]):
        neuron = Neuron.query.filter_by(uid=uid).first()
        if neuron:
            neuron.hotkey = metagraph_info["hotkeys"][uid]
            neuron.coldkey = metagraph_info["coldkeys"][uid]
        else:
            new_neuron = Neuron(
                uid=uid,
                hotkey=metagraph_info["hotkeys"][uid],
                coldkey=metagraph_info["coldkeys"][uid],
                type=bool(stake>=1.024e3)
            )
            # Add the new record to the session
            db.session.add(new_neuron)
    tplr.logger.info(f"\nSynchronized neurons")
    db.session.commit()
    # Consider to add tbl_neuron_third_party table

def insert_validator_eval_info(window_id):
    api = wandb.Api()
    runs = api.runs(f"tplr/templar")
    run_id = "hvf4v9fp" # Run ID for V1
    for run in runs:
        if run.name == "V1" and run.state == "running":
            run_id = run.id
            break
    run = api.run(f"tplr/templar/{run_id}")
    history = run.history(pandas=False)
    tplr.logger.info(f"\nWandb run {run_id}")
    tplr.logger.info(f"\nWandb run.state {run.state}")
    tplr.logger.info(f"\nWandb run.history {len(history)}")
    if history:
        eval_info = {}
        eval_info_detail = {}
        last_row = history[-1]  # Get the last row
        for key, value in last_row.items():
            if "latest/validator/loss/own/before" in key:
                eval_info["loss_before"] = value
            elif "latest/validator/loss/own/after" in key:
                eval_info["loss_after"] = value
            elif "latest/validator/loss/own/improvement" in key:
                eval_info["loss_improvement"] = value
            elif "latest/validator/loss/random/before" in key:
                eval_info["loss_random_before"] = value
            elif "latest/validator/loss/random/after" in key:
                eval_info["loss_random_after"] = value
            elif "latest/validator/loss/random/improvement" in key:
                eval_info["loss_random_improvement"] = value
            elif "latest/validator/network/evaluated_uids" in key:
                eval_info["eval_uids"] = value
            elif "latest/validator/scores/mean" in key:
                tplr.logger.info(f"\nWandb key {key}, value {value}")
                eval_info["mean_scores"] = value
            elif "latest/validator/moving_avg_scores/mean" in key:
                tplr.logger.info(f"\nWandb key {key}, value {value}")
                eval_info["mean_moving_avg_scores"] = value
            elif "latest/validator/gradient_scores/" in key or \
                 "latest/validator/final_moving_avg_scores/" in key  or \
                 "latest/validator/weights/" in key:
                try:
                    uid = int(key.split("/")[-1])  # Extract miner ID
                    field = key.split("/")[-2]  # Extract field type

                    if uid not in eval_info_detail:
                        eval_info_detail[uid] = {}

                    eval_info_detail[uid][field] = value

                except ValueError as e:
                    tplr.logger.error(f"Error parsing key: {key} - {e}")

        # Create a dummy validator eval info record
        new_validator_eval_info = ValidatorEvalInfo(
            window_id=window_id,
            neuron_id=1,
            loss_before=eval_info.get("loss_before", 0),
            loss_after=eval_info.get("loss_after", 0),
            loss_improvement=eval_info.get("loss_improvement", 0),
            loss_random_before=eval_info.get("loss_random_before", 0),
            loss_random_after=eval_info.get("loss_random_after", 0),
            loss_random_improvement=eval_info.get("loss_random_improvement", 0),
            # current_eval_uid="10",
            eval_uids=eval_info.get("eval_uids", 0),
            mean_scores=eval_info.get("mean_scores", 0),
            mean_moving_avg_scores=eval_info.get("mean_moving_avg_scores", 0)
        )

        # Add the new record to the session
        db.session.add(new_validator_eval_info)
        
        for uid, details in eval_info_detail.items():
            new_eval_info_detail = EvalInfoDetail(
                window_id=window_id,
                vali_id=1,
                miner_id=uid,
                score=details.get("gradient_scores", 0),
                moving_avg_score=details.get("final_moving_avg_scores", 0),
                weight=details.get("weights", 0)
            )

            # Add the new record to the session
            db.session.add(new_eval_info_detail)

def insert_gradients(window_id, active_miners):
    for item in active_miners:
        # Create a new gradient record
        new_gradient = Gradients(
            window_id=window_id,
            neuron_id=item["uid"],
            r2_bucketname=item["bucket_name"],
            gradient_filename=item["filename"],
            gradient_filesize=item["content_length"],
            gradient_timestamp=item["timestamp"]
        )

        # Add the new record to the session
        db.session.add(new_gradient)

async def get_active_miners(grafana, step_window):
    active_miners, error_miners = await grafana.get_active_miners(step_window)

    active_peers = grafana.grad_dict.get(step_window, [])
    active_uids = [peer["uid"] for peer in active_peers]

    active_miners = grafana.comms.eval_peers
    diff_miners_uids = []
    for uid in active_miners:
        if uid not in active_uids:
            diff_miners_uids.append(uid)
    active_miners_uids = [miner["uid"] for miner in active_miners]
    error_miners_uids = [miner["uid"] for miner in error_miners]

    gradients = {}
    download_uids = []
    for peer in active_peers:
        if peer["uid"] not in gradients.keys():
            download_uids.append(peer["uid"])
    
    # Download gradients
    num_samples = min(7, len(download_uids))  # Ensure we don’t exceed available elements
    download_uids = np.random.choice(download_uids, size=num_samples, replace=False)
    
    result_gradients, result_metadata = await grafana.download_gradients(download_uids, step_window, key="gradient")
    similarities = await grafana.compute_cosine_similarities(gradients)

    # Prepare Data for Heatmap
    await grafana.print_similarity_matrix(similarities)

    grafana.comms.update_peers_with_buckets()
    grafana.peers = grafana.comms.peers
    tplr.logger.info(f"\nGather Peers: {grafana.peers}")
    
    # SAVE THIS LIST TO DB AND SHOW IN GRAFANA!
    bad_peers = await grafana.analyze_similarities(similarities, active_peers, window=step_window, threshold=0.99)
    tplr.logger.info(f"\nBad peers {bad_peers}")
    return active_miners_uids, error_miners_uids, bad_peers, grafana.peers, diff_miners_uids

# Async function moved from grafana_tools.py
async def run_grafana():
    print("--------Running grafana task------")
    grafana = Grafana()
    await grafana.initialize()

    grafana.grad_dict = {}
    step_window = grafana.current_window - WINDOW_OFFSET

    tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
    timer = 0

    while True:
        if step_window != grafana.current_window - WINDOW_OFFSET:
            try:
                step_window = grafana.current_window - WINDOW_OFFSET
                tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
                grafana.comms.update_peers_with_buckets()
                # Check a version
                version = get_tplr_version()
                version_record = get_current_version_record()
                if not version_record or version != version_record.version:
                    update_current_version(version, version_record, grafana.started_time, step_window)
                    tplr.logger.info(f"\nUpdated version {version} started_time {grafana.started_time}")
                # Insert a new window
                global_step = step_window - grafana.start_window
                window_id = insert_window(step_window, global_step, grafana.hparams.learning_rate)
                tplr.logger.info(f"\nInserted a new window {step_window}, window_id {window_id}")
                # Insert a run metadata
                insert_run_metadata(window_id, grafana.get_avg_wnd_duration(), grafana.hparams.blocks_per_window, 100)
                tplr.logger.info(f"\nInserted a run metadata {step_window}")
                # Insert active miners
                active_miners_uids, error_miners_uids, bad_miners_uids, gather_miners_uids, diff_miners_uids = await get_active_miners(grafana, step_window)
                insert_active_miners(window_id, active_miners_uids, error_miners_uids, bad_miners_uids, gather_miners_uids, diff_miners_uids)
                tplr.logger.info(f"\nInserted active miners {step_window}")

                # Insert validator eval info & eval info detail
                insert_validator_eval_info(window_id)
                tplr.logger.info(f"\nInserted validator eval info {step_window}")

                # Insert gradients
                active_miners, error_miners = await grafana.get_active_miners(step_window)
                insert_gradients(window_id, active_miners)
                tplr.logger.info(f"\nInserted gradients {step_window}")

                # Commit the session to save the changes
                db.session.commit()
            except Exception as e:
                tplr.logger.error(f"Exception - {e}")

        grafana.grad_dict.setdefault(step_window, [])
        grafana.grad_error_dict.setdefault(step_window, [])

        # Retrieve and update active miners
        active_miners, error_miners = await grafana.get_active_miners(step_window)
        grafana.grad_dict[step_window].extend(active_miners)
        grafana.grad_error_dict[step_window].extend(error_miners) 

        if timer % 60 == 0: # Every 10 minutes
            sync_neurons(grafana.get_metagraph_info())

        timer += 1
        await asyncio.sleep(10)


# Run Grafana background task
async def start():
    task = asyncio.create_task(run_grafana())
    await task


# Testing endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    """A simple endpoint to verify the API is running."""
    return jsonify({"message": "API is up and running!"})

if __name__ == "__main__":
    print("-------App started-------")

    # Start Grafana task in a separate thread
    asyncio.run(start())  # Start Grafana loop

    app.run(debug=True, port=5000)
