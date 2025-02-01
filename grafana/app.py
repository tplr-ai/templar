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

def update_current_version(current_version, old_version_record):
    # Create a new version record
    new_window_info = Version(
        version=current_version,
        created_at=datetime.utcnow(),  # current timestamp
    )
    db.session.add(new_window_info)

    # Update old version
    if old_version_record:
        old_version_record.is_running = False
        db.session.update(old_version_record)

def insert_window(window_number, global_step, learning_rate):    
    # Create a new WindowInfo record
    new_window_info = WindowInfo(
        window_number=window_number,
        window_time=datetime.utcnow(),  # current timestamp
        global_step=global_step,
        learning_rate=learning_rate
    )

    # Add the new record to the session
    db.session.add(new_window_info)

    return new_window_info.id

def insert_run_metadata(window_id, avg_window_duration, gradient_retention):    
    # Create a new run metadata record
    new_run_metadata = RunMetadata(
        window_id=window_id,
        avg_window_duration=avg_window_duration,
        gradient_retention=gradient_retention
    )

    # Add the new record to the session
    db.session.add(new_run_metadata)

def insert_active_miners(window_id, active_miners, error_miners):    
    tplr.logger.info(f"\n active_miners: {active_miners}, error_miners: {error_miners}")
    # Create a new active miners record
    new_active_miners = ActiveMiners(
        window_id=window_id,
        active_miners=active_miners,
        error_miners=error_miners
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
    db.session.commit()
    # Consider to add tbl_neuron_third_party table

def insert_dummy_validator_eval_info(window_id):
    # Create a dummy validator eval info record
    new_validator_eval_info = ValidatorEvalInfo(
        window_id=window_id,
        neuron_id=1,
        loss_before=0.4,
        loss_after=0.5,
        loss_improvement=0.2,
        current_eval_uid="10",
        eval_uids="10,11",
        mean_scores=1.45,
        mean_moving_avg_scores=1.57
    )

    # Add the new record to the session
    db.session.add(new_validator_eval_info)
   
def insert_dummy_eval_info_detail(window_id):
    # Create dummy eval info detail record
    new_eval_info_detail = EvalInfoDetail(
        window_id=window_id,
        vali_id=1,
        miner_id=10,
        score=1.54,
        moving_avg_score=1.57,
        weight=0.65
    )

    # Add the new record to the session
    db.session.add(new_eval_info_detail)

def insert_gradients(window_id, active_miners):
    for item in active_miners:
        tplr.logger.info(f"\n item: {item}")
        # Create a new gradient record
        new_gradient = Gradients(
            window_id=window_id,
            neuron_id=item.uid,
            r2_bucketname=item.bucket_name,
            gradient_filename=item.filename,
            gradient_filesize=item.content_length,
            gradient_timestamp=item.timestamp
        )

        # Add the new record to the session
        db.session.add(new_gradient)

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
            print(f"window: {step_window}, Active list: {grafana.grad_dict.get(step_window, [])}")
            step_window = grafana.current_window - WINDOW_OFFSET
            tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
            grafana.comms.update_peers_with_buckets()
            # Check a version
            version = get_tplr_version()
            version_record = get_current_version_record()
            if not version_record or version != version_record.version:
                update_current_version(version, version_record)
                tplr.logger.info(f"\nUpdated version {version}")
            # Insert a new window
            window_id = insert_window(step_window, grafana.global_step, grafana.hparams.learning_rate)
            tplr.logger.info(f"\nInserted a new window {step_window}")
            # Insert a run metadata
            insert_run_metadata(window_id, grafana.hparams.blocks_per_window, 100)
            tplr.logger.info(f"\nInserted a run metadata {step_window}")
            # Insert active miners
            active_miners, error_miners = await grafana.get_active_miners(step_window)
            insert_active_miners(window_id, active_miners, error_miners)
            tplr.logger.info(f"\nInserted active miners {step_window}")

            # Insert validator eval info & eval info detail
            # We will get this via wandb log and data, and insert into tbl_validator_eval_info, tbl_eval_info_detail tables
            insert_dummy_validator_eval_info(window_id)
            tplr.logger.info(f"\nInserted validator eval info {step_window}")
            insert_dummy_eval_info_detail(window_id)
            tplr.logger.info(f"\nInserted eval info detail {step_window}")

            # Insert gradients
            insert_gradients(window_id, active_miners)
            tplr.logger.info(f"\nInserted gradients {step_window}")

            # Commit the session to save the changes
            db.session.commit()

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
