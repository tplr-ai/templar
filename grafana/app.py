import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Now proceed with other imports
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import asyncio
from classes.grafana_tools import Grafana, WINDOW_OFFSET
import tplr
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


# Async function moved from grafana_tools.py
async def run_grafana():
    print("--------Running grafana task------")
    grafana = Grafana()
    await grafana.initialize()

    grafana.grad_dict = {}
    step_window = grafana.current_window - WINDOW_OFFSET

    tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
    print(f"\n{'-' * 20} Window: {step_window} {'-' * 20} ---------")

    while True:
        if step_window != grafana.current_window - WINDOW_OFFSET:
            print(f"window: {step_window}, Active list: {grafana.grad_dict.get(step_window, [])}")
            step_window = grafana.current_window - WINDOW_OFFSET
            tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
            grafana.comms.update_peers_with_buckets()

        grafana.grad_dict.setdefault(step_window, [])
        grafana.grad_error_dict.setdefault(step_window, [])

        # Retrieve and update active miners
        active_miners, error_miners = await grafana.get_active_miners(step_window)
        grafana.grad_dict[step_window].extend(active_miners)
        grafana.grad_error_dict[step_window].extend(error_miners)

        # Create a new WindowInfo record
        new_window_info = WindowInfo(
            window_number=step_window,
            window_time=datetime.utcnow(),  # current timestamp
            global_step=1000,
            learning_rate=0.01
        )

        # Add the new record to the session
        db.session.add(new_window_info)

        # Commit the session to save the changes
        db.session.commit()

        await asyncio.sleep(10)


# Run Grafana background task
@app.before_first_request
def start_grafana_task():
    print("-------Started grafana task------")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_grafana())


# Testing endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    """A simple endpoint to verify the API is running."""
    return jsonify({"message": "API is up and running!"})

if __name__ == "__main__":
    print("-------App started-------")
    app.run(debug=True, port=5000)
