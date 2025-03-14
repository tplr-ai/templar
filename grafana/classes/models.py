from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Import the shared db instance
from app import db

class Version(db.Model):
    __tablename__ = 'tbl_version'
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_running = db.Column(db.Boolean, nullable=False, default=True)
    window_id = db.Column(db.Integer, nullable=True)

    @classmethod
    def get_last(cls):
        return cls.query.order_by(cls.created_at.desc()).first()

class RunMetadata(db.Model):
    __tablename__ = 'tbl_run_metadata'
    id = db.Column(db.Integer, primary_key=True)
    window_id = db.Column(db.Integer, nullable=False)
    blocks_per_window = db.Column(db.Float, nullable=True)
    avg_window_duration = db.Column(db.Float, nullable=True)
    gradient_retention = db.Column(db.Float, nullable=True)

class WindowInfo(db.Model):
    __tablename__ = 'tbl_window_info'
    id = db.Column(db.Integer, primary_key=True)
    window_number = db.Column(db.Integer, nullable=False)
    sync_window_number = db.Column(db.Integer, nullable=True)
    window_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    global_step = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    @classmethod
    def get_last(cls):
        return cls.query.order_by(cls.id.desc()).first()

class ActiveMiners(db.Model):
    __tablename__ = 'tbl_active_miners'
    id = db.Column(db.Integer, primary_key=True)
    window_id = db.Column(db.Integer, db.ForeignKey('tbl_window_info.id'), nullable=False)
    active_miners = db.Column(db.String(1000), nullable=True)
    error_miners = db.Column(db.String(1000), nullable=True)
    bad_miners = db.Column(db.String(1000), nullable=True)
    gather_miners = db.Column(db.String(1000), nullable=True)
    diff_miners = db.Column(db.String(1000), nullable=True)

class Neuron(db.Model):
    __tablename__ = 'tbl_neuron'
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, nullable=False, unique=True)
    hotkey = db.Column(db.String(100), nullable=False)
    coldkey = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)

class NeuronThirdParty(db.Model):
    __tablename__ = 'tbl_neuron_third_party'
    id = db.Column(db.Integer, primary_key=True)
    neuron_id = db.Column(db.Integer, db.ForeignKey('tbl_neuron.uid'), nullable=False)
    r2_bucket_name = db.Column(db.String(200), nullable=False)
    r2_read_access_key_id = db.Column(db.String(200), nullable=False)
    r2_read_secrect_access_key = db.Column(db.String(200), nullable=False)

class ValidatorEvalInfo(db.Model):
    __tablename__ = 'tbl_validator_eval_info'
    id = db.Column(db.Integer, primary_key=True)
    window_id = db.Column(db.Integer, db.ForeignKey('tbl_window_info.id'), nullable=False)
    neuron_id = db.Column(db.Integer, db.ForeignKey('tbl_neuron.uid'), nullable=False)
    loss_before = db.Column(db.Float, nullable=True)
    loss_after = db.Column(db.Float, nullable=True)
    loss_improvement = db.Column(db.Float, nullable=True)
    loss_random_before = db.Column(db.Float, nullable=True)
    loss_random_after = db.Column(db.Float, nullable=True)
    loss_random_improvement = db.Column(db.Float, nullable=True)
    current_eval_uid = db.Column(db.String(200), nullable=True)
    eval_uids = db.Column(db.String(1000), nullable=True)
    mean_scores = db.Column(db.Float, nullable=True)
    mean_moving_avg_scores = db.Column(db.Float, nullable=True)

class Gradients(db.Model):
    __tablename__ = 'tbl_gradients'
    id = db.Column(db.Integer, primary_key=True)
    window_id = db.Column(db.Integer, db.ForeignKey('tbl_window_info.id'), nullable=False)
    neuron_id = db.Column(db.Integer, db.ForeignKey('tbl_neuron.uid'), nullable=False)
    r2_bucketname = db.Column(db.String(200), nullable=False)
    gradient_filename = db.Column(db.String(200), nullable=False)
    gradient_filesize = db.Column(db.Float, nullable=False)
    gradient_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class EvalInfoDetail(db.Model):
    __tablename__ = 'tbl_eval_info_detail'
    id = db.Column(db.Integer, primary_key=True)
    window_id = db.Column(db.Integer, db.ForeignKey('tbl_window_info.id'), nullable=False)
    vali_id = db.Column(db.Integer, db.ForeignKey('tbl_neuron.uid'), nullable=False)
    miner_id = db.Column(db.Integer, db.ForeignKey('tbl_neuron.uid'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    moving_avg_score = db.Column(db.Float, nullable=True)
    weight = db.Column(db.Float, nullable=True)
