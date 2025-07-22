import threading
import time
from types import SimpleNamespace

import bittensor as bt
import pytest
import websockets
from websockets.frames import Close  # websockets ≥11


# ── speed up back‑off ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)


# ── helper to build a minimal listener ------------------------------------------------
def _make_listener(monkeypatch):
    from neurons.base_node import BaseNode  # << your concrete class

    # always create a client on the public "finney" test‑net
    real_subtensor = bt.subtensor
    monkeypatch.setattr(
        bt, "subtensor", lambda *a, **k: real_subtensor(network="finney")
    )

    class _TestNode(BaseNode):
        async def run(self):  # unused in this test
            pass

    node = _TestNode()
    node.config = bt.Config()  # dummy cfg is fine
    node.hparams = SimpleNamespace(blocks_per_window=1)
    return node


# ── the test ------------------------------------------------------------------------
def test_block_listener_handles_timeouts(monkeypatch):
    listener = _make_listener(monkeypatch)

    # create the real client *once* so we can patch its class method
    listener.subtensor_rpc = bt.subtensor()

    # patch the class method, so every call (also after .initialize())
    # goes through our flaky implementation
    calls = {"n": 0}
    SubClass = listener.subtensor_rpc.substrate.__class__

    def flaky_subscribe(self, handler):
        calls["n"] += 1
        if calls["n"] < 3:  # first two attempts drop
            raise websockets.exceptions.ConnectionClosedError(
                Close(1006, "simulated‑drop"), None
            )
        handler({"header": {"number": 42}})  # third attempt succeeds

    monkeypatch.setattr(
        SubClass, "subscribe_block_headers", flaky_subscribe, raising=True
    )

    # ── run the listener thread ───────────────────────────────────────────────
    t = threading.Thread(target=listener.block_listener, daemon=True)
    t.start()

    # wait until we move into at least window 1 (blocks_per_window == 1)
    start = time.time()
    while listener.current_window == 0 and time.time() - start < 5:
        time.sleep(0.05)

    listener.stop_event.set()
    t.join(timeout=2)

    # verify reconnect logic really ran
    assert listener.current_window >= 1, "listener never advanced to a new window"
    assert calls["n"] >= 3, "subscribe_block_headers wasn't retried"
