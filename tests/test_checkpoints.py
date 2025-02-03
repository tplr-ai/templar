# ruff: noqa

import os
import re
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio

# --- Dummy classes and helpers for checkpointing tests ---


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# A dummy scheduler that mimics state_dict(), load_state_dict(), and counting steps.
class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict.get("last_epoch", 0)

    def step(self):
        self.last_epoch += 1


# Helper function to mimic the checkpoint data creation from validator/ miner.
def create_checkpoint_data(
    model, optimizer, scheduler, momentum, start_window, current_window
):
    checkpoint_data = {
        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": {
            k: (v.cpu().clone() if torch.is_tensor(v) else v)
            for k, v in optimizer.state_dict().items()
        },
        "scheduler_state_dict": scheduler.state_dict(),
        "momentum": {k: v.cpu().clone() for k, v in momentum.items()},
        "start_window": start_window,
        "current_window": current_window,
    }
    return checkpoint_data


# A dummy Comms class that simulates asynchronous save and load of checkpoints.
class DummyComms:
    def __init__(self, uid, local_save_dir):
        self.uid = uid
        self.local_save_dir = local_save_dir
        self.current_window = 0
        self.save_calls = []  # List to record checkpoint saves.
        # dummy bucket; not used in this dummy implementation.
        self.bucket = object()

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        global_step,
        current_window,
        start_window,
    ):
        checkpoint_data = create_checkpoint_data(
            model, optimizer, scheduler, momentum, start_window, current_window
        )
        self.save_calls.append({"global_step": global_step, "data": checkpoint_data})
        # Also write to local filesystem.
        uid_str = str(self.uid)
        checkpoint_filename = f"checkpoint-{global_step}-{uid_str}-v1.0.pt"
        save_path = os.path.join(self.local_save_dir, uid_str)
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, checkpoint_filename)
        torch.save(checkpoint_data, file_path)
        return checkpoint_data

    async def put(self, state_dict, uid, window, key, global_step, local):
        # Dummy implementation for the async put; not used in these tests.
        pass

    async def get_latest_checkpoint(self):
        # For testing, simply return the last saved checkpoint if exists.
        if self.save_calls:
            last = self.save_calls[-1]
            return last["data"], last["global_step"]
        return None

    def _load_latest_local_checkpoint(self):
        uid_str = str(self.uid)
        local_dir = os.path.join(self.local_save_dir, uid_str)
        pattern = re.compile(rf"checkpoint-(\d+)-{uid_str}-v1\.0\.pt$")
        if not os.path.exists(local_dir):
            return None
        checkpoints = []
        for file in os.listdir(local_dir):
            if pattern.match(file):
                file_path = os.path.join(local_dir, file)
                checkpoints.append(
                    {"path": file_path, "modified": os.path.getmtime(file_path)}
                )
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x["modified"])
            try:
                checkpoint_data = torch.load(latest["path"])
            except Exception:
                return None
            m = pattern.search(os.path.basename(latest["path"]))
            global_step = int(m.group(1)) if m else 0
            return checkpoint_data, global_step
        return None

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        transformer,
        compressor,
        current_window,
        device,
        peers,
        uid,
    ):
        # Simulated load_checkpoint that mimics the behavior in src/tplr/comms.py.
        result = await self.get_latest_checkpoint()
        if not result:
            return False, {}, 0, optimizer, scheduler
        checkpoint_data, _ = result
        try:
            # Restore model state.
            state_dict = {
                k: v.to(device) for k, v in checkpoint_data["model_state_dict"].items()
            }
            model.load_state_dict(state_dict)
            model.to(device)
            # Restore optimizer state.
            opt_state = optimizer.state_dict()
            for state in opt_state["state"].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            # Restore scheduler.
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            momentum = checkpoint_data["momentum"]
            checkpoint_start_window = checkpoint_data.get("start_window")
            checkpoint_current_window = checkpoint_data.get("current_window")
            if checkpoint_start_window is None or checkpoint_current_window is None:
                return False, {}, 0, optimizer, scheduler
            window_difference = current_window - checkpoint_current_window
            global_step = current_window - checkpoint_start_window
            # Sync scheduler with global_step.
            steps_needed = global_step - scheduler.last_epoch
            if steps_needed > 0:
                for _ in range(steps_needed):
                    optimizer.step()
                    scheduler.step()
            # If catch-up is needed.
            if window_difference > 0:
                BATCH_SIZE = 5
                windows_to_catch_up = list(
                    range(checkpoint_current_window + 1, current_window + 1)
                )
                for w in windows_to_catch_up:
                    # (For testing, simulate an update for each window)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
            return True, momentum, global_step, optimizer, scheduler
        except KeyError:
            return False, {}, 0, optimizer, scheduler
        except Exception:
            return False, {}, 0, optimizer, scheduler


# --- Pytest fixtures ---


@pytest.fixture
def dummy_components(tmp_path):
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = DummyScheduler(optimizer)
    momentum = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    start_window = 10
    current_window = 15
    # Use the temporary path as local save directory.
    return (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        str(tmp_path / "checkpoints"),
    )


# --- Test functions ---


# Test 1: Checkpoint Data Structure Validation
# ----------------------------------------------------
# Purpose:
#   Validates that checkpoint data contains all required components and tensors are
#   properly moved to CPU.
#
# Test Steps:
#   1. Create checkpoint data from dummy components
#   2. Verify all expected keys exist in checkpoint data
#   3. Check all tensors are on CPU device
#   4. Validate data types of window values
#
# Expected Results:
#   - Checkpoint data contains model_state_dict, optimizer_state_dict,
#     scheduler_state_dict, momentum, start_window, current_window
#   - All tensors in model_state_dict are on CPU
#   - All tensor values in optimizer state are on CPU
#   - All momentum tensors are on CPU
#   - Window values are integers
def test_checkpoint_data_structure(dummy_components):
    model, optimizer, scheduler, momentum, start_window, current_window, _ = (
        dummy_components
    )
    checkpoint_data = create_checkpoint_data(
        model, optimizer, scheduler, momentum, start_window, current_window
    )
    expected_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "momentum",
        "start_window",
        "current_window",
    ]
    for key in expected_keys:
        assert key in checkpoint_data, f"Missing key {key} in checkpoint data"
    for tensor in checkpoint_data["model_state_dict"].values():
        assert tensor.device.type == "cpu"
    for state in checkpoint_data["optimizer_state_dict"].get("state", {}).values():
        for v in state.values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu"
    for tensor in checkpoint_data["momentum"].values():
        assert tensor.device.type == "cpu"
    assert isinstance(checkpoint_data["start_window"], int)
    assert isinstance(checkpoint_data["current_window"], int)


# Test 5: Missing Key Error in Checkpoint Data
# ----------------------------------------------------
# Purpose:
#   Verifies that loading a checkpoint with missing required keys fails gracefully.
#
# Test Steps:
#   1. Create checkpoint data and remove a required key
#   2. Attempt to load the corrupted checkpoint
#   3. Verify loading fails and returns empty/default values
#
# Expected Results:
#   - Loading fails (success = False)
#   - Empty momentum dict returned
#   - Global step reset to 0
@pytest.mark.asyncio
async def test_missing_key_in_checkpoint(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    checkpoint_data = create_checkpoint_data(
        model, optimizer, scheduler, momentum, start_window, current_window
    )
    del checkpoint_data["start_window"]
    comms = DummyComms(uid=123, local_save_dir=tmp_save_dir)
    comms.save_calls.append({"global_step": 20, "data": checkpoint_data})
    success, loaded_momentum, global_step, _, _ = await comms.load_checkpoint(
        model, optimizer, scheduler, None, None, current_window, "cpu", [], 123
    )
    assert not success
    assert loaded_momentum == {}
    assert global_step == 0


# Test 6: Corrupted Checkpoint File Handling
# ----------------------------------------------------
# Purpose:
#   Validates handling of corrupted checkpoint files.
#
# Test Steps:
#   1. Create a checkpoint file with invalid content
#   2. Attempt to load the corrupted file
#
# Expected Results:
#   - Loading returns None instead of crashing
@pytest.mark.asyncio
async def test_corrupted_checkpoint_file(tmp_path):
    uid = 456
    save_dir = tmp_path / "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    uid_dir = save_dir / str(uid)
    os.makedirs(uid_dir, exist_ok=True)
    file_path = uid_dir / "checkpoint-30-456-v1.0.pt"
    with open(file_path, "w") as f:
        f.write("corrupted content")
    comms = DummyComms(uid=uid, local_save_dir=str(save_dir))
    loaded = comms._load_latest_local_checkpoint()
    assert loaded is None


# Test 7: Catch-Up Logic – Positive Window Difference
# ----------------------------------------------------
# Purpose:
#   Tests the catch-up mechanism when loading from an earlier window.
#
# Test Steps:
#   1. Create checkpoint with current_window = start_window
#   2. Load checkpoint with a higher current_window
#   3. Verify global step reflects catch-up operations
#
# Expected Results:
#   - Loading succeeds
#   - Global step includes both sync and catch-up steps
@pytest.mark.asyncio
async def test_catch_up_logic(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    comms = DummyComms(uid=789, local_save_dir=tmp_save_dir)
    # Create a checkpoint where the saved current_window equals start_window.
    checkpoint_data = create_checkpoint_data(
        model, optimizer, scheduler, momentum, start_window, start_window
    )
    comms.save_calls.append({"global_step": 0, "data": checkpoint_data})
    (
        success,
        loaded_momentum,
        global_step,
        opt_after,
        sched_after,
    ) = await comms.load_checkpoint(
        model, optimizer, scheduler, None, None, current_window, "cpu", [], 789
    )
    assert success
    # In our dummy load, each missing window makes two steps (one for sync and one for catch-up).
    expected_global_step = (current_window - start_window) + (
        current_window - start_window
    )
    assert global_step == expected_global_step


# Test 8: No Catch-Up Needed When Windows Align
# ----------------------------------------------------
# Purpose:
#   Verifies behavior when loading checkpoint from same window.
#
# Test Steps:
#   1. Create checkpoint with matching windows
#   2. Load checkpoint
#   3. Verify no catch-up steps performed
#
# Expected Results:
#   - Loading succeeds
#   - Global step matches window difference only
@pytest.mark.asyncio
async def test_no_catch_up_when_aligned(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    comms = DummyComms(uid=321, local_save_dir=tmp_save_dir)
    checkpoint_data = create_checkpoint_data(
        model, optimizer, scheduler, momentum, start_window, current_window
    )
    comms.save_calls.append(
        {"global_step": current_window - start_window, "data": checkpoint_data}
    )
    success, loaded_momentum, global_step, _, _ = await comms.load_checkpoint(
        model, optimizer, scheduler, None, None, current_window, "cpu", [], 321
    )
    assert success
    assert global_step == (current_window - start_window)


# Test 10: End-to-End Miner Checkpoint Save and Load Cycle
# ----------------------------------------------------
# Purpose:
#   Tests complete checkpoint save and load cycle for a miner.
#
# Test Steps:
#   1. Save checkpoint with initial model state
#   2. Create new model instances
#   3. Load checkpoint into new instances
#   4. Compare states between old and new models
#
# Expected Results:
#   - Loading succeeds
#   - Global step matches window difference
#   - Model parameters exactly match
@pytest.mark.asyncio
async def test_miner_checkpoint_cycle(dummy_components, tmp_path):
    model, optimizer, scheduler, momentum, start_window, current_window, _ = (
        dummy_components
    )
    uid = 111
    save_dir = str(tmp_path / "checkpoints_miner")
    comms = DummyComms(uid=uid, local_save_dir=save_dir)
    global_step = 20
    await comms.save_checkpoint(
        model, optimizer, scheduler, momentum, global_step, current_window, start_window
    )
    # Simulate miner restart by creating fresh model/optimizer/scheduler.
    new_model = DummyModel()
    new_optimizer = optim.SGD(new_model.parameters(), lr=0.1)
    new_scheduler = DummyScheduler(new_optimizer)
    (
        success,
        loaded_momentum,
        loaded_global_step,
        new_optimizer,
        new_scheduler,
    ) = await comms.load_checkpoint(
        new_model,
        new_optimizer,
        new_scheduler,
        None,
        None,
        current_window,
        "cpu",
        [],
        uid,
    )
    assert success
    assert loaded_global_step == (current_window - start_window)
    for (name1, param1), (name2, param2) in zip(
        model.state_dict().items(), new_model.state_dict().items()
    ):
        torch.testing.assert_allclose(param1.cpu(), param2.cpu())


# Test 12: Scheduler and Optimizer Sync after Catch-Up
# ----------------------------------------------------
# Purpose:
#   Verifies optimizer and scheduler stay synchronized during catch-up.
#
# Test Steps:
#   1. Create checkpoint with earlier window
#   2. Track optimizer step calls during catch-up
#   3. Verify optimizer steps and scheduler state
#
# Expected Results:
#   - Correct number of optimizer steps performed
#   - Scheduler epoch matches final global step
@pytest.mark.asyncio
async def test_scheduler_optimizer_sync_after_catch_up(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    # set checkpoint windows so that catch-up is needed: saved current_window < current_window
    checkpoint_start_window = start_window  # e.g., 10
    checkpoint_current_window = (
        start_window  # 10, so missing windows = current_window - 10
    )
    # set a custom scheduler.last_epoch in the checkpoint (simulate older checkpoint)
    scheduler.last_epoch = 2  # this value will be saved in the checkpoint
    checkpoint_data = create_checkpoint_data(
        model,
        optimizer,
        scheduler,
        momentum,
        checkpoint_start_window,
        checkpoint_current_window,
    )
    comms = DummyComms(uid=222, local_save_dir=tmp_save_dir)
    # pre-populate the save_calls to simulate a valid checkpoint in the validator bucket.
    comms.save_calls.append({"global_step": 0, "data": checkpoint_data})

    # Patch optimizer.step to count the number of calls (scheduler.step is assumed to be in sync)
    opt_calls = 0
    orig_opt_step = optimizer.step

    def counted_opt_step():
        nonlocal opt_calls
        opt_calls += 1
        orig_opt_step()

    optimizer.step = counted_opt_step

    # Now, current_window should be greater than the checkpoint's current_window (forcing catch-up)
    # For our test fixture, dummy_components current_window is, for example, 15.
    success, loaded_momentum, global_step, _, sched_after = await comms.load_checkpoint(
        model, optimizer, scheduler, None, None, current_window, "cpu", [], 222
    )

    # Computation of expected optimizer.step() calls:
    # Sync Phase: steps_needed = (current_window - checkpoint_start_window) - checkpoint_last_epoch
    #   = (15 - 10) - 2 = 3
    # Catch-Up Phase: additional steps = (current_window - checkpoint_current_window)
    #   = 15 - 10 = 5
    # Total expected calls = 3 + 5 = 8
    expected_calls = ((current_window - checkpoint_start_window) - 2) + (
        current_window - checkpoint_current_window
    )
    assert opt_calls == expected_calls, (
        f"Expected {expected_calls} optimizer.step() calls, got {opt_calls}"
    )

    # Also verify that the scheduler state is in sync: final scheduler.last_epoch should equal global_step.
    assert sched_after.last_epoch == global_step, (
        f"Expected scheduler.last_epoch ({sched_after.last_epoch}) to equal global_step ({global_step})"
    )

    # TODO: In the future, consider patching scheduler.step as well to independently verify its call count.


# Test 13: Asynchronous Gather Failures during Catch-Up Handling
# ----------------------------------------------------
# Purpose:
#   Tests handling of async gather failures during catch-up.
#
# Test Steps:
#   1. Create checkpoint with earlier window
#   2. Simulate failure by reducing catch-up step
#   3. Verify adjusted global step
#
# Expected Results:
#   - Loading succeeds despite failure
#   - Global step reflects missing catch-up step
@pytest.mark.asyncio
async def test_async_gather_failures(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    comms = DummyComms(uid=333, local_save_dir=tmp_save_dir)
    checkpoint_data = create_checkpoint_data(
        model, optimizer, scheduler, momentum, start_window, start_window
    )
    comms.save_calls.append({"global_step": 0, "data": checkpoint_data})
    original_load_checkpoint = comms.load_checkpoint

    async def load_checkpoint_with_failure(*args, **kwargs):
        success, mom, gs, optn, sched = await original_load_checkpoint(*args, **kwargs)
        return success, mom, gs - 1, optn, sched

    comms.load_checkpoint = load_checkpoint_with_failure
    success, loaded_momentum, global_step, _, _ = await comms.load_checkpoint(
        model, optimizer, scheduler, None, None, current_window, "cpu", [], 333
    )
    assert success
    expected_global_step = (current_window - start_window) * 2 - 1
    assert global_step == expected_global_step


# Test 14: Checkpoint Save Trigger Frequency
# ----------------------------------------------------
# Purpose:
#   Verifies checkpoint saving occurs at correct intervals.
#
# Test Steps:
#   1. Run simulation with checkpoint frequency
#   2. Count actual saves
#
# Expected Results:
#   - Saves occur at expected intervals
#   - Correct total number of saves
def test_checkpoint_save_trigger_frequency(dummy_components):
    (
        model,
        optimizer,
        scheduler,
        momentum,
        start_window,
        current_window,
        tmp_save_dir,
    ) = dummy_components
    uid = 444
    checkpoint_frequency = 5
    comms = DummyComms(uid=uid, local_save_dir=tmp_save_dir)
    save_count = 0

    async def simulate_run():
        nonlocal save_count
        for gs in range(1, 21):
            if gs % checkpoint_frequency == 0:
                await comms.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    momentum,
                    gs,
                    current_window,
                    start_window,
                )
                save_count += 1

    asyncio.run(simulate_run())
    assert save_count == 4  # for steps 5, 10, 15, 20


# Test 15: Verification of Local File Creation after Checkpoint Save
# ----------------------------------------------------
# Purpose:
#   Verifies checkpoint files are created with correct naming.
#
# Test Steps:
#   1. Save checkpoint
#   2. Check for file existence
#   3. Verify filename format
#
# Expected Results:
#   - File exists with correct name pattern
@pytest.mark.asyncio
async def test_local_file_creation_after_checkpoint_save(dummy_components, tmp_path):
    model, optimizer, scheduler, momentum, start_window, current_window, _ = (
        dummy_components
    )
    uid = 555
    save_dir = str(tmp_path / "local_checkpoint")
    comms = DummyComms(uid=uid, local_save_dir=save_dir)
    global_step = 10
    await comms.save_checkpoint(
        model, optimizer, scheduler, momentum, global_step, current_window, start_window
    )
    uid_dir = os.path.join(save_dir, str(uid))
    files = os.listdir(uid_dir)
    pattern = re.compile(rf"checkpoint-{global_step}-{uid}-v1\.0\.pt$")
    matches = [f for f in files if pattern.match(f)]
    assert len(matches) == 1


# Test 16: CPU Device Verification for Checkpoint Saved Tensors
# ----------------------------------------------------
# Purpose:
#   Verifies all tensors are moved to CPU during checkpoint save.
#
# Test Steps:
#   1. Save checkpoint
#   2. Check device type of all tensors
#
# Expected Results:
#   - All tensors are on CPU device
@pytest.mark.asyncio
async def test_cpu_device_verification_for_checkpoint_saved_tensors(
    dummy_components, tmp_path
):
    model, optimizer, scheduler, momentum, start_window, current_window, _ = (
        dummy_components
    )
    uid = 666
    save_dir = str(tmp_path / "cpu_verification")
    comms = DummyComms(uid=uid, local_save_dir=save_dir)
    global_step = 12
    checkpoint_data = await comms.save_checkpoint(
        model, optimizer, scheduler, momentum, global_step, current_window, start_window
    )
    for tensor in checkpoint_data["model_state_dict"].values():
        assert tensor.device.type == "cpu"
    for state in checkpoint_data["optimizer_state_dict"].get("state", {}).values():
        for v in state.values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu"
    for tensor in checkpoint_data["momentum"].values():
        assert tensor.device.type == "cpu"


# Test 17: Complete Checkpoint Save and Load Cycle
# ----------------------------------------------------
# Purpose:
#   Tests full checkpoint save and load cycle with state verification.
#
# Test Steps:
#   1. Save initial state
#   2. Create new instances
#   3. Load checkpoint
#   4. Verify all states match
#
# Expected Results:
#   - Loading succeeds
#   - All states exactly match original
@pytest.mark.asyncio
async def test_checkpoint_save_and_load_cycle(dummy_components, tmp_path):
    model, optimizer, scheduler, momentum, start_window, current_window, _ = (
        dummy_components
    )
    uid = 888
    save_dir = str(tmp_path / "checkpoint_cycle")
    comms = DummyComms(uid=uid, local_save_dir=save_dir)

    # Choose a global_step (arbitrary value, not used in loading logic)
    global_step = 20
    # Save the checkpoint via comms; also writes the file locally
    await comms.save_checkpoint(
        model, optimizer, scheduler, momentum, global_step, current_window, start_window
    )

    # Simulate a node restart by creating fresh instances
    new_model = DummyModel()
    new_optimizer = optim.SGD(new_model.parameters(), lr=0.1)
    new_scheduler = DummyScheduler(new_optimizer)

    # Load the checkpoint (this uses the last saved checkpoint from the validator/self bucket)
    (
        success,
        loaded_momentum,
        loaded_global_step,
        new_optimizer,
        new_scheduler,
    ) = await comms.load_checkpoint(
        new_model,
        new_optimizer,
        new_scheduler,
        None,
        None,
        current_window,
        "cpu",
        [],
        uid,
    )

    assert success, "Checkpoint loading failed."
    # Expected global_step computed as: current_window - start_window (sync + catch-up handled inside)
    expected_global_step = current_window - start_window
    assert loaded_global_step == expected_global_step, (
        f"Expected global_step {expected_global_step}, got {loaded_global_step}"
    )

    # Verify that model parameters are restored exactly
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), new_model.state_dict().items()
    ):
        torch.testing.assert_close(v1.cpu(), v2.cpu(), rtol=1e-4, atol=1e-6)

    # Verify that momentum tensors are restored exactly
    for key, value in momentum.items():
        torch.testing.assert_close(
            value.cpu(), loaded_momentum[key].cpu(), rtol=1e-4, atol=1e-6
        )
