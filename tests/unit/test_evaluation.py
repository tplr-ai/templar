import torch
from unittest.mock import MagicMock
import pytest
from types import SimpleNamespace
import random

from tplr.evaluation import *
from tests.mocks.model import MockModel, DummyOutput, MockTransformer, MockCompressor, MockOptimizer, MockScheduler
from tests.mocks.loader import MockLoader
from tplr import logger
from tplr.r2_dataset import R2DatasetLoader
from tests.mocks.r2_dataset import MockR2DatasetLoader

#############################################
# Test 1: test_evaluate_model_loss_basic
#############################################
def test_evaluate_model_loss_basic():
    """
    Test that evaluate_model_loss computes the correct average loss and batch count.
    
    - Uses MockModel from tests/mocks/model.py.
    - Uses MockLoader from tests/mocks/loader.py with two pre-defined batches.
    - The dummy model is set to always return a dummy output with loss.item() == 3.0.
    """
    # Create dummy tokenizer with pad_token_id
    dummy_tokenizer = MagicMock()
    dummy_tokenizer.pad_token_id = 0

    # Instantiate dummy model using our existing mocks.
    dummy_model = MockModel()
    
    # Create dummy output where outputs.loss.item() returns 3.0.
    dummy_loss = MagicMock()
    dummy_loss.item.return_value = 3.0
    dummy_output = MagicMock()
    dummy_output.loss = dummy_loss

    # Override forward so that model(input_ids, labels) returns our dummy output.
    dummy_model.forward = MagicMock(return_value=dummy_output)
    
    # Create a dummy loader with 2 batches.
    batches = [
        [1, 2, 3],   # Batch 1
        [4, 5, 6]    # Batch 2
    ]
    dummy_loader = MockLoader(batches)

    # Device can be "cpu" for our tests.
    avg_loss, num_batches = evaluate_model_loss(dummy_model, dummy_loader, dummy_tokenizer, device="cpu")

    # Expect two batches and an average loss equal to 3.0 for the batches.
    assert num_batches == 2, f"Expected 2 batches, got {num_batches}"
    assert avg_loss == 3.0, f"Expected average loss 3.0, got {avg_loss}"


#############################################
# Test 2: test_evaluate_model_loss_empty_loader
#############################################
def test_evaluate_model_loss_empty_loader():
    """
    Test that evaluate_model_loss returns 0.0 average loss and 0 batches when given an empty loader.
    
    - Pass an empty loader (or generator that yields nothing).
    - Verify that the function returns an average loss of 0.0 and batches count as 0.
    """
    # Create dummy tokenizer with pad_token_id
    dummy_tokenizer = MagicMock()
    dummy_tokenizer.pad_token_id = 0

    # Instantiate dummy model using our existing mocks.
    dummy_model = MockModel()
    
    # Create an empty dummy loader using MockLoader with an empty list of batches.
    dummy_loader = MockLoader([])

    # Device can be "cpu" for our tests.
    avg_loss, num_batches = evaluate_model_loss(dummy_model, dummy_loader, dummy_tokenizer, device="cpu")

    # Expect 0 batches and an average loss equal to 0.0.
    assert num_batches == 0, f"Expected 0 batches, got {num_batches}"
    assert avg_loss == 0.0, f"Expected average loss 0.0, got {avg_loss}"


#############################################
# Test 3: test_apply_compressed_gradient_success
#############################################
def test_apply_compressed_gradient_success():
    """
    Test that apply_compressed_gradient applies the compressed gradient correctly.
    
    - Creates a dummy model with parameters.
    - Prepares a dummy state_dict with valid keys and dummy gradient data.
    - Uses dummy transformer and compressor objects with predictable decode/decompress behavior.
    - Verifies that parameters are updated as expected (e.g. subtracting lr * torch.sign(ones)).
    """
    # Instantiate dummy model (with two parameters: "layer1.weight" and "layer1.bias")
    dummy_model = MockModel()
    
    # Save original parameter values (clone each tensor) for comparison later.
    orig_params = {name: p.clone() for name, p in dummy_model.named_parameters()}
    
    # Create a dummy state_dict with keys for each parameter.
    # The actual values here don't matter because our dummy compressor will return a gradient tensor of ones.
    state_dict = {}
    for name, p in dummy_model.named_parameters():
        state_dict[name + "idxs"] = torch.tensor([0, 1])
        state_dict[name + "vals"] = torch.tensor([0.5, 0.5])
    
    # Create dummy transformer: simply returns identity for decode.
    class DummyTransformer:
        def decode(self, tensor):
            return tensor  # Identity behavior.
    
    # Create dummy compressor: decompress always returns a tensor of ones with shape xshape.
    class DummyCompressor:
        def decompress(self, p, idxs, vals, xshape, totalk):
            return torch.ones(xshape)
        def batch_decompress(self, p, idxs, vals, xshape, totalk):
            return torch.ones(xshape)
    
    dummy_transformer = DummyTransformer()
    dummy_compressor = DummyCompressor()
    
    # Prepare xshapes and totalks dictionaries (for each parameter, use its shape and total number of elements).
    xshapes = {}
    totalks = {}
    for name, p in dummy_model.named_parameters():
        xshapes[name] = p.shape
        totalks[name] = p.numel()
    
    device = "cpu"
    lr = 0.1

    # Call apply_compressed_gradient.
    # Expect that a decompressed gradient of ones is produced for each parameter,
    # so the update will be: new_value = old_value - lr * torch.sign(ones) = old_value - lr.
    updated_model = apply_compressed_gradient(dummy_model, state_dict, dummy_transformer, dummy_compressor, xshapes, totalks, device, lr)
    
    # Validate that each parameter has been updated correctly.
    for name, p in updated_model.named_parameters():
        expected = orig_params[name] - lr * torch.ones_like(p)
        assert torch.allclose(p, expected), f"Parameter {name} not updated correctly. Expected {expected}, got {p}"


#############################################
# Test 4: test_apply_compressed_gradient_missing_data
#############################################
def test_apply_compressed_gradient_missing_data(monkeypatch):
    """
    Test that apply_compressed_gradient applies gradients when available and skips parameters
    with missing gradient data.
    """
    device = "cpu"
    lr = 0.1
    # Instantiate dummy model
    dummy_model = MockModel()
    orig_params = {n: p.clone() for n, p in dummy_model.named_parameters()}

    # Create a state_dict that only provides gradient data for "layer1.weight".
    state_dict = {
        "layer1.weightidxs": torch.arange(5),
        "layer1.weightvals": torch.ones(5),
        # "layer1.bias" data missing.
    }
    dummy_transformer = MockTransformer()
    dummy_compressor = MockCompressor()
    xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
    totalks = {"layer1.weight": 50, "layer1.bias": 5}

    logs = []
    original_info = logger.info
    monkeypatch.setattr(logger, "info", lambda msg: logs.append(msg))
    
    updated_model = apply_compressed_gradient(dummy_model, state_dict, dummy_transformer, dummy_compressor, xshapes, totalks, device, lr)
    
    # For "layer1.weight": gradient should be applied: new_value = original - lr*sign(ones)= original - lr.
    # For "layer1.bias": missing gradient data means no update.
    for name, p in updated_model.named_parameters():
        if name == "layer1.weight":
            expected = orig_params[name] - lr * torch.ones_like(p)
            assert torch.allclose(p, expected), f"Parameter {name} not updated correctly."
        elif name == "layer1.bias":
            expected = orig_params[name]
            assert torch.allclose(p, expected), f"Parameter {name} should remain unchanged."
    
    # Ensure a log message mentioning "missing" was emitted.
    assert any("missing" in w.lower() for w in logs), "Expected log message for missing gradient data."


async def fake_load_r2_metadata(self):
    # Return dummy metadata with the expected structure for shard sizes,
    # using "num_rows" and "path" keys to match R2DatasetLoader._process_page.
    return (
        {
            "config1": {"shards": [{"num_rows": 10, "path": "dummy_path_config1"}]},
            "config2": {"shards": [{"num_rows": 10, "path": "dummy_path_config2"}]},
            "config3": {"shards": [{"num_rows": 10, "path": "dummy_path_config3"}]},
            "config4": {"shards": [{"num_rows": 10, "path": "dummy_path_config4"}]},
        },
        None
    )

#############################################
# Test 5: test_load_and_verify_pages_match
#############################################
@pytest.mark.asyncio
async def test_load_and_verify_pages_match(monkeypatch):
    """
    Simulate a scenario where state_dict includes miner's pages_info.
    Monkeypatch R2DatasetLoader.next_pages to return the same pages list.
    Check that a "pages match" message is logged and that both returned page lists are identical.
    """
    # Define pages list as tuples of three items.
    pages = [("config1", 1, "splitA"), ("config2", 2, "splitB")]
    state_dict = {"metadata": {"pages_info": pages}}
    
    # Monkeypatch R2DatasetLoader.next_pages to return the same pages list using an async function.
    async def fake_next_pages(*args, **kwargs):
        return pages
    monkeypatch.setattr(R2DatasetLoader, "next_pages", fake_next_pages)
    
    # Monkeypatch _load_r2_metadata to return dummy metadata.
    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", fake_load_r2_metadata)
    
    # Capture logger.info calls.
    logs = []
    monkeypatch.setattr(logger, "info", lambda msg: logs.append(msg))
    
    # Prepare dummy inputs required by evaluate_peer.
    uid = "dummy_uid"
    sync_window = 0
    hparams = SimpleNamespace(pages_per_window=2, batch_size=2, sequence_length=10, validator_sample_rate=1.0)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0  # set pad token to an integer
    config = {}
    model = MockModel()
    
    # Patch clone to return a proper copy (for simplicity, we return the same instance)
    model.clone = lambda: model
    # Patch forward to return a dummy output with a constant numeric loss value
    class DummyLoss:
        def item(self):
            return 1.0
    class DummyOutput:
        @property
        def loss(self):
            return DummyLoss()
    model.forward = lambda *args, **kwargs: DummyOutput()
    
    # Supply additional dummy arguments required by evaluate_peer.
    class DummyTransformer:
        def decode(self, tensor):
            return tensor
    class DummyCompressor:
        pass
    dummy_transformer = DummyTransformer()
    dummy_compressor = DummyCompressor()
    xshapes = {}
    totalks = {}
    device = "cpu"
    lr = 0.1
    optimizer = MagicMock()
    scheduler = MagicMock()
    
    # Then monkey-patch the loader before calling evaluate_peer:
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader.create", MockR2DatasetLoader.create
    )
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader._load_r2_metadata", MockR2DatasetLoader._load_r2_metadata
    )
    # Optionally, you can also patch get_loader:
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader.get_loader", MockR2DatasetLoader.get_loader
    )
    
    # Await the asynchronous evaluate_peer call.
    result = await evaluate_peer(
        uid, state_dict, sync_window, hparams, tokenizer, config, model,
        dummy_transformer, dummy_compressor, xshapes, totalks, device, lr, optimizer, scheduler
    )
    
    # Check that a "pages match" message is logged.
    match_logged = any("match" in w.lower() for w in logs)
    assert match_logged, "Expected log message indicating that pages match."
    
    # Validate that both miner_pages and local_pages equal the expected pages list.
    assert result.get("miner_pages") == pages, "Miner pages should match the state_dict pages."
    assert result.get("local_pages") == pages, "Local pages should match the state_dict pages."


#############################################
# Test 6: test_load_and_verify_pages_mismatch
#############################################
@pytest.mark.asyncio
async def test_load_and_verify_pages_mismatch(monkeypatch):
    """
    Simulate a scenario where state_dict pages_info is different from locally loaded pages.
    Monkeypatch R2DatasetLoader.next_pages to return a different list.
    Verify that a warning is logged about the mismatch and that both versions (miner and local) are returned.
    """
    pages_miner = [("config1", 1, "splitA"), ("config2", 2, "splitB")]
    pages_local = [("config3", 3, "splitC"), ("config4", 4, "splitD")]
    state_dict = {"metadata": {"pages_info": pages_miner}}
    
    # Monkeypatch R2DatasetLoader.next_pages to return pages_local using an async function.
    async def fake_next_pages(*args, **kwargs):
        return pages_local
    monkeypatch.setattr(R2DatasetLoader, "next_pages", fake_next_pages)
    
    # Monkeypatch _load_r2_metadata to return dummy metadata.
    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", fake_load_r2_metadata)
    
    # Capture logger.warning calls.
    logs = []
    monkeypatch.setattr(logger, "warning", lambda msg: logs.append(msg))
    
    # Prepare dummy inputs required by evaluate_peer.
    uid = "dummy_uid"
    sync_window = 0
    hparams = SimpleNamespace(pages_per_window=2, batch_size=2, sequence_length=10, validator_sample_rate=1.0)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0  # set pad token to an integer
    config = {}
    model = MockModel()
    
    # Patch clone to return a proper copy (for simplicity, we return the same instance)
    model.clone = lambda: model
    # Patch forward to return a dummy output with a constant numeric loss value
    class DummyLoss:
        def item(self):
            return 1.0
    class DummyOutput:
        @property
        def loss(self):
            return DummyLoss()
    model.forward = lambda *args, **kwargs: DummyOutput()
    
    # Supply additional dummy arguments.
    class DummyTransformer:
        def decode(self, tensor):
            return tensor
    class DummyCompressor:
        pass
    dummy_transformer = DummyTransformer()
    dummy_compressor = DummyCompressor()
    xshapes = {}
    totalks = {}
    device = "cpu"
    lr = 0.1
    optimizer = MagicMock()
    scheduler = MagicMock()
    
    # Then monkey-patch the loader before calling evaluate_peer:
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader.create", MockR2DatasetLoader.create
    )
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader._load_r2_metadata", MockR2DatasetLoader._load_r2_metadata
    )
    # Optionally, you can also patch get_loader:
    monkeypatch.setattr(
        "tplr.r2_dataset.R2DatasetLoader.get_loader", MockR2DatasetLoader.get_loader
    )
    
    # Await the asynchronous evaluate_peer call.
    result = await evaluate_peer(
        uid, state_dict, sync_window, hparams, tokenizer,
        config, model, dummy_transformer, dummy_compressor, xshapes, totalks, device, lr, optimizer, scheduler
    )
    
    # Check that a warning log mentioning "mismatch" was produced.
    mismatch_logged = any("mismatch" in w.lower() for w in logs)
    assert mismatch_logged, "Expected warning log for pages mismatch."
    
    # Validate that the returned dictionary contains both miner_pages and local_pages.
    assert result.get("miner_pages") == pages_miner, "Miner pages should match those in state_dict."
    assert result.get("local_pages") == pages_local, "Local pages should match the monkeypatched value."


#############################################
# Test 7: test_create_loader_from_pages
#############################################
@pytest.mark.asyncio
async def test_create_loader_from_pages(monkeypatch):
    """
    Provide dummy pages info.
    Invoke create_loader_from_pages and check that the returned loader yields
    batches in the expected format (e.g., correct batch dimensions or type).
    """
    # Dummy pages info.
    pages = [("config1", 1, "splitA"), ("config2", 2, "splitB")]
    # Dummy hyperparameters.
    hparams = SimpleNamespace(batch_size=2, sequence_length=10, pages_per_window=2)
    # Dummy tokenizer with pad_token_id; necessary for proper downstream tensor conversion.
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    # Patch R2DatasetLoader.create to return a dummy loader.
    async def fake_create(batch_size, sequence_length, pages_info, tokenizer, pack_samples=True):
        # For testing purposes we define a dummy loader that yields two batches.
        from tests.mocks.loader import MockLoader
        return MockLoader([[1, 2, 3], [4, 5, 6]])
    
    monkeypatch.setattr(R2DatasetLoader, "create", fake_create)

    # Call create_loader_from_pages.
    loader = await create_loader_from_pages(pages, hparams, tokenizer, sync_window=0)
    batches = list(loader)
    expected = [[1, 2, 3], [4, 5, 6]]
    assert batches == expected, f"Expected {expected}, got {batches}"


#############################################
# Test 8: test_collect_batches
#############################################
def test_collect_batches():
    """
    Create a dummy loader (a generator yielding known batches).
    Verify that collect_batches aggregates these batches into a list with
    the same elements.
    """
    # Define a dummy loader as a generator.
    def dummy_loader():
        yield [10, 20]
        yield [30, 40]
        yield [50, 60]
    
    batches = collect_batches(dummy_loader())
    expected = [[10, 20], [30, 40], [50, 60]]
    assert batches == expected, f"Expected {expected}, got {batches}"


#############################################
# Test 9: test_compute_average_loss_full_sampling
#############################################
def test_compute_average_loss_full_sampling():
    """
    Supply a list of dummy batches and use MockModel (from tests/mocks/model.py)
    that returns a constant loss. Call compute_average_loss with sample_rate=1
    (full sampling) and confirm that the average loss is computed as the constant
    loss value and that the correct number of batches are used.
    """
    # Use MockModel.
    model = MockModel()
    # Define dummy batches.
    batches = [[1, 2], [3, 4], [5, 6], [7, 8]]
    # Dummy tokenizer.
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    device = "cpu"
    sample_rate = 1.0

    # Call compute_average_loss.
    avg_loss, count, sampled_indices, total_batches = compute_average_loss(
        model, batches, tokenizer, device, sample_rate
    )
    # sample_rate = 1.0 means full sampling.
    assert count == len(batches), f"Expected count to be {len(batches)}, got {count}"
    assert total_batches == len(batches), f"Expected total_batches to be {len(batches)}, got {total_batches}"
    # Since every batch loss is 3.0, the average should equal 3.0.
    assert avg_loss == 3.0, f"Expected avg_loss of 3.0, got {avg_loss}"
    # For full sampling, sampled_indices should be [0, 1, 2, 3] (sorted).
    assert sampled_indices == list(range(len(batches))), f"Expected indices {list(range(len(batches)))}, got {sampled_indices}"


#############################################
# Test 10: test_evaluate_peer_success
#############################################
@pytest.mark.asyncio
async def test_evaluate_peer_success(monkeypatch):
    """
    Set up a dummy state_dict, dummy model (with a clone method), dummy optimizer,
    transformer, compressor, and other required variables. Monkey-patch R2DatasetLoader.get_loader
    and next_pages to return predictable dummy loaders for both "own" and "random" data.
    Verify that evaluate_peer returns a dictionary with all expected keys and that the gradient_score,
    binary_indicator, and page comparisons are computed as expected.
    """
    uid = "test_uid"
    sync_window = 0
    hparams = SimpleNamespace(
        batch_size=2,
        sequence_length=10,
        pages_per_window=1,
        validator_sample_rate=1.0
    )
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    config = {}
    model = MockModel()
    
    # Dummy optimizer and scheduler.
    optimizer = MockOptimizer(list(model.parameters()), lr=0.1)
    scheduler = MockScheduler(optimizer, step_size=10)
    transformer = MockTransformer()
    compressor = MockCompressor()
    # xshapes and totalks for the model parameters.
    xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
    totalks = {"layer1.weight": 50, "layer1.bias": 5}

    # state_dict with metadata pages so that pages match.
    state_dict = {
        "metadata": {
            "pages_info": [("dummy", 1, "A")]
        },
        # Dummy gradient keys can be omitted since apply_compressed_gradient will log missing data.
    }

    # Fake compute_average_loss to simulate loss before and after gradient application.
    # The order of calls in evaluate_peer:
    # 1. Own before gradient -> (4.0, 1, [0], 1)
    # 2. Own after gradient -> (2.0, 1, [0], 1)
    # 3. Random before gradient -> (5.0, 1, [0], 1)
    # 4. Random after gradient -> (5.0, 1, [0], 1)
    fake_results = iter([
         (4.0, 1, [0], 1),
         (2.0, 1, [0], 1),
         (5.0, 1, [0], 1),
         (5.0, 1, [0], 1)
    ])
    def fake_compute_average_loss(model_inst, batches, tokenizer_inst, device, sample_rate):
         return next(fake_results)
    import tplr.evaluation as evaluation_mod
    monkeypatch.setattr(evaluation_mod, "compute_average_loss", fake_compute_average_loss)

    # Patch R2DatasetLoader.get_loader for both "own" and "random" evaluation.
    async def fake_get_loader(window, hparams, tokenizer, seed=None, data_type="own", pack_samples=True):
        if data_type == "random":
            return (iter([[4, 5, 6]]), [("dummy_random", 1, "B")])
        else:
            return (iter([[1, 2, 3]]), [("dummy", 1, "A")])
    monkeypatch.setattr(R2DatasetLoader, "get_loader", fake_get_loader)

    # Patch R2DatasetLoader.next_pages to always return a fixed local pages value.
    async def fake_next_pages(offset, n_pages, seed):
        return [("dummy", 1, "A")]
    monkeypatch.setattr(R2DatasetLoader, "next_pages", fake_next_pages)

    # Call evaluate_peer.
    result = await evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer,
                                 config, model, transformer, compressor, xshapes, totalks,
                                 "cpu", 0.1, optimizer, scheduler)

    expected_keys = {
        "uid",
        "loss_before_per_batch_own",
        "loss_after_per_batch_own",
        "relative_improvement_own",
        "loss_before_per_batch_random",
        "loss_after_per_batch_random",
        "relative_improvement_random",
        "gradient_score",
        "binary_indicator",
        "miner_pages",
        "local_pages",
        "pages_random",
    }
    assert isinstance(result, dict), "Expected result to be a dictionary."
    assert expected_keys.issubset(result.keys()), (
        f"Missing expected keys in result. Got {result.keys()}"
    )

    # Check computed values from fake_compute_average_loss.
    # Own: loss improvement = 4.0 - 2.0 = 2.0 -> relative 0.5; Random: improvement = 0 -> relative 0.
    assert result["loss_before_per_batch_own"] == 4.0
    assert result["loss_after_per_batch_own"] == 2.0
    assert result["relative_improvement_own"] == 0.5
    assert result["loss_before_per_batch_random"] == 5.0
    assert result["loss_after_per_batch_random"] == 5.0
    assert result["relative_improvement_random"] == 0.0
    assert result["gradient_score"] == 0.5
    # Since 0.5 > 0.0, binary_indicator should be 1.
    assert result["binary_indicator"] == 1

    # Check page comparison.
    assert result["miner_pages"] == [("dummy", 1, "A")]
    assert result["local_pages"] == [("dummy", 1, "A")]
    # And for random evaluation, pages_random.
    assert result["pages_random"] == [("dummy_random", 1, "B")]


#############################################
# Test 11: test_evaluate_peer_division_by_zero
#############################################
@pytest.mark.asyncio
async def test_evaluate_peer_division_by_zero(monkeypatch):
    """
    Create a scenario where the computed loss_before values are 0 (e.g., dummy model always returns 0 loss).
    Ensure that evaluate_peer handles division-by-zero gracefully (by setting relative improvements,
    gradient_score to 0, etc.).
    """
    uid = "test_uid_div0"
    sync_window = 0
    hparams = SimpleNamespace(
        batch_size=2,
        sequence_length=10,
        pages_per_window=1,
        validator_sample_rate=1.0
    )
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    config = {}
    model = MockModel()
    optimizer = MockOptimizer(list(model.parameters()), lr=0.1)
    scheduler = MockScheduler(optimizer, step_size=10)
    transformer = MockTransformer()
    compressor = MockCompressor()
    xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
    totalks = {"layer1.weight": 50, "layer1.bias": 5}

    state_dict = {
        "metadata": {
            "pages_info": [("dummy", 1, "A")]
        }
    }

    # Fake compute_average_loss always returning 0 loss.
    def fake_compute_average_loss_zero(model_inst, batches, tokenizer_inst, device, sample_rate):
         return (0.0, 1, [0], 1)
    import tplr.evaluation as evaluation_mod
    monkeypatch.setattr(evaluation_mod, "compute_average_loss", fake_compute_average_loss_zero)

    # Patch R2DatasetLoader.get_loader for random evaluation.
    async def fake_get_loader_zero(window, hparams, tokenizer, seed=None, data_type="own", pack_samples=True):
        if data_type == "random":
            return (iter([[4, 5, 6]]), [("dummy_random", 1, "B")])
        else:
            return (iter([[1, 2, 3]]), [("dummy", 1, "A")])
    monkeypatch.setattr(R2DatasetLoader, "get_loader", fake_get_loader_zero)

    # Patch R2DatasetLoader.next_pages.
    async def fake_next_pages(offset, n_pages, seed):
        return [("dummy", 1, "A")]
    monkeypatch.setattr(R2DatasetLoader, "next_pages", fake_next_pages)

    # Call evaluate_peer.
    result = await evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer,
                                 config, model, transformer, compressor, xshapes, totalks,
                                 "cpu", 0.1, optimizer, scheduler)

    # In case of zero loss before, relative improvements should be 0 and gradient_score 0.
    assert result["loss_before_per_batch_own"] == 0.0
    assert result["loss_after_per_batch_own"] == 0.0
    assert result["relative_improvement_own"] == 0.0
    assert result["loss_before_per_batch_random"] == 0.0
    assert result["loss_after_per_batch_random"] == 0.0
    assert result["relative_improvement_random"] == 0.0
    assert result["gradient_score"] == 0.0
    # Since relative_improvement_own is not greater than relative_improvement_random, binary_indicator should be -1.
    assert result["binary_indicator"] == -1
    # Page comparisons should be preserved.
    assert result["miner_pages"] == [("dummy", 1, "A")]
    assert result["local_pages"] == [("dummy", 1, "A")]
    assert result["pages_random"] == [("dummy_random", 1, "B")]

