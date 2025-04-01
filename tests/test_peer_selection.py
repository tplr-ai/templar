import numpy as np
import pytest
import torch

import tplr
from neurons.validator import Validator

hparams = tplr.load_hparams()


class TestPeerSelection:
    """Tests for peer selection functionality."""

    @classmethod
    def assert_basic_requirements(
        cls,
        mock_validator: Validator,
        selected_peers: tplr.comms.PeerArray | None,
        initial_peers: tplr.comms.PeerArray,
    ):
        """Helper method to verify peer selection results.

        Args:
            mock_validator: The mock validator instance
            selected_peers: Array of selected peer UIDs
        """
        # Peers are successfully updated and are of the right type
        assert isinstance(selected_peers, np.ndarray)
        assert selected_peers.dtype == np.int64

        # All peers should be active
        assert set(selected_peers).issubset(set(mock_validator.comms.active_peers))

        # Number of peers is within range
        assert (
            mock_validator.hparams.minimum_peers
            <= len(selected_peers)
            <= mock_validator.hparams.max_topk_peers
        )

        # Verify peer list hasn't changed
        assert np.array_equal(mock_validator.comms.peers, initial_peers)

    class TestSelectNextPeers:
        """Tests for select_next_peers method."""

        def _setup_validator_state(
            self,
            mock_validator: Validator,
            num_non_zero_weights: int,
            peers_to_replace: int = 2,
            num_initial_peers: int = hparams.minimum_peers,
        ):
            """Helper to setup common validator test state.

            Args:
                mock_validator: The mock validator instance
                weight_indices: Tuple of (start, end) indices for non-zero weights, or single index
                peers_to_replace: Number of peers to replace
            """
            mock_validator.comms.peers = np.arange(num_initial_peers)
            mock_validator.hparams.peers_to_replace = peers_to_replace

            mock_validator.weights = torch.zeros(250)
            mock_validator.weights[:num_non_zero_weights] = torch.rand(
                num_non_zero_weights
            )
            mock_validator.weights[:num_non_zero_weights] /= mock_validator.weights[
                :num_non_zero_weights
            ].sum()

        @pytest.mark.parametrize(
            "num_non_zero_weights",
            [
                4 + hparams.max_topk_peers,  # Default case
                10 + hparams.max_topk_peers,  # More candidates
                250,  # Many candidates
            ],
        )
        @pytest.mark.parametrize(
            "peers_to_replace",
            [
                1,  # Minimum replacement
                2,  # Default case
                4,  # Maximum replacement
            ],
        )
        def test_successful_peer_update(
            self, mock_validator, num_non_zero_weights, peers_to_replace
        ):
            """Test successful peer update with sufficient candidates."""
            # Setup initial peers and weights
            self._setup_validator_state(
                mock_validator,
                num_non_zero_weights=num_non_zero_weights,
                peers_to_replace=peers_to_replace,
            )
            initial_peers = mock_validator.comms.peers

            # Call select_next_peers
            selected_peers = mock_validator.select_next_peers()

            TestPeerSelection.assert_basic_requirements(
                mock_validator=mock_validator,
                selected_peers=selected_peers,
                initial_peers=initial_peers,
            )

            # Verify some peers were replaced
            assert set(selected_peers) != {0, 1, 2, 3, 4}

            # Verify new peers came from candidates with non-zero weights
            new_peers = set(selected_peers) - {0, 1, 2, 3, 4}
            assert all(mock_validator.weights[peer] > 0 for peer in new_peers)

        def test_maintains_peer_list_size(self, mock_validator):
            """Test peer list size remains constant after update."""
            # Setup
            self._setup_validator_state(
                mock_validator,
                num_non_zero_weights=250,
                num_initial_peers=hparams.max_topk_peers,
            )
            initial_peers = mock_validator.comms.peers

            # Call select_next_peers
            selected_peers = mock_validator.select_next_peers()

            TestPeerSelection.assert_basic_requirements(
                mock_validator=mock_validator,
                selected_peers=selected_peers,
                initial_peers=initial_peers,
            )

            # Verify size hasn't changed
            assert len(selected_peers) == len(initial_peers)

        @pytest.mark.parametrize("seed", [42, 123, 456])
        def test_random_selection(self, mock_validator, seed):
            """Test that peer selection has random behavior."""
            # Setup
            self._setup_validator_state(mock_validator, num_non_zero_weights=10)

            # Do multiple updates and collect results
            results = []
            for idx in range(3):
                np.random.seed(idx)

                initial_peers = mock_validator.comms.peers

                # Call select_next_peers
                selected_peers = mock_validator.select_next_peers()

                TestPeerSelection.assert_basic_requirements(
                    mock_validator=mock_validator,
                    selected_peers=selected_peers,
                    initial_peers=initial_peers,
                )
                results.append(selected_peers)

            # Verify we got different peer lists
            assert not all(np.array_equal(r, results[0]) for r in results[1:])

    class TestSelectInitialPeers:
        """Tests for select_initial_peers method."""

        def _assert_all_miners_with_non_zero_incentive_in_peers(
            self, mock_validator: Validator, selected_peers: tplr.comms.PeerArray
        ):
            """Verify that all miners with non-zero incentive are included in peers."""
            non_zero_incentive_uids = np.nonzero(mock_validator.metagraph.I)[0]
            return set(non_zero_incentive_uids).issubset(selected_peers)

        @pytest.mark.parametrize("num_non_zero_incentive", [100, 200])
        def test_with_only_incentive_peers(self, mock_validator, top_incentive_indices):
            initial_peers = mock_validator.comms.peers

            # Call the actual method
            selected_peers = mock_validator.select_initial_peers()

            TestPeerSelection.assert_basic_requirements(
                mock_validator=mock_validator,
                selected_peers=selected_peers,
                initial_peers=initial_peers,
            )

            # Peers should be the top incentive miners
            assert set(selected_peers) == set(top_incentive_indices)

            # Verify peer count
            assert len(selected_peers) == mock_validator.hparams.max_topk_peers

        @pytest.mark.parametrize(
            "num_non_zero_incentive", [0, hparams.minimum_peers - 1]
        )
        def test_with_some_zero_incentive_peers(self, mock_validator: Validator):
            initial_peers = mock_validator.comms.peers

            # Call the actual method
            selected_peers = mock_validator.select_initial_peers()

            TestPeerSelection.assert_basic_requirements(
                mock_validator=mock_validator,
                selected_peers=selected_peers,
                initial_peers=initial_peers,
            )
            self._assert_all_miners_with_non_zero_incentive_in_peers(
                mock_validator, selected_peers
            )

            # Verify peer count
            assert len(selected_peers) == mock_validator.hparams.max_topk_peers

        @pytest.mark.parametrize(
            "num_non_zero_incentive",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        @pytest.mark.parametrize(
            "num_active_miners",
            [hparams.minimum_peers, hparams.max_topk_peers - 1],
            indirect=True,
        )
        def test_with_non_full_peer_list(self, mock_validator):
            initial_peers = mock_validator.comms.peers

            # Call the actual method
            selected_peers = mock_validator.select_initial_peers()

            TestPeerSelection.assert_basic_requirements(
                mock_validator=mock_validator,
                selected_peers=selected_peers,
                initial_peers=initial_peers,
            )
            self._assert_all_miners_with_non_zero_incentive_in_peers(
                mock_validator, selected_peers
            )

            # Assert that all active peers are in peers and that it's fewer than
            # max allowed
            assert len(selected_peers) < hparams.max_topk_peers

        @pytest.mark.parametrize(
            "num_non_zero_incentive",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        @pytest.mark.parametrize(
            "num_active_miners",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        def test_with_too_few_active_peers(self, mock_validator):
            # Call the actual method
            selected_peers = mock_validator.select_initial_peers()

            assert selected_peers is None

    @pytest.fixture
    def top_incentive_indices(self, mock_validator):
        incentives = mock_validator.metagraph.I
        return np.argsort(incentives)[::-1][: mock_validator.hparams.max_topk_peers]

    @pytest.fixture
    def num_non_zero_incentive(self):
        """Default fixture for number of non-zero incentive miners."""
        return 100  # Default value if not parameterized

    @pytest.fixture
    def num_active_miners(self, request):
        """Fixture for number of active miners.
        Returns parameterized value if available, otherwise returns default."""
        try:
            return request.param
        except AttributeError:
            return 250  # Default value if not parameterized

    @pytest.fixture
    def mock_metagraph(self, mocker, num_non_zero_incentive, num_miners=250):
        """Fixture that creates a mock metagraph with a specified number of miners and incentive distribution."""
        metagraph = mocker.Mock()

        metagraph.uids = np.arange(num_miners)

        # Create incentive distribution
        non_zero_incentives = np.random.rand(num_non_zero_incentive)
        non_zero_incentives /= non_zero_incentives.sum()  # Normalize to sum to 1
        zero_incentives = np.zeros(num_miners - num_non_zero_incentive)

        # Combine and shuffle incentives
        incentives = np.concatenate([non_zero_incentives, zero_incentives])
        np.random.shuffle(incentives)
        metagraph.I = incentives

        return metagraph

    @pytest.fixture
    def mock_validator(self, mocker, mock_metagraph, num_active_miners):
        # Initialize validator without calling the constructor
        validator = object.__new__(Validator)

        # Define necessary attributes
        validator.metagraph = mock_metagraph
        validator.hparams = hparams

        validator.comms = mocker.Mock(spec=["peers", "active_peers"])
        validator.comms.peers = None

        # Use the num_active_miners parameter from the fixture
        if num_active_miners > 0:
            validator.comms.active_peers = np.random.choice(
                a=validator.metagraph.uids,
                size=num_active_miners,
                replace=False,
            )
        else:
            validator.comms.active_peers = np.array([])
        tplr.logger.info(f"Created {len(validator.comms.active_peers)} active peers.")

        return validator
