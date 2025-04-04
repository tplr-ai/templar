import numpy as np
import pytest
import torch

import tplr
from neurons.validator import Validator

hparams = tplr.load_hparams()

NUM_MINERS = 250
NUM_NEURONS = 256


class TestPeerSelection:
    """Tests for peer selection functionality."""

    @staticmethod
    def active_non_zero_weight_uids(
        mock_validator: Validator,
    ) -> tplr.comms.PeerArray:
        non_zero_weight_uids = torch.nonzero(mock_validator.weights).flatten().numpy()
        return np.intersect1d(
            np.array(list(mock_validator.comms.active_peers)), non_zero_weight_uids
        )

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

        # Number of peers is within range
        assert (
            mock_validator.hparams.minimum_peers
            <= len(selected_peers)
            <= mock_validator.hparams.max_topk_peers
        )

        # Verify peer list hasn't changed
        assert np.array_equal(mock_validator.comms.peers, initial_peers)

        # All peers are active
        assert np.isin(
            selected_peers, np.array(list(mock_validator.comms.active_peers))
        ).all()

        # Number of peers is the smallest of max_topk_peers and number of
        # active peers
        assert len(selected_peers) == min(
            hparams.max_topk_peers, len(mock_validator.comms.active_peers)
        )

        # All active peers are selected, if there are <= max_topk_peers of them
        if len(mock_validator.comms.active_peers) <= hparams.max_topk_peers:
            assert np.isin(
                np.array(list(mock_validator.comms.active_peers)), selected_peers
            ).all(), (
                f"Not all active peers were selected: {len(selected_peers)=}: "
                f"{selected_peers}, {len(mock_validator.comms.active_peers)=}: "
                f"{mock_validator.comms.active_peers}"
            )

    class TestSelectNextPeers:
        """Tests for select_next_peers method."""

        def _setup_validator_state(
            self,
            mock_validator: Validator,
            num_non_zero_weights: int,
            peers_to_replace: int = 2,
            num_initial_peers: int = hparams.minimum_peers,
        ):
            """Helper to setup common validator test state."""
            mock_validator.comms.peers = np.random.choice(
                np.arange(NUM_MINERS), num_initial_peers, replace=False
            )
            mock_validator.hparams.peers_to_replace = peers_to_replace

            mock_validator.weights = torch.zeros(NUM_NEURONS)
            mock_validator.weights[:num_non_zero_weights] = torch.rand(
                num_non_zero_weights
            )
            mock_validator.weights[:num_non_zero_weights] /= mock_validator.weights[
                :num_non_zero_weights
            ].sum()

        @pytest.mark.parametrize(
            "num_non_zero_weights",
            [
                5,
                10,
                NUM_MINERS,
            ],
        )
        @pytest.mark.parametrize(
            "peers_to_replace",
            [
                1,  # Minimum replacement
                2,  # Default case
                4,
            ],
        )
        @pytest.mark.parametrize(
            "num_initial_peers",
            [
                5,
                10,
                hparams.max_topk_peers,
            ],
        )
        def test_successful_peer_update(
            self,
            mock_validator,
            num_non_zero_weights,
            peers_to_replace,
            num_initial_peers,
        ):
            """Test successful peer update with sufficient candidates."""
            tplr.logger.info(
                f"Using {num_initial_peers=}, {peers_to_replace=}, {num_non_zero_weights=}"
            )
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
            # Selected peers are not exactly current peers
            assert np.setdiff1d(selected_peers, mock_validator.comms.peers).size != 0

            # All active peers with non-zero weight are selected, if there are <=
            # max_topk_peers of them
            non_zero_weight_uids = (
                torch.nonzero(mock_validator.weights).flatten().numpy()
            )
            active_peers_with_weights = TestPeerSelection.active_non_zero_weight_uids(
                mock_validator
            )
            np.intersect1d(
                np.array(list(mock_validator.comms.active_peers)), non_zero_weight_uids
            )
            if len(active_peers_with_weights) <= hparams.max_topk_peers:
                assert np.isin(active_peers_with_weights, selected_peers).all(), (
                    f"Not all active peers with non-zero weights were selected: "
                    f"{len(selected_peers)=}: {selected_peers}, "
                    f"{len(active_peers_with_weights)=}: {active_peers_with_weights}"
                )

            # The correct number of peers are replaced, if we get
            # to the last (replacement) step
            active_peers_with_weights = TestPeerSelection.active_non_zero_weight_uids(
                mock_validator
            )
            if len(active_peers_with_weights) > hparams.max_topk_peers:
                num_replaced = np.setdiff1d(initial_peers, selected_peers).size
                assert num_replaced == min(
                    peers_to_replace,
                    active_peers_with_weights.size - hparams.max_topk_peers,
                )

        def test_random_selection(self, mock_validator):
            """Test that peer selection has random behavior."""
            # Setup
            self._setup_validator_state(
                mock_validator,
                num_non_zero_weights=NUM_MINERS,
                peers_to_replace=2,
                num_initial_peers=hparams.max_topk_peers,
            )

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

        def test_unsuccessful_peer_update(
            self,
            mock_validator,
        ):
            """Test unsuccessful peer update."""
            # Setup initial peers and weights
            self._setup_validator_state(
                mock_validator,
                num_non_zero_weights=hparams.max_topk_peers,
                peers_to_replace=2,
            )
            mock_validator.comms.peers = np.random.choice(
                a=TestPeerSelection.active_non_zero_weight_uids(mock_validator),
                size=hparams.max_topk_peers,
                replace=False,
            )
            initial_peers = mock_validator.comms.peers

            # Call select_next_peers
            selected_peers = mock_validator.select_next_peers()

            # Verify None is returned
            assert selected_peers is None

            # Verify peer list hasn't changed
            assert np.array_equal(mock_validator.comms.peers, initial_peers)

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
            return NUM_MINERS  # Default value if not parameterized

    @pytest.fixture
    def mock_metagraph(self, mocker, num_non_zero_incentive, num_miners=NUM_MINERS):
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
