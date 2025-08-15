import logging
import unittest
from unittest.mock import MagicMock, patch

from tplr.logging import P, T, debug, setup_loki_logger, trace


class TestLogging(unittest.TestCase):
    def test_T(self):
        with patch("time.time", return_value=123.456):
            self.assertEqual(T(), 123.456)

    def test_P(self):
        self.assertEqual(
            P(1, 1.2345), "[steel_blue]1[/steel_blue] ([grey63]1.23s[/grey63])"
        )

    @patch("tplr.logging.logger")
    def test_debug(self, mock_logger):
        debug()
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    @patch("tplr.logging.logger")
    def test_trace(self, mock_logger):
        trace()
        mock_logger.setLevel.assert_called_once_with(5)

    @patch("tplr.logging.logging.getLogger")
    @patch("tplr.logging.logging.handlers.QueueHandler")
    @patch("tplr.logging.logging.handlers.QueueListener")
    @patch("tplr.logging.logging_loki.LokiHandler")
    def test_setup_loki_logger(
        self, mock_loki, mock_listener, mock_handler, mock_getLogger
    ):
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        logger = setup_loki_logger("test_service", "test_uid", "1.0")
        self.assertIsNotNone(logger)
        mock_loki.assert_called_once()


if __name__ == "__main__":
    unittest.main()
