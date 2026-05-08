#!/usr/bin/env python3
"""Unit tests for gendata.py MiniMax integration."""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add the directory to path so we can import gendata
sys.path.insert(0, os.path.dirname(__file__))
import gendata


class TestMiniMaxDefaults(unittest.TestCase):
    """Test MiniMax default configuration values."""

    def test_default_model_defined(self):
        self.assertEqual(gendata.DEFAULT_MINIMAX_MODEL, 'MiniMax-M2.7')

    def test_default_claude_model_unchanged(self):
        self.assertEqual(gendata.DEFAULT_CLAUDE_MODEL, 'claude-sonnet-4-20250514')

    def test_default_ollama_model_unchanged(self):
        self.assertEqual(gendata.DEFAULT_MODEL, 'gemma2:9b')


class TestMiniMaxJson(unittest.TestCase):
    """Test minimax_json function."""

    def setUp(self):
        gendata.total_input_tokens = 0
        gendata.total_output_tokens = 0

    def test_returns_none_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove MINIMAX_API_KEY if it exists
            os.environ.pop('MINIMAX_API_KEY', None)
            result = gendata.minimax_json('MiniMax-M2.7', 'test prompt')
            self.assertIsNone(result)

    @patch('gendata.urllib.request.urlopen')
    def test_successful_json_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"pairs": [{"q": "is it big", "a": "YES"}]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            result = gendata.minimax_json('MiniMax-M2.7', 'test prompt')

        self.assertIsNotNone(result)
        self.assertIn('pairs', result)
        self.assertEqual(gendata.total_input_tokens, 10)
        self.assertEqual(gendata.total_output_tokens, 20)

    @patch('gendata.urllib.request.urlopen')
    def test_correct_request_url(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test')

        # Verify the URL used
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.full_url, 'https://api.minimax.io/v1/chat/completions')

    @patch('gendata.urllib.request.urlopen')
    def test_custom_base_url(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {
            'MINIMAX_API_KEY': 'test-key',
            'MINIMAX_BASE_URL': 'https://api.minimaxi.com/v1'
        }):
            gendata.minimax_json('MiniMax-M2.7', 'test')

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.full_url, 'https://api.minimaxi.com/v1/chat/completions')

    @patch('gendata.urllib.request.urlopen')
    def test_authorization_header(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'my-secret-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test')

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.get_header('Authorization'), 'Bearer my-secret-key')

    @patch('gendata.urllib.request.urlopen')
    def test_request_body_structure(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test prompt', max_tokens=100)

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode('utf-8'))

        self.assertEqual(body['model'], 'MiniMax-M2.7')
        self.assertEqual(body['max_tokens'], 100)
        self.assertIn('messages', body)
        self.assertEqual(len(body['messages']), 1)
        self.assertEqual(body['messages'][0]['role'], 'user')
        self.assertIn('JSON only', body['messages'][0]['content'])

    @patch('gendata.urllib.request.urlopen')
    def test_temperature_default_is_one(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test')

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode('utf-8'))
        self.assertEqual(body['temperature'], 1.0)

    @patch('gendata.urllib.request.urlopen')
    def test_temperature_zero_clamped(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test', temperature=0)

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode('utf-8'))
        self.assertGreater(body['temperature'], 0)

    @patch('gendata.urllib.request.urlopen')
    def test_temperature_above_one_clamped(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test', temperature=2.0)

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode('utf-8'))
        self.assertLessEqual(body['temperature'], 1.0)

    @patch('gendata.urllib.request.urlopen')
    def test_strips_markdown_code_blocks(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '```json\n{"result": "ok"}\n```'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            result = gendata.minimax_json('MiniMax-M2.7', 'test')

        self.assertIsNotNone(result)
        self.assertEqual(result['result'], 'ok')

    @patch('gendata.urllib.request.urlopen')
    def test_strips_think_tags(self, mock_urlopen):
        """MiniMax may return <think>...</think> chain-of-thought tags."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '<think>\nLet me think about this...\n</think>\n{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            result = gendata.minimax_json('MiniMax-M2.7', 'test')

        self.assertIsNotNone(result)
        self.assertEqual(result['result'], 'ok')

    @patch('gendata.urllib.request.urlopen')
    def test_handles_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            result = gendata.minimax_json('MiniMax-M2.7', 'test')

        self.assertIsNone(result)

    @patch('gendata.urllib.request.urlopen')
    def test_handles_invalid_json_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "not valid json"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            result = gendata.minimax_json('MiniMax-M2.7', 'test')

        self.assertIsNone(result)

    @patch('gendata.urllib.request.urlopen')
    def test_no_response_format_in_request(self, mock_urlopen):
        """Verify response_format is not sent (MiniMax doesn't support it)."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'}):
            gendata.minimax_json('MiniMax-M2.7', 'test')

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data.decode('utf-8'))
        self.assertNotIn('response_format', body)


class TestMiniMaxCLI(unittest.TestCase):
    """Test CLI argument parsing for --minimax flag."""

    def test_minimax_flag_exits_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('MINIMAX_API_KEY', None)
            with patch('sys.argv', ['gendata.py', '--topic', 'test', '--minimax']):
                with self.assertRaises(SystemExit):
                    gendata.main()

    def test_claude_and_minimax_flags_exist(self):
        """Verify both --claude and --minimax flags are available."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--claude', action='store_true')
        parser.add_argument('--minimax', action='store_true')
        args = parser.parse_args(['--minimax'])
        self.assertTrue(args.minimax)
        self.assertFalse(args.claude)


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration test with real MiniMax API (skipped if no API key)."""

    @unittest.skipUnless(os.environ.get('MINIMAX_API_KEY'), 'MINIMAX_API_KEY not set')
    def test_real_api_call(self):
        """Test actual MiniMax API call with a simple prompt."""
        gendata.total_input_tokens = 0
        gendata.total_output_tokens = 0

        result = gendata.minimax_json(
            'MiniMax-M2.7',
            'Generate 2 yes/no questions about elephants. Return JSON: {"pairs": [{"q": "question", "a": "YES"}]}',
            max_tokens=1024
        )

        self.assertIsNotNone(result, "MiniMax API returned None - check API key and connectivity")
        self.assertIn('pairs', result)
        self.assertGreater(len(result['pairs']), 0)
        self.assertGreater(gendata.total_input_tokens, 0)
        self.assertGreater(gendata.total_output_tokens, 0)


if __name__ == '__main__':
    unittest.main()
