from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import evaluate_ollama_top_n_no_none as subject


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode('utf-8')

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def http_error(code: int, details: str) -> HTTPError:
    return HTTPError(
        'http://localhost:11434/api/chat',
        code,
        'error',
        hdrs=None,
        fp=io.BytesIO(details.encode('utf-8')),
    )


class OllamaClientRetryTests(unittest.TestCase):
    def test_retries_http_500_and_returns_the_next_response(self) -> None:
        client = subject.OllamaClient(
            'http://localhost:11434',
            timeout_seconds=30,
            max_retries=2,
            retry_backoff_seconds=0.25,
        )

        with (
            patch.object(
                subject,
                'urlopen',
                side_effect=[
                    http_error(500, '{"error":"unexpected EOF"}'),
                    FakeResponse({'done': True}),
                ],
            ) as mocked_urlopen,
            patch.object(subject.time, 'sleep') as mocked_sleep,
        ):
            response = client.request_json('/api/chat', {'model': 'test'})

        self.assertEqual(response, {'done': True})
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once_with(0.25)

    def test_does_not_retry_a_non_transient_http_error(self) -> None:
        client = subject.OllamaClient(
            'http://localhost:11434',
            timeout_seconds=30,
            max_retries=4,
            retry_backoff_seconds=0.25,
        )

        with (
            patch.object(
                subject,
                'urlopen',
                side_effect=http_error(400, '{"error":"bad request"}'),
            ) as mocked_urlopen,
            patch.object(subject.time, 'sleep') as mocked_sleep,
            self.assertRaisesRegex(RuntimeError, 'HTTP 400'),
        ):
            client.request_json('/api/chat', {'model': 'test'})

        self.assertEqual(mocked_urlopen.call_count, 1)
        mocked_sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
