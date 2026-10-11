import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.download_numerai import download_file, download_url, verify_file


class Response:
    def __init__(self, body=b"", status=200, headers=None, payload=None):
        self.body, self.status_code = body, status
        self.headers, self.payload = headers or {}, payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        yield self.body

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, url, **kwargs):
        self.get_arguments = kwargs
        return self.response

    def post(self, url, **kwargs):
        self.post_arguments = kwargs
        return self.response


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.bin"
        self.body = b"numerai fixture"
        self.expected = {"bytes": len(self.body), "sha256": hashlib.sha256(self.body).hexdigest()}

    def test_verify_checks_size_and_contents(self):
        self.assertFalse(verify_file(self.path, self.expected))
        self.path.write_bytes(self.body)
        self.assertTrue(verify_file(self.path, self.expected))
        self.path.write_bytes(b"x" * len(self.body))
        self.assertFalse(verify_file(self.path, self.expected))

    @patch("experiments.download_numerai.download_url", return_value="https://example.invalid/file")
    def test_resume_range_and_install_only_verified_file(self, _):
        partial = self.path.with_name(self.path.name + ".part")
        partial.write_bytes(self.body[:4])
        session = Session(Response(self.body[4:], 206, {"Content-Range": f"bytes 4-{len(self.body)-1}/{len(self.body)}"}))
        download_file(session, "input.bin", self.path, self.expected)
        self.assertEqual(session.get_arguments["headers"], {"Range": "bytes=4-"})
        self.assertEqual(self.path.read_bytes(), self.body)
        self.assertFalse(partial.exists())

    @patch("experiments.download_numerai.download_url", return_value="https://example.invalid/file")
    def test_server_ignoring_range_restarts_partial(self, _):
        self.path.with_name(self.path.name + ".part").write_bytes(self.body[:4])
        download_file(Session(Response(self.body)), "input.bin", self.path, self.expected)
        self.assertEqual(self.path.read_bytes(), self.body)

    @patch("experiments.download_numerai.download_url", return_value="https://example.invalid/file")
    def test_invalid_download_does_not_create_final_file(self, _):
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            download_file(Session(Response(b"bad")), "input.bin", self.path, self.expected)
        self.assertFalse(self.path.exists())
        self.assertTrue(self.path.with_name(self.path.name + ".part").exists())

    def test_api_requests_fixed_version_and_handles_errors(self):
        session = Session(Response(payload={"data": {"dataset": "https://example.invalid/file"}}))
        self.assertEqual(download_url(session, "features.json"), "https://example.invalid/file")
        self.assertEqual(session.post_arguments["json"]["variables"]["filename"], "v5.3/features.json")
        with self.assertRaises(RuntimeError):
            download_url(Session(Response(payload={"data": None, "errors": [{}]})), "features.json")
