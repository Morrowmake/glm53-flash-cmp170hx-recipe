"""Readiness retries must stop at the original deadline and first reply."""
import errno
import importlib.util
import io
from pathlib import Path
import urllib.error
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    'recipe_boot_check', Path(__file__).resolve().parents[2] / 'boot_check.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


@pytest.fixture
def client(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(boot.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(boot.time, 'sleep', lambda duration: now.__setitem__(0, now[0] + duration))
    result = boot.Client('http://127.0.0.1:8000', 'test')
    result.ready_deadline = 13.0
    result.opener = Mock()
    return result


def unavailable(path='/metrics', code=503):
    return urllib.error.HTTPError(path, code, 'not ready', {}, io.BytesIO())


@pytest.mark.parametrize('path,body', [('/health', None), ('/metrics', None),
                                     ('/v1/chat/completions', {'model': 'test'})])
def test_startup_503_then_ready(client, path, body):
    response = object()
    client.opener.open.side_effect = [unavailable(path), response]
    assert client.open(path, body) is response
    assert client.opener.open.call_count == 2
    assert client.opener.open.call_args.kwargs['timeout'] == 2.0


@pytest.mark.parametrize('body', [None, {'model': 'test'}])
def test_refused_connection_then_ready(client, body):
    response = object()
    client.opener.open.side_effect = [urllib.error.URLError(
        OSError(errno.ECONNREFUSED, 'connection refused')), response]
    assert client.open('/v1/chat/completions', body) is response
    assert client.opener.open.call_count == 2


def test_unavailable_expires_without_resetting_deadline(client):
    client.opener.open.side_effect = unavailable()
    with pytest.raises(boot.CheckError):
        client.open('/metrics')
    assert client.opener.open.call_count == 3
    assert client.ready_deadline == 13.0


@pytest.mark.parametrize('code', [400, 401, 403, 404, 500])
def test_semantic_errors_are_immediately_fatal(client, code):
    client.opener.open.side_effect = unavailable(code=code)
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1


def test_uncertain_post_is_never_replayed(client):
    client.opener.open.side_effect = ConnectionResetError(errno.ECONNRESET, 'reset')
    with pytest.raises(ConnectionResetError):
        client.open('/v1/chat/completions', {'model': 'test'})
    assert client.opener.open.call_count == 1


def test_after_first_reply_503_is_fatal(client):
    client.ready_retry = False
    client.opener.open.side_effect = unavailable()
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1


def test_without_startup_mode_503_is_fatal(client):
    client.ready_deadline = None
    client.opener.open.side_effect = unavailable()
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1
