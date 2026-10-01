#!/usr/bin/env python3
"""Two-request startup check; also supervises a directly launched server."""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

OK_PROMPT = 'Reply with exactly OK, with no punctuation or other text.'
DRAFT_PROMPT = ('Write a Python module with 100 simple arithmetic functions named '
                'add_1 through add_100. Each function adds its number to its argument. '
                'Output only the code, one function after another.')
COUNTERS = ('vllm:spec_decode_num_drafts_total',
            'vllm:spec_decode_num_draft_tokens_total',
            'vllm:spec_decode_num_accepted_tokens_total')
SUCCESS = 'vllm:request_success_total'
TRACKED = (*COUNTERS, SUCCESS)


class CheckError(RuntimeError):
    pass


def banner(message):
    print('[boot-check] ' + message, flush=True)


class Client:
    def __init__(self, base, model, key=''):
        self.base, self.model, self.key = base.rstrip('/'), model, key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, path, body=None, timeout=120):
        headers = {}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        if body is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(body).encode()
        return self.opener.open(urllib.request.Request(self.base + path, body, headers), timeout=timeout)

    def metrics(self):
        with self.open('/metrics', timeout=5) as response:
            return parse_metrics(response.read().decode(), self.model)

    def request(self, prompt, **kwargs):
        return self.open('/v1/chat/completions', dict(
            model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=0, reasoning_effort='low', **kwargs))


def parse_metrics(text, model):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([^\s{]+)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?', line)
        if not match or match[1] not in TRACKED:
            continue
        labels = match[2] or ''
        label = re.search(r'(?:\{|,)model_name=("(?:\\.|[^"\\])*")', labels)
        if label and json.loads(label[1]) != model:
            continue
        value = float(match[3])
        if not math.isfinite(value) or value < 0:
            raise CheckError('invalid spec-decode counter')
        result[(match[1], labels)] = value
    if {key[0] for key in result} != set(TRACKED):
        raise CheckError('missing spec-decode counters for served model')
    return result


def deltas(before, after):
    if before.keys() != after.keys() or any(after[k] < v for k, v in before.items()):
        raise CheckError('spec-decode counters reset or changed series during boot')
    return tuple(sum(after[k] - before[k] for k in before if k[0] == name) for name in COUNTERS)


def check(client, metrics_wait=60):
    banner('enabled: two temperature-0 requests')
    initial = client.metrics()
    with client.request(OK_PROMPT, max_tokens=128) as response:
        answer = json.load(response)['choices'][0]['message'].get('content')
    if not isinstance(answer, str) or answer.strip() != 'OK':
        raise CheckError('expected exactly OK after stripping surrounding whitespace')
    # Snapshot after the short reply: only the long request should satisfy the gate.
    deadline = time.monotonic() + metrics_wait
    while True:
        before = client.metrics()
        deltas(initial, before)
        if sum(before[k] - initial[k] for k in initial if k[0] == SUCCESS) >= 1:
            break
        if time.monotonic() >= deadline:
            raise CheckError('OK request completion did not reach metrics')
        time.sleep(1)
    done, chunks = False, 0
    stream_deadline = time.monotonic() + 180
    with client.request(DRAFT_PROMPT, stream=True, max_tokens=768,
                        min_tokens=768, ignore_eos=True) as response:
        for line in response:
            if time.monotonic() >= stream_deadline:
                raise CheckError('generation stream timeout')
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event:
                raise CheckError('generation stream returned an error')
            if event.get('choices'):
                chunks += 1
    if not done or not chunks:
        raise CheckError('generation stream was empty or incomplete')
    deadline = time.monotonic() + metrics_wait
    while True:
        drafts, proposed, accepted = deltas(before, client.metrics())
        if drafts >= 64 and proposed >= 64 and accepted > 0:
            banner(f'PASS: OK; drafts={drafts:g} proposed={proposed:g} accepted={accepted:g}')
            return
        if time.monotonic() >= deadline:
            raise CheckError(f'insufficient DFlash activity: drafts={drafts:g} '
                             f'proposed={proposed:g} accepted={accepted:g}; '
                             'need >=64 drafts and proposed tokens, >0 accepted')
        time.sleep(1)


def supervise(client, command, ready_timeout):
    child = subprocess.Popen(command, start_new_session=True)

    def interrupted(signum, frame):
        raise CheckError('server supervisor interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        deadline = time.monotonic() + ready_timeout
        while True:
            if child.poll() is not None:
                raise CheckError('server exited before health')
            try:
                with client.open('/health', timeout=5):
                    break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise CheckError('server health timeout')
                time.sleep(1)
        check(client)
        return child.wait()
    finally:
        # This group was created here; it contains only this launch's descendants.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ready-timeout', type=int, default=1800)
    parser.add_argument('--serve', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    client = Client(args.base, args.model, os.getenv('API_KEY', os.getenv('VLLM_API_KEY', '')))
    try:
        if args.serve:
            return supervise(client, args.serve, args.ready_timeout)
        check(client)
        return 0
    except Exception as error:
        # HTTP errors must not echo request headers or credentials.
        banner('FAIL: ' + (str(error) if isinstance(error, CheckError) else type(error).__name__))
        return 1


if __name__ == '__main__':
    sys.exit(main())
