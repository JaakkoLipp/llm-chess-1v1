#!/usr/bin/env python3
"""One-file local LLM chess arena. Python 3.10+.

Install:  python -m pip install python-chess
Run:      python llm_chess.py
Demo:     python llm_chess.py --demo

For real games enter an OpenRouter key in the local UI, or set OPENROUTER_API_KEY.
Open http://127.0.0.1:8765. Model IDs are entered in the browser.
PGN and JSON records are saved after every move in ./chess_games (configurable).
Keys entered in the UI are kept in server memory only, never in game files.
The app binds only to the local machine.

Rules: three attempts per turn, all sharing one monotonic deadline. Illegal
moves forfeit after three attempts; API failures interrupt the match. A deadline
expiry forfeits, except when the opponent has insufficient mating material.
Claimable draws are accepted automatically. The default move cap is 0 (unlimited).
Reaching an optional configured move cap
interrupts the game (it does not manufacture a draw). Stopped/unfinished games
have PGN result '*'. Stop/deadline cancels locally and ignores late responses;
the remote provider may still finish/bill an already submitted request.
"""

import argparse
import copy
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import secrets
import threading
import time
from urllib.parse import urlparse
import webbrowser

try:
    import chess
    import chess.pgn
    import chess.svg
except ImportError:
    raise SystemExit('Missing dependency. Run: python -m pip install python-chess')


class ApiError(Exception):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class Cancelled(Exception):
    pass


APP_VERSION = '1.1.1'
REASONING_EFFORTS = ('default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')


def provider_error_detail(raw):
    """Return a bounded, redacted provider message, never a raw response dump."""
    try:
        error = json.loads(raw).get('error', {})
        detail = error.get('message', '') if isinstance(error, dict) else ''
    except (ValueError, AttributeError, TypeError):
        return ''
    if not isinstance(detail, str):
        return ''
    key = os.environ.get('OPENROUTER_API_KEY', '')
    if key:
        detail = detail.replace(key, '[REDACTED]')
    detail = re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', detail)
    return ' '.join(detail.split())[:600]


class RequestControl:
    """Best-effort transport cancellation; the game deadline is independent."""
    def __init__(self):
        self.event = threading.Event()
        self.connection = None
        self.lock = threading.Lock()

    def attach(self, connection):
        with self.lock:
            if self.event.is_set():
                raise Cancelled()
            self.connection = connection

    def cancel(self):
        self.event.set()
        with self.lock:
            connection = self.connection
        if connection is not None:
            # Closing a socket cannot cancel provider-side computation.
            connection.close()


def openrouter(config, board, history, feedback, remaining, control):
    color = 'White' if board.turn else 'Black'
    prompt = (
        f'You are playing {color}. Choose the strongest legal chess move.\n'
        f'FEN: {board.fen()}\n'
        f'Moves played (SAN): {" ".join(history) or "None"}\n'
        f'Legal moves (UCI): {", ".join(m.uci() for m in board.legal_moves)}\n'
        'Reply only with a JSON object such as {"move":"e2e4"}. '
        'Use a legal UCI move, including a promotion suffix when required.'
    )
    if feedback:
        prompt += '\nPrevious attempt failed: ' + feedback
    payload = {
        'model': config['white'] if board.turn else config['black'],
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': config['max_tokens'],
        'stream': False,
    }
    effort = config.get('white_reasoning' if board.turn else 'black_reasoning', 'default')
    if effort != 'default':
        payload['reasoning'] = {'effort': effort}
    connection = http.client.HTTPSConnection('openrouter.ai', timeout=max(.05, remaining))
    control.attach(connection)
    try:
        connection.request('POST', '/api/v1/chat/completions', json.dumps(payload), {
            'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'],
            'Content-Type': 'application/json',
            'X-Title': 'Local LLM Chess',
        })
        response = connection.getresponse()
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ApiError('Provider response exceeded 2 MB.')
        if response.status != 200:
            detail = provider_error_detail(raw)
            message = f'OpenRouter HTTP {response.status}' + (': ' + detail if detail else '')
            raise ApiError(message, response.status in (408, 429) or response.status >= 500)
        data = json.loads(raw)
        if data.get('error'):
            raise ApiError('OpenRouter returned an error response.', True)
        try:
            choice = data['choices'][0]
            content = choice['message']['content']
        except (KeyError, IndexError, TypeError):
            raise ApiError('OpenRouter returned no completion.', True)
        return {'content': content or '', 'usage': data.get('usage', {}),
                'finish_reason': choice.get('finish_reason'),
                'model': data.get('model', payload['model']), 'id': data.get('id')}
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise ApiError(f'Transport/response error: {type(exc).__name__}', True) from exc
    finally:
        connection.close()


def demo_provider(config, board, history, feedback, remaining, control):
    if control.event.wait(1.2):
        raise Cancelled()
    return {'content': json.dumps({'move': random.choice(list(board.legal_moves)).uci()}),
            'usage': {}, 'model': 'demo/random-legal'}


def parse_move(content, board):
    if not isinstance(content, str):
        raise ValueError('Expected a text response containing a move.')
    text = content.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
        candidate = value.get('move') if isinstance(value, dict) else value
    except json.JSONDecodeError:
        # Accept an explicit JSON move surrounded by explanation or code fences.
        # Never guess a move from prose or pick among competing JSON moves.
        candidates = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', text):
            try:
                value, _ = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and 'move' in value:
                candidates.append(value['move'])
        if candidates:
            if not all(isinstance(item, str) for item in candidates) or len(set(candidates)) != 1:
                raise ValueError('Response contains conflicting moves. Return exactly one JSON move.')
            candidate = candidates[0]
        else:
            candidate = text
    if isinstance(candidate, str):
        candidate = candidate.strip()
    if not isinstance(candidate, str) or not re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', candidate):
        raise ValueError('Return only {"move":"e2e4"} with a legal UCI move.')
    move = chess.Move.from_uci(candidate)
    if move not in board.legal_moves:
        raise ValueError(f'{candidate} is illegal in this position; choose from the supplied legal moves.')
    return move


def response_move(response, board):
    content = response.get('content')
    truncated = response.get('finish_reason') in ('length', 'max_tokens')
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError('No move text was returned.')
        return parse_move(content, board)
    except ValueError:
        if truncated:
            raise ApiError('Token limit reached before a complete move was returned. '
                           'Increase Tokens / request or lower reasoning effort.')
        if not isinstance(content, str) or not content.strip():
            raise ApiError('Model returned no move text. Try a larger token budget or lower '
                           'reasoning effort; see finish_reason and usage in request details.')
        raise


class Arena:
    def __init__(self, directory, provider=openrouter, demo=False):
        self.directory = Path(directory)
        self.provider = provider
        self.demo = demo
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.board = chess.Board()
        self.game_id = None
        self.config = {}
        self.moves = []
        self.attempts = []
        self.status = 'idle'
        self.reason = 'Fresh board. Choose your two OpenRouter models to begin.'
        self.result = '*'
        self.deadline = None
        self.storage_error = None

    def set_key(self, value):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('Stop the match before changing the API key.')
            if not isinstance(value, str):
                raise ValueError('Enter an OpenRouter API key.')
            value = value.strip()
            if not 10 <= len(value) <= 1024 or any(ord(c) < 33 or ord(c) > 126 for c in value):
                raise ValueError('Enter a valid OpenRouter API key.')
            # Process environment only: not persisted, logged, or returned to the client.
            os.environ['OPENROUTER_API_KEY'] = value

    def start(self, values):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('Wait for the current game to stop before starting another.')
            config = {}
            for side in ('white', 'black'):
                model = str(values.get(side, '')).strip()
                if not model or len(model) > 200 or any(ord(c) < 32 for c in model):
                    raise ValueError(f'Enter a valid {side} model ID.')
                config[side] = model
                effort = values.get(side + '_reasoning', 'default')
                if effort not in REASONING_EFFORTS:
                    raise ValueError(f'Invalid reasoning effort for {side}.')
                config[side + '_reasoning'] = effort
            for key, default, low, high in [('turn_seconds', 120, 1, 600),
                                          ('max_moves', 0, 0, 1000),
                                          ('max_tokens', 8192, 64, 32768)]:
                try:
                    value = float(values.get(key, default))
                except (ValueError, TypeError):
                    raise ValueError(f'Invalid {key}.')
                if not math.isfinite(value) or not low <= value <= high or not value.is_integer():
                    raise ValueError(f'{key} must be an integer from {low} to {high}.')
                config[key] = int(value)
            if not self.demo and self.provider is openrouter and not os.environ.get('OPENROUTER_API_KEY', '').strip():
                raise ValueError('Add your OpenRouter API key above before starting a match.')
            self.directory.mkdir(parents=True, exist_ok=True)
            self.board = chess.Board()
            self.config = config
            self.game_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(3)
            self.moves, self.attempts = [], []
            self.status, self.reason, self.result = 'running', 'Starting game', '*'
            self.deadline, self.storage_error = None, None
            self.stop_event = threading.Event()
            self._save()
            if self.storage_error:
                self.status, self.reason = 'interrupted', 'Cannot save game files.'
                raise ValueError(self.reason + ' Check the data directory permissions.')
            self.worker = threading.Thread(target=self._run_guarded, daemon=True)
            self.worker.start()

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if self.status == 'running':
                self._finish('stopped', '*', 'Stopped by user')

    def _finish(self, status, result, reason):
        self.status, self.result, self.reason = status, result, reason
        self.deadline = None
        self._save()

    def _forfeit(self, reason):
        winner = not self.board.turn
        if self.board.has_insufficient_material(winner):
            self._finish('finished', '1/2-1/2', reason + '; opponent has insufficient mating material')
        else:
            self._finish('finished', '1-0' if winner else '0-1', reason)

    def _request(self, board, history, feedback, deadline):
        if self.stop_event.is_set():
            raise Cancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError()
        result_queue = queue.Queue(maxsize=1)
        control = RequestControl()

        def call():
            try:
                if control.event.is_set():
                    raise Cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                result = self.provider(self.config, board, history, feedback,
                                       remaining, control)
                result_queue.put((True, result))
            except Exception as exc:
                result_queue.put((False, exc))

        threading.Thread(target=call, daemon=True).start()
        try:
            while True:
                if self.stop_event.is_set():
                    raise Cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                try:
                    success, value = result_queue.get(timeout=min(.05, remaining))
                except queue.Empty:
                    continue
                if self.stop_event.is_set():
                    raise Cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                if success:
                    return value
                raise value
        finally:
            control.cancel()

    def _run_guarded(self):
        try:
            self._run()
        except Cancelled:
            return
        except Exception as exc:
            with self.lock:
                if self.status == 'running':
                    self._finish('interrupted', '*', f'Unexpected server error: {type(exc).__name__}')

    def _run(self):
        while not self.stop_event.is_set():
            with self.lock:
                outcome = self.board.outcome(claim_draw=True)
                if outcome:
                    self._finish('finished', outcome.result(), outcome.termination.name.replace('_', ' ').capitalize())
                    return
                if self.config['max_moves'] and len(self.moves) >= self.config['max_moves'] * 2:
                    self._finish('interrupted', '*', 'Maximum game length reached')
                    return
                board = self.board.copy()
                history = [m['san'] for m in self.moves]
                side = 'White' if board.turn else 'Black'
                model = self.config[side.lower()]
                started = time.monotonic()
                deadline = started + self.config['turn_seconds']
                self.deadline = deadline
                self.reason = side + ' is thinking'
            feedback = ''
            for attempt in range(1, 4):
                response, failure, move = {}, None, None
                api_failure, api_retryable = False, False
                attempt_started = time.monotonic()
                try:
                    response = self._request(board, history, feedback, deadline)
                    move = response_move(response, board)
                except Cancelled:
                    return
                except TimeoutError:
                    failure = 'Turn deadline exceeded'
                except ValueError as exc:
                    failure = str(exc)
                except ApiError as exc:
                    failure = str(exc)
                    api_retryable = exc.retryable
                    api_failure = True
                with self.lock:
                    if self.stop_event.is_set() or self.status != 'running':
                        return
                    elapsed = time.monotonic() - started
                    timed_out = time.monotonic() >= deadline
                    if timed_out:
                        failure, move = 'Turn deadline exceeded', None
                    entry = {'ply': len(self.moves) + 1, 'side': side, 'model': model,
                             'attempt': attempt, 'seconds': round(time.monotonic() - attempt_started, 3),
                             'response': response, 'error': failure}
                    self.attempts.append(entry)
                    if timed_out:
                        self._forfeit(side + ' lost on time')
                        return
                    if move is not None:
                        san = self.board.san(move)
                        self.board.push(move)
                        self.moves.append({'ply': len(self.moves) + 1, 'side': side, 'model': model,
                                           'san': san, 'uci': move.uci(), 'seconds': round(elapsed, 3),
                                           'fen': self.board.fen(), 'usage': response.get('usage', {})})
                        self.deadline = None
                        self._save()
                        break
                    # Only transport/provider errors are API failures; malformed moves are forfeits.
                    is_api = api_failure
                    if is_api and (not api_retryable or attempt == 3):
                        self._finish('interrupted', '*', failure)
                        return
                    if attempt == 3:
                        self._forfeit(side + ' returned three invalid moves')
                        return
                    feedback = failure
                    self.reason = f'{side}: retry {attempt + 1}/3'
                    self._save()
                # Backoff counts against the same turn deadline.
                if is_api:
                    self.stop_event.wait(min(.5 * attempt, max(0, deadline - time.monotonic())))

    def pgn(self):
        game = chess.pgn.Game()
        game.headers.update({'Event': 'Local LLM Chess' + (' (demo)' if self.demo else ''),
                             'Site': 'Local', 'Date': datetime.now().strftime('%Y.%m.%d'),
                             'White': self.config.get('white', '?'), 'Black': self.config.get('black', '?'),
                             'Result': self.result})
        if self.config:
            game.headers['TimeControl'] = '1/' + str(self.config['turn_seconds'])
            game.headers['WhiteReasoningEffort'] = self.config.get('white_reasoning', 'default')
            game.headers['BlackReasoningEffort'] = self.config.get('black_reasoning', 'default')
        node = game
        for record in self.moves:
            node = node.add_variation(chess.Move.from_uci(record['uci']))
            node.comment = f"Time taken: {record['seconds']:.3f}s"
        node.comment = (node.comment + ' ' + self.reason).strip()
        return str(game) + '\n'

    def record(self):
        return {'app_version': APP_VERSION, 'id': self.game_id, 'demo': self.demo, 'config': self.config,
                'status': self.status, 'result': self.result, 'reason': self.reason,
                'fen': self.board.fen(), 'moves': self.moves, 'attempts': self.attempts}

    def _save(self):
        if not self.game_id:
            return
        try:
            for suffix, content in [('json', json.dumps(self.record(), ensure_ascii=False, indent=2)), ('pgn', self.pgn())]:
                path = self.directory / f'{self.game_id}.{suffix}'
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_text(content, encoding='utf-8')
                temp.replace(path)
            self.storage_error = None
        except OSError as exc:
            self.storage_error = f'Could not save game: {type(exc).__name__}. Downloads remain available.'

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.record())
            last = self.board.peek() if self.board.move_stack else None
            state.update({'remaining': max(0, self.deadline - time.monotonic()) if self.deadline else None,
                          'active': 'White' if self.board.turn else 'Black',
                          'can_start': not (self.worker and self.worker.is_alive()),
                          'key_ready': self.demo or bool(os.environ.get('OPENROUTER_API_KEY', '').strip()),
                          'storage_error': self.storage_error,
                          'svg': chess.svg.board(self.board, lastmove=last,
                              check=self.board.king(self.board.turn) if self.board.is_check() else None,
                              size=560, colors={'square light': '#e9e5d9', 'square dark': '#769085',
                                                'square light lastmove': '#d9cf7f', 'square dark lastmove': '#b7b463'})})
            return state


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Chess Arena</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101917;color:#edf3ef}
*{box-sizing:border-box}body{margin:0;padding:32px;max-width:1260px;margin-inline:auto}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}h1{font-size:29px;margin:0}h2{font-size:17px;margin:0}p{color:#a9bcb2;font-size:14px;line-height:1.5}
.tag{font-size:12px;color:#abd5bd;border:1px solid #3a5748;padding:6px 10px;border-radius:20px}
.layout{display:grid;grid-template-columns:minmax(320px,1.2fr) minmax(320px,1fr);gap:24px}.panel{background:#192521;border:1px solid #30443a;border-radius:16px;padding:20px}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}.small-fields{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
label{display:block;font-size:12px;color:#b2c7ba}input,select{width:100%;padding:11px;margin-top:6px;background:#101917;border:1px solid #40584b;border-radius:7px;color:#fff;min-width:0}input:focus,select:focus{outline:2px solid #94cba9}
button,.download{border:1px solid #466050;border-radius:7px;background:#263c30;color:#eff8f1;padding:10px 16px;cursor:pointer;font:inherit;font-size:13px;text-decoration:none}
button.primary{background:#acd6ba;color:#102519;border:0;font-weight:650}button:disabled{opacity:.4;cursor:not-allowed}.actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#board{max-width:560px;margin:0 auto}#board svg{display:block;width:100%;height:auto}.player{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:13px 5px;font-size:14px;overflow-wrap:anywhere}.player span{color:#a9bcb2;font-size:12px}
.clock{font-size:32px;font-variant-numeric:tabular-nums;font-weight:650}.status{display:flex;justify-content:space-between;align-items:center;margin:18px 0 8px;gap:12px}.status p{margin:5px 0}.bar{height:4px;background:#30443a;border-radius:5px;margin-bottom:20px;overflow:hidden}#fill{height:100%;background:#acd6ba;width:0;transition:width .1s linear}
.log{max-height:300px;overflow:auto;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th{color:#a8beb0;font-size:11px;text-transform:uppercase;letter-spacing:1px;position:sticky;top:0;background:#192521}td,th{padding:10px 7px;border-bottom:1px solid #30443a}td small{color:#a4b7ab;margin-left:8px;font-variant-numeric:tabular-nums}.empty{padding:32px 0;text-align:center;color:#8aa394}.notice{color:#e8bc82;min-height:20px;font-size:13px;margin:12px 0}details{margin-top:16px;font-size:13px;color:#a9bcb2}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:220px;overflow:auto;font-size:11px}summary{cursor:pointer}.footnote{font-size:12px;margin-bottom:0}
@media(max-width:800px){body{padding:18px}.layout{grid-template-columns:1fr}header{gap:10px}.fields{grid-template-columns:1fr}.small-fields{grid-template-columns:repeat(3,1fr)}}
</style>
<header><div><h1>LLM Chess Arena</h1><p>Two models. One board. Every move on the clock.</p></div><span class="tag" id="mode">LOCAL</span></header>
<main class="layout"><section class="panel"><div class="player"><b id="blackLabel">Black</b><span>BLACK</span></div><div id="board" aria-label="Live chessboard"></div><div class="player"><b id="whiteLabel">White</b><span>WHITE</span></div></section>
<section><div class="panel"><h2>OpenRouter 1v1</h2>
<form id="connection" style="margin-top:16px"><label>OpenRouter API key<input id="api_key" type="password" autocomplete="off" spellcheck="false" placeholder="Paste your OpenRouter key" maxlength="1024" required></label><div class="actions" style="margin-top:10px"><button id="connect" type="submit">Use key for this session</button><span id="connectionStatus" class="footnote">Key required</span></div><p class="footnote">Kept in server memory for this session. Sent only to this local server and OpenRouter; never saved in game logs. Model requests use your OpenRouter credits.</p></form>
<form id="setup"><div class="fields"><label>White model ID<input id="white" required placeholder="Enter an OpenRouter model ID" maxlength="200"></label><label>Black model ID<input id="black" required placeholder="Enter an OpenRouter model ID" maxlength="200"></label></div>
<div class="fields"><label>White reasoning effort<select id="white_reasoning"><option value="default">Model default</option><option value="none">None</option><option value="minimal">Minimal</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="xhigh">Extra high</option><option value="max">Max</option></select></label><label>Black reasoning effort<select id="black_reasoning"><option value="default">Model default</option><option value="none">None</option><option value="minimal">Minimal</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="xhigh">Extra high</option><option value="max">Max</option></select></label></div>
<p class="footnote">Choose an effort supported by each model. Model default leaves reasoning unchanged. Higher effort may need more tokens and a longer turn limit; reasoning usually shares the request's token budget.</p>
<div class="small-fields"><label>Seconds / turn<input id="turn_seconds" type="number" value="120" min="1" max="600" required></label><label>Move cap (0 = none)<input id="max_moves" type="number" value="0" min="0" max="1000" required></label><label>Tokens / request<input id="max_tokens" type="number" value="8192" min="64" max="32768" required></label></div>
<div class="actions"><button class="primary" id="start">Start game</button><button type="button" id="stop" disabled>Stop</button></div></form><div class="notice" id="notice" role="alert"></div>
<p class="footnote">Starts from the standard position and plays automatically until a result. Retries share the turn deadline. Claimable draws are accepted automatically. Leave the move cap at 0 for a full match.</p></div>
<div class="panel" style="margin-top:18px"><div class="status"><div><h2 id="status" aria-live="polite">Ready</h2><p id="reason">Connecting to the local server…</p></div><div class="clock" id="clock">—</div></div><div class="bar"><div id="fill"></div></div>
<div class="log" id="log"><div class="empty">The first move will appear here.</div></div>
<div class="actions"><button id="pgn" type="button">Download PGN</button><button id="json" type="button">Download JSON log</button></div><details><summary>Request details &amp; errors</summary><pre id="details">No requests yet.</pre></details></div></section></main>
<script>
const token='__TOKEN__';const $=id=>document.getElementById(id);
let current=null,receivedAt=0,lastBoard='',lastMoves='',lastAttempts='',notice='',initialized=false;
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'X-Arena-Token':token,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});if(!r.ok){const e=await r.json();throw new Error(e.error||'Request failed')}return r;}
function render(s){current=s;receivedAt=performance.now();$('mode').textContent=(s.demo?'DEMO · RANDOM MOVES':'OPENROUTER · 1v1')+' · v'+s.app_version;
$('connection').hidden=s.demo;$('connectionStatus').textContent=s.key_ready?'Key loaded · verified when a match starts':'Key required';$('connect').disabled=!s.can_start;$('api_key').disabled=!s.can_start;
if(!initialized){for(const k of ['white','black','white_reasoning','black_reasoning','turn_seconds','max_moves','max_tokens']){if(s.config[k]!==undefined)$(k).value=s.config[k];else if((k==='white'||k==='black')&&s.demo)$(k).value='demo/random-legal';}initialized=true;}
if(s.svg!==lastBoard){$('board').innerHTML=s.svg;lastBoard=s.svg;}
$('whiteLabel').textContent=s.config.white||'White';$('blackLabel').textContent=s.config.black||'Black';
$('status').textContent=s.status==='running'?s.active+' to move':(s.result==='*'?s.status.charAt(0).toUpperCase()+s.status.slice(1):'Game over · '+s.result);
$('reason').textContent=s.reason;$('start').disabled=!s.can_start||!s.key_ready;$('start').textContent=s.id?'New game':'Start game';$('stop').disabled=s.status!=='running';
document.querySelectorAll('#setup input, #setup select').forEach(e=>e.disabled=!s.can_start);
$('notice').textContent=notice||s.storage_error||(!s.key_ready?'Add your OpenRouter key, choose both models, then start the match.':'');
const signature=JSON.stringify(s.moves);if(signature!==lastMoves){lastMoves=signature;const log=$('log');log.replaceChildren();if(!s.moves.length){const e=document.createElement('div');e.className='empty';e.textContent='The first move will appear here.';log.append(e);}else{const table=document.createElement('table');table.innerHTML='<thead><tr><th>Move</th><th>White</th><th>Black</th></tr></thead>';const body=document.createElement('tbody');for(let i=0;i<s.moves.length;i+=2){const tr=document.createElement('tr');const number=document.createElement('td');number.textContent=i/2+1;tr.append(number);for(const move of [s.moves[i],s.moves[i+1]]){const td=document.createElement('td');if(move){td.textContent=move.san;td.title=move.model+' · '+move.uci;const timing=document.createElement('small');timing.textContent=move.seconds.toFixed(1)+'s';td.append(timing);}else{td.textContent='—';}tr.append(td);}body.append(tr);}table.append(body);log.append(table);log.scrollTop=log.scrollHeight;}}
const attempts=JSON.stringify(s.attempts);if(attempts!==lastAttempts){lastAttempts=attempts;$('details').textContent=s.attempts.length?JSON.stringify(s.attempts,null,2):'No requests yet.';}
$('pgn').disabled=!s.id;$('json').disabled=!s.id;}
async function poll(){try{render(await(await api('/api/state')).json());}catch(e){$('notice').textContent='Connection lost: '+e.message;$('start').disabled=true;$('stop').disabled=true;current=null;$('clock').textContent='—';}setTimeout(poll,350);}
setInterval(()=>{if(!current)return;const r=current.remaining;if(r===null){$('clock').textContent='—';$('fill').style.width='0';return;}const left=Math.max(0,r-(performance.now()-receivedAt)/1000);$('clock').textContent=left.toFixed(1)+'s';$('fill').style.width=(100*left/current.config.turn_seconds)+'%';$('fill').style.background=left<5?'#e8a177':'#acd6ba';},80);
$('connection').onsubmit=async e=>{e.preventDefault();notice='';$('connect').disabled=true;try{await api('/api/key',{key:$('api_key').value});$('api_key').value='';render(await(await api('/api/state')).json());}catch(e){notice=e.message;}};
$('setup').onsubmit=async e=>{e.preventDefault();notice='';$('start').disabled=true;const body={};for(const k of ['white','black','white_reasoning','black_reasoning','turn_seconds','max_moves','max_tokens'])body[k]=$(k).value;try{await api('/api/start',body);}catch(e){notice=e.message;}};
$('stop').onclick=async()=>{try{await api('/api/stop',{});}catch(e){notice=e.message;}};
for(const ext of ['pgn','json'])$(ext).onclick=async()=>{try{const r=await api('/api/export/'+ext);const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;a.download=(current.id||'chess')+'.'+ext;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice=e.message;}};
poll();</script></html>'''


def make_handler(arena, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, value, content_type='application/json; charset=utf-8'):
            data = value.encode('utf-8') if isinstance(value, str) else json.dumps(value).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') not in (expected, f'localhost:{self.server.server_port}'):
                self.send(403, {'error': 'Use the local address printed by the server.'})
                return False
            if not secrets.compare_digest(self.headers.get('X-Arena-Token', ''), token):
                self.send(403, {'error': 'Reload the app to reconnect.'})
                return False
            return True

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                    return self.send(403, {'error': 'Invalid host'})
                return self.send(200, HTML.replace('__TOKEN__', token), 'text/html; charset=utf-8')
            if not self.authorized():
                return
            if path == '/api/state':
                return self.send(200, arena.snapshot())
            with arena.lock:
                if path == '/api/export/pgn':
                    return self.send(200, arena.pgn(), 'application/x-chess-pgn; charset=utf-8')
                if path == '/api/export/json':
                    return self.send(200, arena.record())
            self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Invalid request length')
                self.connection.settimeout(5)
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                if self.path == '/api/start':
                    arena.start(body)
                elif self.path == '/api/stop':
                    arena.stop()
                elif self.path == '/api/key':
                    arena.set_key(body.get('key'))
                else:
                    return self.send(404, {'error': 'Not found'})
                self.send(200, {'ok': True})
            except (ValueError, OSError) as exc:
                self.send(400, {'error': str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demo', action='store_true', help='Use random legal moves; no API key or charges')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd() / 'chess_games')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    arena = Arena(args.data_dir, demo_provider if args.demo else openrouter, args.demo)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(arena, secrets.token_urlsafe(32)))
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'LLM Chess v{APP_VERSION}: {url}\nSource: {Path(__file__).resolve()}\nMode: {"DEMO (random legal moves)" if args.demo else "OpenRouter"}\nLogs: {args.data_dir.resolve()}\nPress Ctrl+C to stop.')
    if not args.demo and not os.environ.get('OPENROUTER_API_KEY'):
        print('Enter your OpenRouter API key and both model IDs in the local browser UI.')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        arena.stop()
        server.server_close()


if __name__ == '__main__':
    main()
