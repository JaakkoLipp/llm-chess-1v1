# LLM Chess Arena 1v1

Two OpenRouter models play chess with a live board, move log, per-player reasoning settings, and turn deadlines. One Python file; Python 3.10+.

```sh
python -m pip install python-chess
python llm_chess.py
```

Open http://127.0.0.1:8765, enter your OpenRouter API key and both model IDs, then click **Start game**. API usage costs OpenRouter credits; the key stays in server memory.

- Defaults: 120 seconds/turn, 8,192 tokens/request, no move cap.
- Retries share the turn deadline. Reasoning support varies by model.
- PGN and JSON logs save to `chess_games/`.
- `--demo`: random legal moves without API calls.
- `--port 8766`: use another port.
- **Ctrl+C** stops the server. Restart after updating the file; refreshing the browser is insufficient.
