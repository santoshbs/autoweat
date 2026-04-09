# AutoWEAT

`AutoWEAT` is an autoresearch prototype for discovering socially meaningful WEAT biases in word embeddings.

The fixed layer is:

- the Stanford GloVe embedding source
- Caliskan et al. (2017) WEAT math
- exact equal-size target permutations for one-sided `p` values
- large-effect and significance thresholds

The LLM layer only proposes candidate bias tests:

- target set `X`
- target set `Y`
- attribute set `A`
- attribute set `B`
- a discipline tag and a short rationale

## Current Design

- final evaluated sets are always equal-sized
- default evaluated size is `8` target words per side and `8` attribute words per side
- large effects mean `abs(effect_size) >= 0.8`
- significance means the directional Caliskan permutation `p < 0.05`
- the evaluator keeps the signed effect size and reports whether the supported direction matched the proposal

## Main Commands

Run these commands from the repository root.

Prepare the real Stanford source:

```bash
python3 scripts/auto_weat.py prepare --download
```

Prepare the tiny toy embeddings for a fast smoke test:

```bash
python3 scripts/auto_weat.py prepare --txt-path specs/toy_glove.txt
```

Run one manual round on the toy file:

```bash
python3 scripts/auto_weat.py round --backend manual --manual-proposals specs/manual_proposals.example.json
```

Run one Ollama round on the active embedding source:

```bash
python3 scripts/auto_weat.py round --backend ollama --model gpt-oss:120b --n-proposals 3 --ollama-timeout 1800 --ollama-think high
```

Run one Ollama round with Gemma 4:

```bash
python3 scripts/auto_weat.py round --backend ollama --model gemma4:latest --n-proposals 1 --ollama-timeout 1800 --ollama-think false
```

Run several rounds:

```bash
python3 scripts/auto_weat.py loop --backend ollama --model gpt-oss:120b --rounds 3 --n-proposals 3 --ollama-timeout 1800 --ollama-think high
```

Reset prior runs and leaderboard state while keeping the prepared embeddings:

```bash
python3 scripts/auto_weat.py reset
```

Run a fresh 10-round search with 1 proposal per round:

```bash
python3 scripts/auto_weat.py loop --backend ollama --model gpt-oss:120b --rounds 10 --n-proposals 1 --ollama-timeout 1800 --ollama-think high
```

Show current active source and best accepted finding:

```bash
python3 scripts/auto_weat.py status
```

## Output Files

- active embedding source: `state/source_manifest.json`
- tested proposals: `state/leaderboard.csv`
- structured round history: `state/history.csv`
- strongest accepted finding so far: `state/best_finding.json`
- dashboard page: `dashboard/index.html`

## Dashboard

The dashboard is a self-contained local HTML page that rebuilds automatically after:

- `prepare`
- `status`
- `reset`
- `round`
- `loop`

It lets you filter by run and discipline, sort by highest `|d|` or smallest `p`, inspect the exact final `X/Y/A/B` seed words for every finding, and auto-refresh every 15 seconds without losing your current filters.

## Stanford Source

Default source:

- [glove.2024.wikigiga.300d.zip](https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip)

This project expects to download the zip into:

- `data/raw/glove.2024.wikigiga.300d.zip`

and extract the text file into:

- `data/raw/glove.2024.wikigiga.300d.txt`

## Important Notes

- The evaluator is fixed; the LLM never changes the math.
- This is not trying to find literally every possible bias.
- It is trying to discover many large, significant, human-interpretable WEATs that would interest organizational, psychology, sociology, and related scholars.
- For `gemma4`, the runner uses the model-family sampling guidance (`temperature=1.0`, `top_p=0.95`, `top_k=64`).
- For `gemma4`, `--ollama-think true` enables thinking and `--ollama-think false` disables it. The default `auto` mode resolves to `false` for `gemma4`.
