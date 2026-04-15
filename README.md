# AutoWEAT

> An autonomous bias-landscape generator. A local LLM proposes Word Embedding
> Association Tests grounded in social-science theory. A locked Python script
> computes them. Results are pushed to a GitHub Pages site as they accrue.

AutoWEAT runs in the background of your workstation. Every few minutes it
asks a local model (through Ollama) to propose one WEAT test in a chosen
social-science domain — management, sociology, psychology, economics,
political science, anthropology, organizational behavior, STS — and then
hands the proposal to a locked Python implementation of Caliskan, Bryson &
Narayanan (2017) for computation. Tests that pass acceptance criteria are
appended to `docs/feed.json` and pushed to GitHub. The site renders the
running landscape as a Charlesworth-style scatter (effect size *d* against
−log₁₀ *p*), with click-through to a Caliskan Table 1–style detail readout
for any test.

The LLM never writes code. It only writes JSON. All math lives in
`autoweat/weat.py`, under version control, and is unit-tested against a
hand-built embedding space.

---

## Architecture

```
┌──────────────────┐    JSON proposal     ┌────────────────────┐
│  persona.md  ──► │ ───────────────────► │  weat.py (locked)  │
│  config.yaml ──► │  proposer.py         │  - in-vocab filter │
│  Ollama LLM      │                      │  - balance |X|=|Y| │
└──────────────────┘                      │  - cosine matrix   │
                                          │  - Caliskan eq. 3  │
                                          │  - perm test       │
                                          └─────────┬──────────┘
                                                    │
                                          accept iff |d| < 2
                                                    │
                                          ┌─────────▼──────────┐
                                          │  docs/feed.json    │
                                          │  git commit + push │
                                          └─────────┬──────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  GitHub Pages site │
                                          │  D3 scatter + UI   │
                                          └────────────────────┘
```

**Why the lock matters.** Karpathy's autoresearch principle: the LLM is
allowed to be creative about *what* to test, never about *how* to compute
it. Every effect size on the site comes from the same vetted code path, so
you can compare across hundreds of LLM-proposed tests without worrying that
the model invented a new statistic on iteration 47.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/santoshbs/autoweat.git
cd autoweat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get an embedding model

Default is the Caliskan 2017 original — Google News word2vec (300d, binary).
Download once and point `config.yaml` at the path:

```bash
mkdir -p ~/embeddings
# GoogleNews-vectors-negative300.bin.gz from the original word2vec release
# (3.4 GB; mirrors via gensim-data, kaggle, or huggingface)
gunzip ~/embeddings/GoogleNews-vectors-negative300.bin.gz
```

GloVe also works — convert once with `gensim.scripts.glove2word2vec` or
download a `.vec` mirror, then set `binary: false` in config.

### 3. Pull a proposer model into Ollama

Any instruction-tuned model with JSON mode works. Defaults to
`llama3.1:8b`; bigger is better for proposal quality:

```bash
ollama pull llama3.1:8b
# or
ollama pull qwen2.5:14b-instruct
```

### 4. Edit `config.yaml` and `persona.md`

`config.yaml` exposes every knob in one place — model, temperature, top_p,
top_k, num_ctx, repeat_penalty, sampling seed, embedding backend, embedding
path, permutation budget, GitHub Pages site metadata.

`persona.md` is the soul. It is concatenated into the system prompt verbatim,
so editing it changes how the LLM thinks about what's worth testing — no
code change required. Swap personas to change research sensibilities.

### 5. Run

```bash
# one iteration, no git push (good for first test)
python run.py --once --no-push

# real loop
python run.py
```

---

## What the LLM is allowed to do

- Pick a domain (rotates through a stoplist of recent ones)
- Pick a hypothesis grounded in a named theoretical construct
- Propose 8–25 single-token English words for each of X, Y, A, B
- Write a 2–4 sentence rationale citing the construct

## What the LLM is not allowed to do

- Write or modify any code in this repo
- Touch the WEAT computation
- Decide what counts as "in vocab"
- Set its own random seed
- Set the acceptance threshold

## Acceptance rules (locked)

A proposal is accepted iff:

1. After in-vocab filtering, every word set has at least 3 words
2. After balancing X and Y to equal size, |X| = |Y| ≥ 3
3. The resulting |*d*| is strictly less than 2
4. The test id (a hash of the sorted word lists + embedding backend) is
   not already in `feed.json`

The |*d*| < 2 cap is there because effect sizes near ±2 in WEAT almost
always indicate degenerate word lists — a single overwhelming term, or two
"target" sets that are actually the same concept under two labels. This
follows Caliskan et al.'s observation that real biases in their corpus
landed in roughly the [0.6, 1.8] range.

---

## The site

`docs/` is a GitHub Pages directory. Two files:

- `docs/index.html` — single self-contained page; D3 scatter, click-through
  detail panel styled after Caliskan et al. 2017 Table 1, per-domain color
  legend, recent-proposals list, soft auto-refresh every 5 minutes.
- `docs/feed.json` — the running list of accepted tests, newest first.

Enable GitHub Pages on the `docs/` folder of `main` and you're done.

---

## Math reference

```
s(w, A, B) = mean_{a∈A} cos(w, a) − mean_{b∈B} cos(w, b)             (1)

S(X, Y, A, B) = Σ_{x∈X} s(x, A, B) − Σ_{y∈Y} s(y, A, B)              (2)

d = ( mean_{x∈X} s(x, A, B) − mean_{y∈Y} s(y, A, B) )                (3)
    / std_dev_{w∈X∪Y} s(w, A, B)
```

`d` is reported with population standard deviation (`ddof=0`), matching the
Caliskan reference implementation. P-values are one-sided permutation
probabilities: exact enumeration of C(2*n*, *n*) partitions when *n* ≤ 10,
Monte Carlo over 100,000 random partitions otherwise.

---

## Tests

```bash
python tests/test_weat_sanity.py
```

Builds a 3-D embedding space by hand where the answer is known
(d should be +2 with p=0) and verifies the implementation. Also runs a
null case (X, Y, A all drawn from the same distribution) and verifies
the test does not falsely reject.

---

## Files

```
autoweat/
├── run.py                  # daemon entry point
├── config.yaml             # all knobs
├── persona.md              # the soul
├── requirements.txt
├── README.md
├── autoweat/
│   ├── __init__.py
│   ├── weat.py             # locked Caliskan 2017 computer
│   ├── embeddings.py       # gensim + ollama backends
│   └── proposer.py         # LLM proposal loop
├── tests/
│   └── test_weat_sanity.py
└── docs/                   # GitHub Pages directory
    ├── index.html          # the bias landscape UI
    └── feed.json           # the running results
```
