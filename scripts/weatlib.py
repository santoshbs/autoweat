from __future__ import annotations

import itertools
import json
import re
import sqlite3
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize_token(term: str) -> str:
    token = re.sub(r"\s+", "_", term.strip().lower())
    return token


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def extract_text_member(zip_path: Path, destination: Path | None = None) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".txt")]
        if not members:
            raise ValueError(f"No .txt member found in {zip_path}")
        member = members[0]
        if destination is None:
            destination = zip_path.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, destination.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return destination


def build_sqlite_index(text_path: Path, index_path: Path) -> dict[str, Any]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()

    conn = sqlite3.connect(index_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE vocab (token TEXT PRIMARY KEY, offset INTEGER NOT NULL)")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    row_count = 0
    dimension = None
    batch: list[tuple[str, int]] = []

    with text_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            parts = line.rstrip().split(b" ")
            if len(parts) < 2:
                continue
            token = parts[0].decode("utf-8", errors="ignore")
            if dimension is None:
                dimension = len(parts) - 1
            batch.append((token, offset))
            row_count += 1
            if len(batch) >= 10000:
                conn.executemany("INSERT INTO vocab(token, offset) VALUES (?, ?)", batch)
                conn.commit()
                batch = []

    if batch:
        conn.executemany("INSERT INTO vocab(token, offset) VALUES (?, ?)", batch)
        conn.commit()

    meta = {
        "text_path": str(text_path),
        "row_count": str(row_count),
        "dimension": str(dimension or 0),
    }
    conn.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", list(meta.items()))
    conn.commit()
    conn.close()
    return {"text_path": str(text_path), "index_path": str(index_path), "row_count": row_count, "dimension": dimension}


class EmbeddingStore:
    def __init__(self, manifest: dict[str, Any]):
        self.text_path = Path(manifest["text_path"])
        self.index_path = Path(manifest["index_path"])
        self.conn = sqlite3.connect(self.index_path)
        self.file = self.text_path.open("rb")
        self._offset_cache: dict[str, int | None] = {}
        self._vector_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.conn.close()
        self.file.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def contains(self, token: str) -> bool:
        return self.lookup_offset(token) is not None

    def lookup_offset(self, token: str) -> int | None:
        if token in self._offset_cache:
            return self._offset_cache[token]
        row = self.conn.execute("SELECT offset FROM vocab WHERE token = ?", (token,)).fetchone()
        offset = None if row is None else int(row[0])
        self._offset_cache[token] = offset
        return offset

    def get_vector(self, token: str) -> np.ndarray:
        if token in self._vector_cache:
            return self._vector_cache[token]
        offset = self.lookup_offset(token)
        if offset is None:
            raise KeyError(token)
        self.file.seek(offset)
        line = self.file.readline().decode("utf-8").rstrip("\n")
        word, vector_text = line.split(" ", 1)
        if word != token:
            raise ValueError(f"Index mismatch for {token}: found {word}")
        vector = np.fromstring(vector_text, sep=" ", dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not norm:
            raise ValueError(f"Zero vector for token {token}")
        vector = vector / norm
        self._vector_cache[token] = vector
        return vector


def canonicalize_terms(raw_terms: list[str], store: EmbeddingStore, required_size: int) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        token = normalize_token(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        if store.contains(token):
            accepted.append(token)
        else:
            dropped.append(token)
        if len(accepted) >= required_size:
            break
    return accepted, dropped


def make_signature(x_terms: list[str], y_terms: list[str], a_terms: list[str], b_terms: list[str]) -> str:
    return "X=" + ",".join(x_terms) + "|Y=" + ",".join(y_terms) + "|A=" + ",".join(a_terms) + "|B=" + ",".join(b_terms)


def association_scores(target_matrix: np.ndarray, a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    return target_matrix @ a_matrix.T.mean(axis=1) - target_matrix @ b_matrix.T.mean(axis=1)


def exact_permutation_p_values(target_scores: np.ndarray, n_target: int, observed_stat: float) -> tuple[float, float]:
    total = float(target_scores.sum())
    stats = []
    for combo in itertools.combinations(range(target_scores.shape[0]), n_target):
        subset_sum = float(target_scores[list(combo)].sum())
        stats.append((2.0 * subset_sum) - total)
    stat_array = np.asarray(stats, dtype=np.float64)
    p_forward = float(np.mean(stat_array > observed_stat))
    p_reverse = float(np.mean(stat_array > (-observed_stat)))
    return p_forward, p_reverse


def compute_weat(
    x_terms: list[str],
    y_terms: list[str],
    a_terms: list[str],
    b_terms: list[str],
    store: EmbeddingStore,
) -> dict[str, Any]:
    x_matrix = np.vstack([store.get_vector(token) for token in x_terms])
    y_matrix = np.vstack([store.get_vector(token) for token in y_terms])
    a_matrix = np.vstack([store.get_vector(token) for token in a_terms])
    b_matrix = np.vstack([store.get_vector(token) for token in b_terms])

    s_x = association_scores(x_matrix, a_matrix, b_matrix)
    s_y = association_scores(y_matrix, a_matrix, b_matrix)
    observed_stat = float(s_x.sum() - s_y.sum())
    pooled_scores = np.concatenate([s_x, s_y])
    pooled_std = float(np.std(pooled_scores, ddof=1))
    effect_size = 0.0 if pooled_std == 0 else float((s_x.mean() - s_y.mean()) / pooled_std)
    p_forward, p_reverse = exact_permutation_p_values(pooled_scores, len(x_terms), observed_stat)

    return {
        "effect_size": effect_size,
        "abs_effect_size": abs(effect_size),
        "test_statistic": observed_stat,
        "p_forward": p_forward,
        "p_reverse": p_reverse,
        "supported_orientation": "x_to_a" if effect_size >= 0 else "y_to_a",
        "directional_p_value": p_forward if effect_size >= 0 else p_reverse,
    }


@dataclass
class EvaluatedProposal:
    payload: dict[str, Any]


def evaluate_proposal(
    proposal: dict[str, Any],
    store: EmbeddingStore,
    research_config: dict[str, Any],
    existing_signatures: set[str],
) -> dict[str, Any]:
    weat_cfg = research_config["weat"]
    target_size = int(weat_cfg["target_set_size"])
    attribute_size = int(weat_cfg["attribute_set_size"])

    x_terms, x_dropped = canonicalize_terms(proposal.get("x_terms", []), store, target_size)
    y_terms, y_dropped = canonicalize_terms(proposal.get("y_terms", []), store, target_size)
    a_terms, a_dropped = canonicalize_terms(proposal.get("a_terms", []), store, attribute_size)
    b_terms, b_dropped = canonicalize_terms(proposal.get("b_terms", []), store, attribute_size)

    final_sets = [x_terms, y_terms, a_terms, b_terms]
    if any(len(terms) < required for terms, required in ((x_terms, target_size), (y_terms, target_size), (a_terms, attribute_size), (b_terms, attribute_size))):
        return {
            "proposal_id": proposal.get("proposal_id", ""),
            "discipline": proposal.get("discipline", ""),
            "bias_name": proposal.get("bias_name", ""),
            "hypothesis": proposal.get("hypothesis", ""),
            "error": "Not enough in-vocabulary terms to form exact equal-sized sets.",
            "canonical_sets": {"x_terms": x_terms, "y_terms": y_terms, "a_terms": a_terms, "b_terms": b_terms},
            "dropped_terms": {"x_terms": x_dropped, "y_terms": y_dropped, "a_terms": a_dropped, "b_terms": b_dropped},
            "accepted": False,
            "rationale_code": "invalid_vocab",
        }

    all_terms = x_terms + y_terms + a_terms + b_terms
    if len(set(all_terms)) != len(all_terms):
        return {
            "proposal_id": proposal.get("proposal_id", ""),
            "discipline": proposal.get("discipline", ""),
            "bias_name": proposal.get("bias_name", ""),
            "hypothesis": proposal.get("hypothesis", ""),
            "error": "Final target and attribute sets overlap after canonicalization.",
            "canonical_sets": {"x_terms": x_terms, "y_terms": y_terms, "a_terms": a_terms, "b_terms": b_terms},
            "dropped_terms": {"x_terms": x_dropped, "y_terms": y_dropped, "a_terms": a_dropped, "b_terms": b_dropped},
            "accepted": False,
            "rationale_code": "overlapping_terms",
        }

    signature = make_signature(x_terms, y_terms, a_terms, b_terms)
    if signature in existing_signatures:
        return {
            "proposal_id": proposal.get("proposal_id", ""),
            "discipline": proposal.get("discipline", ""),
            "bias_name": proposal.get("bias_name", ""),
            "hypothesis": proposal.get("hypothesis", ""),
            "error": "Duplicate final signature.",
            "signature": signature,
            "canonical_sets": {"x_terms": x_terms, "y_terms": y_terms, "a_terms": a_terms, "b_terms": b_terms},
            "dropped_terms": {"x_terms": x_dropped, "y_terms": y_dropped, "a_terms": a_dropped, "b_terms": b_dropped},
            "accepted": False,
            "rationale_code": "duplicate_signature",
        }

    metrics = compute_weat(x_terms, y_terms, a_terms, b_terms, store)
    accepted = (
        metrics["abs_effect_size"] >= float(weat_cfg["min_abs_effect_size"])
        and metrics["directional_p_value"] < float(weat_cfg["p_value_threshold"])
    )
    if accepted:
        rationale = "large_and_significant"
    elif metrics["abs_effect_size"] < float(weat_cfg["min_abs_effect_size"]):
        rationale = "effect_too_small"
    else:
        rationale = "not_significant"

    return {
        "proposal_id": proposal.get("proposal_id", ""),
        "discipline": proposal.get("discipline", ""),
        "bias_name": proposal.get("bias_name", ""),
        "hypothesis": proposal.get("hypothesis", ""),
        "signature": signature,
        "canonical_sets": {"x_terms": x_terms, "y_terms": y_terms, "a_terms": a_terms, "b_terms": b_terms},
        "dropped_terms": {"x_terms": x_dropped, "y_terms": y_dropped, "a_terms": a_dropped, "b_terms": b_dropped},
        "metrics": metrics,
        "accepted": accepted,
        "rationale_code": rationale,
    }
