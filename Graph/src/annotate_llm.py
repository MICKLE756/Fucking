#!/usr/bin/env python
"""Offline LLM reasoning-evidence annotation (Section 3.7).

This is the *teacher* side of the distillation described in the Method. It is
run **once, offline**, before (or between) training and produces a pickle cache
consumed by ``ReasoningAnnotationStore`` in ``src/method.py``. No LLM is ever
called during training or inference.

Pipeline
--------
1. Read raw dialogues (doc_id, utterances, speakers, gold emotions / pairs).
2. Select the candidate pairs to annotate (heuristic superset): every
   (emotion_i, cause_j) pair whose emotion utterance i is non-neutral and
   |i-j| <= window, plus all gold pairs. Over-annotating is harmless and
   intended: at training time the distillation loss only *reads* pairs that
   fall in the model's top-M evidence population, and the store returns
   ``valid=False`` for any (doc_id,i,j) key it was never given -- those pairs
   are simply skipped. So a superset that covers the top-M pairs is correct;
   the only cost is some wasted LLM calls on pairs the model never queries.

   (The paper's exact selection draws ``build_annotation_set`` = top-M ∩
   hard-pairs ∪ sample(easy) from the frozen warmup model theta0. That needs
   the full BERT + dataset + audio/video stack on a GPU box; the heuristic
   superset above is a dependency-free stand-in that provably covers it.)
3. For each selected pair, build a fixed **counterfactual-intervention** prompt with
   the real utterance text and query a **local Qwen2.5-7B-Instruct** k times
   (sampling). Parse the JSON scores [s_nec, s_suf, s_dir, s_spur] in [0,1] and
   estimate self-consistency rho.
4. Write {(doc_id, i, j): [s_nec, s_suf, s_dir, s_spur, rho]} to ``--out`` (pickle).

The four counterfactual scores are the teacher signal distilled into z^rea:
   s_nec  -- necessity: would i's emotion disappear if j were removed/neutralized?
   s_suf  -- sufficiency: is j alone enough to elicit i's emotion?
   s_dir  -- direction: 1 = j causes i, 0 = reverse / reaction.
   s_spur -- spuriousness: do i,j merely co-occur (no causal link)?
These are *textual* counterfactual judgements (not formal SCM/do-calculus); they
complement the model's perturbation-based necessity evidence z^nec in fusion.

The key (i, j) follows the model convention: i = emotion utterance, j = cause
utterance, both 0-indexed within the dialogue (see ``soft_relevance_gate`` /
``_distillation_loss``).

Usage
-----
    # (recommended) serve Qwen with vLLM, then annotate over HTTP:
    #   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    python -m src.annotate_llm --train data/dataset/train.txt \
        --vllm-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
        --k 5 --out data/llm_anno.pkl

    # in-process transformers backend (loads the model locally, slower):
    python -m src.annotate_llm --train data/dataset/train.txt \
        --qwen-path /path/to/Qwen2.5-7B-Instruct --out data/llm_anno.pkl

    # plumbing dry-run (no Qwen, no torch): emits deterministic stub scores so
    # you can validate the pickle format end-to-end:
    python -m src.annotate_llm --train data/dataset/train.txt \
        --dry-run --out /tmp/llm_anno_stub.pkl

Then point the trainer at the cache via ``src/config.yaml``:
    llm_anno_path: data/llm_anno.pkl
"""

import os
import re
import json
import time
import pickle as pkl
import hashlib
import argparse
import urllib.request
import urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOTE: torch / transformers are imported lazily inside QwenAnnotator so this
# module (and --dry-run / the vLLM HTTP backend) work without them installed.


# ---------------------------------------------------------------------------
# Data reading (kept independent of loader.py so the script has no heavy deps).
# ---------------------------------------------------------------------------
def read_dialogues(path):
    """Parse the dataset txt format used by loader.read_data.

    Returns a list of dicts: {doc_id, lines, speakers, emotions, pairs}
    where pairs are 1-indexed (emotion_utt, cause_utt) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    out, idx = [], 0
    while idx < len(data):
        scene_id, num_lines = map(int, data[idx].split(" "))
        pair_line = data[idx + 1].strip()
        if len(pair_line) > 0:
            pairs = [tuple(map(int, p.split(","))) for p in pair_line.strip("()").split("),(")]
        else:
            pairs = []
        lines, speakers, emotions = [], [], []
        for i in range(num_lines):
            parts = data[idx + 2 + i].split(" | ")
            _utt_id, speaker, emotion, utterance = parts[:4]
            speakers.append(speaker)
            emotions.append(emotion)
            lines.append(utterance)
        out.append({"doc_id": scene_id, "lines": lines, "speakers": speakers,
                    "emotions": emotions, "pairs": pairs})
        idx += 2 + num_lines
    return out


# ---------------------------------------------------------------------------
# Candidate selection.
# ---------------------------------------------------------------------------
def heuristic_candidates(dialogue, window=12):
    """Superset of plausible (i=emotion, j=cause) pairs, 0-indexed.

    Every non-neutral emotion utterance i is paired with:
      * the **self candidate** (i, i) -- the self-contained / intrinsic cause,
        added unconditionally (independent of `window`). In this corpus roughly
        half of the gold pairs are self-cause (cause == emotion utterance), so
        this candidate must never be dropped; it is scored via the dedicated
        self-cause branch of ``build_prompt``.
      * cause utterances j within `window` (causes usually precede / are near
        the emotion).
    Gold pairs are always included regardless of window or neutrality.
    """
    n = len(dialogue["lines"])
    cand = set()
    for i in range(n):
        emo = dialogue["emotions"][i].lower()
        if emo == "neutral":
            continue
        cand.add((i, i))                      # forced self-cause candidate
        lo, hi = max(0, i - window), min(n - 1, i + window)
        for j in range(lo, hi + 1):
            cand.add((i, j))
    for (e, c) in dialogue["pairs"]:          # gold pairs are 1-indexed
        if 1 <= e <= n and 1 <= c <= n:
            cand.add((e - 1, c - 1))
    return sorted(cand)


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------
# (#3) System prompts kept terse: redundancy dilutes a 7B's attention and wastes
# context. Each carries the one distinguishing instruction plus the (#4)
# scoring-spread note.
_SYS_CROSS = (
    "You are an expert annotator for Multimodal Emotion-Cause Pair Extraction. "
    "Judge COUNTERFACTUALLY whether one utterance causes the emotion expressed in "
    "another. Reason briefly step by step, then output the scores as a JSON object. "
    "Use the full [0,1] range to reflect strength; do not default to 0.5 unless a "
    "case is genuinely ambiguous."
)

_SYS_SELF = (
    "You are an expert annotator for Multimodal Emotion-Cause Pair Extraction. "
    "A cause may be ANOTHER utterance, OR the event/content described WITHIN the "
    "emotion utterance itself (a self-contained / intrinsic cause) -- in this "
    "corpus the self-contained case is very common, so do NOT assume an emotion "
    "utterance is a mere reaction by default. Reason COUNTERFACTUALLY about the "
    "EVENT described, not about deleting text. Reason briefly step by step, then "
    "output the scores as a JSON object. Use the full [0,1] range to reflect "
    "strength; do not default to 0.5 unless a case is genuinely ambiguous."
)

# (#1) Few-shot exemplars (format + behaviour anchors). Kept dialogue-independent
# so they are stable across pairs. Beyond the two extremes (strong / unrelated)
# the sets now cover the MIDDLE of the spectrum -- partial/contributing cause and
# REVERSE direction -- because a 7B drifts most on those, inflating sample
# variance (low rho). The self set leans on real corpus patterns: a described /
# reported event or a situational outburst is its own trigger, while only an
# explicit echo/agreement of a specific prior turn is a pure reaction.
_FEWSHOT_CROSS = (
    "Examples:\n"
    "- Clear cause -> later emotion (j directly triggers i; removing j removes "
    'the cause). -> {"s_nec": 0.85, "s_suf": 0.55, "s_dir": 0.95, "s_spur": 0.08}\n'
    "- Partial / contributing cause (j is one of several triggers; i would still "
    'feel some emotion without it). -> {"s_nec": 0.50, "s_suf": 0.30, "s_dir": 0.70, "s_spur": 0.40}\n'
    "- REVERSE direction (i was said first and j is a reaction to i, so j does NOT "
    'cause i). -> {"s_nec": 0.15, "s_suf": 0.10, "s_dir": 0.10, "s_spur": 0.55}\n'
    "- Same topic, no causal link (i would feel the same without j). -> "
    '{"s_nec": 0.12, "s_suf": 0.08, "s_dir": 0.30, "s_spur": 0.88}\n\n'
)
_FEWSHOT_SELF = (
    "Examples (candidate = the same utterance):\n"
    "- surprise (discovery): \"Then I look down, and I realize there is a phone ... there.\"\n"
    '  The realization described IS the trigger. -> {"s_nec": 0.85, "s_suf": 0.80, "s_dir": 0.95, "s_spur": 0.08}\n'
    "- sadness (reported event): \"I just found out I lost my job.\"\n"
    '  The described event is the trigger even though it is reported second-hand -- still self-contained. -> '
    '{"s_nec": 0.85, "s_suf": 0.75, "s_dir": 0.90, "s_spur": 0.12}\n'
    "- joy (situational outburst): \"Oh my gosh, you are finally here!\"\n"
    '  Delight at the present situation, not echoing any specific earlier line -- self-contained. -> '
    '{"s_nec": 0.70, "s_suf": 0.60, "s_dir": 0.80, "s_spur": 0.20}\n'
    "- partial self-cause (own content contributes, but a prior turn also matters):\n"
    '  -> {"s_nec": 0.55, "s_suf": 0.40, "s_dir": 0.60, "s_spur": 0.45}\n'
    "- pure reaction / agreement (echoes a specific prior turn, no event described): \"That is right.\"\n"
    '  The emotion is a reaction to the conversation, not triggered by this utterance own content. -> '
    '{"s_nec": 0.15, "s_suf": 0.10, "s_dir": 0.08, "s_spur": 0.88}\n\n'
)

# (#5) Explicit rules for the tricky cases that make a 7B guess inconsistently.
_RULES_SELF = (
    "Rules for tricky self-cause cases:\n"
    "- A described event, realization, discovery, or piece of (even reported / "
    "second-hand) news in the utterance counts as a self-contained cause: high "
    "s_dir, low s_spur.\n"
    "- A greeting, exclamation, or emotional outburst about the CURRENT situation "
    "(not echoing a specific earlier line) is self-contained -- the speaker's own "
    "reaction to the present moment is the cause: high s_dir.\n"
    "- Treat the utterance as a pure reaction (low s_dir, high s_spur) ONLY when it "
    "clearly ECHOES or AGREES WITH a specific prior utterance (e.g. 'That is right.', "
    "'Me too.').\n"
    "- If genuinely undecidable, score near 0.5 rather than guessing an extreme, and "
    "keep your scores consistent.\n\n"
)
_RULES_CROSS = (
    "Rules for tricky cases:\n"
    "- If i was said BEFORE j, or j is a reaction to i, the direction is reversed: "
    "low s_dir.\n"
    "- A reported / second-hand mention of an event in j can still cause i.\n"
    "- If genuinely undecidable, score near 0.5 rather than guessing an extreme, and "
    "keep your scores consistent.\n\n"
)

# (#2) Local-context window: instead of dumping the whole dialogue (redundant far
# context dilutes the j->i signal and inflates s_spur), keep only the turns around
# the pair. We anchor on BOTH endpoints and pad each side so the candidate j (and
# everything between i and j) is always retained -- never truncating away the
# cause itself. Short dialogues are unaffected (the window covers them whole).
_CTX_PAD = 8


def build_prompt(dialogue, i, j):
    """Build a **counterfactual-intervention** chat prompt. The four scores
    (necessity / sufficiency / direction / spuriousness) form the teacher signal
    distilled into z^rea.

    Two branches share the same 4-score JSON schema:
      * cross-utterance (j != i): is utterance j a cause of i's emotion?
      * self-cause (j == i): is i's emotion triggered by the event/content
        described within i itself? The counterfactual is over the described
        EVENT ("had it not happened"), not over deleting the sentence -- the
        latter is ill-posed when the candidate IS the emotion utterance and is
        exactly what made the symmetric prompt fail on self-cause pairs."""
    lines = dialogue["lines"]
    spk = dialogue["speakers"]
    n = len(lines)
    # (#2) windowed context around both i and j (keeps absolute indices intact so
    # the [k] references below still match), with ellipsis markers when truncated.
    lo = max(0, min(i, j) - _CTX_PAD)
    hi = min(n - 1, max(i, j) + _CTX_PAD)
    ctx = [f"[{k}] {spk[k]}: {lines[k]}" for k in range(lo, hi + 1)]
    if lo > 0:
        ctx.insert(0, "... (earlier turns omitted) ...")
    if hi < n - 1:
        ctx.append("... (later turns omitted) ...")
    convo = "\n".join(ctx)
    emo = dialogue["emotions"][i]

    if i == j:
        sys = _SYS_SELF
        user = (
            f"Conversation (index, speaker, text):\n{convo}\n\n"
            f"Emotion utterance: index {i} (\"{lines[i]}\"), emotion = {emo}.\n"
            f"Candidate cause: the SAME utterance {i} -- judge whether the emotion "
            "is triggered by the EVENT/CONTENT described WITHIN this utterance "
            "(self-contained cause), rather than being a reaction to some OTHER "
            "utterance.\n\n"
            f"{_FEWSHOT_SELF}"
            "Think step by step:\n"
            "1. What EVENT or SITUATION is described in the emotion utterance?\n"
            "2. If that event had NOT happened, would the emotion still arise "
            "from the surrounding conversation alone?\n"
            "3. Is the utterance content itself the trigger, or just a reaction?\n\n"
            f"{_RULES_SELF}"
            "Then score each dimension as a float in [0,1]:\n"
            "- s_nec (necessity): imagine the event/situation described in "
            f"utterance {i} had NOT happened. How likely would its emotion then "
            "DISAPPEAR or clearly weaken? 1 = its own content is necessary.\n"
            "- s_suf (sufficiency): is the content of this utterance ALONE enough "
            "to elicit its emotion, needing no other utterance? 1 = self-sufficient.\n"
            "- s_dir (direction): 1 = the emotion arises from THIS utterance's own "
            "content (self-contained cause); 0 = it is actually a reaction to some "
            "OTHER utterance.\n"
            "- s_spur (spuriousness): probability that the utterance carries the "
            "emotion but its own content is NOT the real trigger (pure reaction / "
            "topic only). 1 = own content not the cause.\n\n"
            'End your response with a JSON object: {"s_nec": <f>, "s_suf": <f>, "s_dir": <f>, "s_spur": <f>}'
        )
    else:
        sys = _SYS_CROSS
        user = (
            f"Conversation (index, speaker, text):\n{convo}\n\n"
            f"Emotion utterance: index {i} (\"{lines[i]}\"), emotion = {emo}.\n"
            f"Candidate cause utterance: index {j} (\"{lines[j]}\").\n\n"
            f"{_FEWSHOT_CROSS}"
            "Think step by step:\n"
            "1. What emotion does utterance i express, and what triggered it?\n"
            "2. If utterance j had NOT been said (or were neutral), would i's "
            "emotion still arise?\n"
            "3. Do i and j merely share a topic, or is there a causal link?\n\n"
            f"{_RULES_CROSS}"
            "Then score each dimension as a float in [0,1]:\n"
            "- s_nec (necessity): imagine utterance j had NOT occurred (or were "
            "neutral/irrelevant). How likely would the emotion in i then DISAPPEAR "
            "or clearly weaken? 1 = j is necessary for i's emotion.\n"
            "- s_suf (sufficiency): imagine ONLY j is given as context. How likely is "
            "j ALONE enough to elicit the emotion in i? 1 = j alone suffices.\n"
            "- s_dir (direction): 1 = j causes i (the cause precedes / produces the "
            "emotion), 0 = the reverse (i causes j, or j is a reaction to i).\n"
            "- s_spur (spuriousness): probability that i and j merely CO-OCCUR or "
            "share a topic WITHOUT a causal link. 1 = likely spurious (not causal).\n\n"
            'End your response with a JSON object: {"s_nec": <f>, "s_suf": <f>, "s_dir": <f>, "s_spur": <f>}'
        )
    return sys, user


# Match a single *flat* JSON object (no nested braces). The score object is
# always flat, so this avoids the greedy ``\{.*\}`` failure mode under
# chain-of-thought: when the model reasons in free text before emitting the
# answer, the reasoning may itself contain braces (e.g. it restates the schema
# ``{"s_nec": ...}``), and a greedy first-brace-to-last-brace match would splice
# reasoning text onto the answer and fail to parse.
_JSON_RE = re.compile(r"\{[^{}]*\}")
_SCORE_KEYS = ("s_nec", "s_suf", "s_dir", "s_spur")


def parse_scores(text):
    """Extract the 4 counterfactual floats from the model output; None on failure.

    Robust to chain-of-thought: scans every flat ``{...}`` object and returns the
    **last** one that parses and carries all four score keys, since under CoT the
    final JSON answer comes after the reasoning (and the reasoning may contain
    other brace-delimited fragments).
    """
    matches = _JSON_RE.findall(text or "")
    if not matches:
        return None
    for frag in reversed(matches):          # CoT puts the answer last
        try:
            obj = json.loads(frag)
        except Exception:
            continue
        if not all(key in obj for key in _SCORE_KEYS):
            continue
        try:
            vals = [float(obj[k]) for k in _SCORE_KEYS]
        except Exception:
            continue
        return [min(1.0, max(0.0, v)) for v in vals]
    return None


def aggregate(samples, k_expected=None):
    """Average the parsed score-vectors and estimate self-consistency rho in [0,1].

    rho = 1 - mean coordinate-wise std across samples (clamped). High agreement
    across stochastic samples -> high reliability weight in the distillation loss.

    Two reliability safeguards (both feed the rho-weighted distillation loss, so
    getting rho right matters):
      * **No parses** (``samples`` empty): the pair could not be annotated at all
        (request error, or every sample failed to parse). We return a neutral
        target with ``rho = 0`` so the pair is effectively dropped from the loss,
        instead of masquerading as a perfectly-reliable 0.5 target.
      * **Partial parses**: when only some of the ``k_expected`` requested samples
        parsed, rho is scaled by the success ratio ``n/k_expected`` so a pair
        backed by few samples is down-weighted rather than trusted as if all
        samples agreed.
    """
    dim = 4
    n = len(samples)
    if n == 0:                              # nothing parsed -> unreliable
        return [0.5] * dim + [0.0]
    mean = [sum(s[c] for s in samples) / n for c in range(dim)]
    if n == 1:
        rho = 1.0
    else:
        var = [sum((s[c] - mean[c]) ** 2 for s in samples) / n for c in range(dim)]
        std = [v ** 0.5 for v in var]
        rho = 1.0 - (sum(std) / dim)
        rho = min(1.0, max(0.0, rho))
    if k_expected and k_expected > 0:       # down-weight when samples were lost
        rho *= min(1.0, n / float(k_expected))
    return mean + [rho]


# ---------------------------------------------------------------------------
# Annotators.
# ---------------------------------------------------------------------------
class StubAnnotator:
    """Deterministic offline stub (no LLM). Used by --dry-run to validate the
    pickle format / plumbing without Qwen or any heavy dependency."""

    def __init__(self, k=1):
        self.k = k

    def _det(self, sys, user, salt):
        h = hashlib.sha256((str(salt) + user).encode("utf-8")).digest()
        return [h[c] / 255.0 for c in range(4)]

    def annotate(self, sys, user):
        return [self._det(sys, user, s) for s in range(self.k)]


# JSON schema for guided decoding (only used when --guided-json is enabled).
# By default guided_json is OFF because CoT reasoning needs free-form text
# output; the model thinks in natural language then outputs the JSON at the end.
_SCORE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "s_nec":  {"type": "number"},
        "s_suf":  {"type": "number"},
        "s_dir":  {"type": "number"},
        "s_spur": {"type": "number"},
    },
    "required": ["s_nec", "s_suf", "s_dir", "s_spur"],
}


def build_chat_payload(model, sys, user, k, temperature, top_p, max_new_tokens,
                       guided_json=False):
    """Build an OpenAI-/vLLM-compatible /chat/completions request body.

    Uses ``n=k`` so the server generates k stochastic samples in a single
    request, sharing the prefix KV-cache across all k samples.  This is much
    faster than sending k separate n=1 requests because:
      - One HTTP round-trip instead of k
      - Prefix (system + conversation) is computed once, shared across k samples
      - vLLM generates the k samples in parallel internally

    When ``guided_json=True``, adds vLLM's ``guided_json`` field with a JSON
    schema that constrains the output to the 4-score structure.
    """
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
        "n": int(k),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
    }
    if guided_json:
        body["guided_json"] = _SCORE_JSON_SCHEMA
    return body


def parse_chat_response(obj):
    """Parse score-vectors from an OpenAI-/vLLM-compatible chat response dict.

    Returns a list of [s_nec,s_suf,s_dir,s_spur]; entries that fail to parse are
    dropped. Returns [] if nothing parses.
    """
    out = []
    for ch in obj.get("choices", []):
        msg = ch.get("message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        scores = parse_scores(content or "")
        if scores is not None:
            out.append(scores)
    return out


class VLLMAnnotator:
    """Annotator that talks to a vLLM (OpenAI-compatible) server over HTTP.

    Uses only the standard library (urllib) so no extra client dependency is
    required. Requests k samples per pair via the ``n`` parameter so the
    prefix KV-cache is shared across all k samples in one request.
    Supports concurrent batch annotation via ``batch_annotate`` which uses
    ThreadPoolExecutor to send multiple requests in parallel.
    """

    def __init__(self, base_url, model, k=5, max_new_tokens=256, temperature=0.3,
                 top_p=0.9, api_key=None, timeout=120, batch_size=32,
                 guided_json=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.k = k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout
        self.batch_size = batch_size
        self.guided_json = guided_json
        self._executor = ThreadPoolExecutor(max_workers=batch_size)

    def _single_request(self, sys, user):
        """Send one n=k HTTP request and return parsed score-vectors."""
        payload = build_chat_payload(self.model, sys, user, self.k,
                                     self.temperature, self.top_p,
                                     self.max_new_tokens,
                                     guided_json=self.guided_json)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            outs = parse_chat_response(obj)
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            outs = []
        # Return whatever parsed (possibly empty). ``aggregate`` maps an empty
        # list to rho=0 so a failed/garbage annotation is down-weighted rather
        # than stored as a perfectly-reliable neutral 0.5 target.
        return outs

    def annotate(self, sys, user):
        """Annotate a single pair."""
        return self._single_request(sys, user)

    def batch_annotate(self, prompts):
        """Annotate a batch of (sys, user) pairs concurrently.

        Returns a list of score-vector lists in the same order as input.
        Uses ThreadPoolExecutor to send ``batch_size`` requests in parallel,
        leveraging vLLM's continuous batching for much higher throughput.
        """
        results = [None] * len(prompts)
        futures = {
            self._executor.submit(self._single_request, sys, user): idx
            for idx, (sys, user) in enumerate(prompts)
        }
        n_fail = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                n_fail += 1
                results[idx] = []          # empty -> rho=0 in aggregate
        if n_fail:
            print(f"\n[warn] {n_fail}/{len(prompts)} batch requests failed")
        return results


class QwenAnnotator:
    """Local Qwen2.5-7B-Instruct annotator via transformers."""

    def __init__(self, model_path, k=5, max_new_tokens=256, temperature=0.7,
                 device=None, dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.k = k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            device_map=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model.eval()

    def annotate(self, sys, user):
        torch = self.torch
        messages = [{"role": "system", "content": sys},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outs = []
        with torch.no_grad():
            for _ in range(self.k):
                gen = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=self.temperature, top_p=0.9,
                )
                new = gen[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(new, skip_special_tokens=True)
                scores = parse_scores(decoded)
                if scores is not None:
                    outs.append(scores)
        # Return whatever parsed (possibly empty); ``aggregate`` maps an empty
        # list to rho=0 so failed parses are down-weighted, not trusted.
        return outs


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/dataset/train.txt",
                    help="raw dialogue txt (used for text + heuristic candidates)")
    ap.add_argument("--out", default="data/llm_anno.pkl", help="output pickle path")
    ap.add_argument("--qwen-path", default=None,
                    help="in-process transformers backend: local model dir / HF id")
    ap.add_argument("--vllm-url", default=None,
                    help="vLLM OpenAI-compatible base url, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="served model name for the vLLM backend")
    ap.add_argument("--api-key", default=None,
                    help="bearer token for the vLLM/OpenAI endpoint (default $OPENAI_API_KEY or EMPTY)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--window", type=int, default=12,
                    help="heuristic candidate window |i-j|")
    ap.add_argument("--k", type=int, default=5, help="LLM samples per pair (rho)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="max tokens for LLM response (needs room for reasoning + JSON, default 256)")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="sampling temperature (lower = more deterministic, default 0.3)")
    ap.add_argument("--limit", type=int, default=0,
                    help="annotate at most this many dialogues (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use deterministic stub instead of Qwen (no heavy deps)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="concurrent requests to vLLM (0 = sequential, default 32)")
    ap.add_argument("--guided-json", action="store_true",
                    help="enable vLLM guided JSON decoding (default: off; CoT needs free-form text)")
    ap.add_argument("--checkpoint-interval", type=int, default=100,
                    help="save checkpoint every N pairs (0 = only at end, default 100)")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not load existing output file; start from scratch")
    args = ap.parse_args()

    dialogues = read_dialogues(args.train)
    by_doc = {d["doc_id"]: d for d in dialogues}
    if args.limit:
        dialogues = dialogues[: args.limit]

    # Candidate (doc_id -> [(i,j)]).
    cand = {d["doc_id"]: heuristic_candidates(d, args.window) for d in dialogues}

    # Annotator: vLLM HTTP > in-process transformers > stub.
    use_batch = False
    if args.dry_run:
        annotator = StubAnnotator(k=args.k)
    elif args.vllm_url:
        batch_size = args.batch_size if args.batch_size > 0 else 1
        annotator = VLLMAnnotator(args.vllm_url, args.model, k=args.k,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_p=args.top_p,
                                  api_key=args.api_key, batch_size=batch_size,
                                  guided_json=args.guided_json)
        use_batch = batch_size > 1
    elif args.qwen_path:
        annotator = QwenAnnotator(args.qwen_path, k=args.k,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
    else:
        print("[warn] no --vllm-url or --qwen-path given; using deterministic "
              "stub (equivalent to --dry-run).")
        annotator = StubAnnotator(k=args.k)

    def _fmt_time(seconds):
        """Format seconds into human-readable string (e.g. 1h23m45s)."""
        s = int(max(0, seconds))
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _progress_bar(frac, width=30):
        """Return a visual progress bar string."""
        filled = int(width * frac)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def _print_progress(done, total, t0, last_print):
        """Print a single-line progress update with ETA.

        Returns the timestamp of this call so the caller can throttle refreshes.
        """
        now = time.time()
        # Throttle: refresh at most every 0.5s (unless this is the final item)
        if done < total and now - last_print < 0.5:
            return last_print
        elapsed = now - t0
        pct = done / max(1, total) * 100
        frac = done / max(1, total)
        bar = _progress_bar(frac)
        rate = done / max(1e-6, elapsed)
        if done > 0 and done < total:
            eta = elapsed / done * (total - done)
        elif done >= total:
            eta = 0
        else:
            eta = 0
        print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  "
              f"elapsed {_fmt_time(elapsed)}  "
              f"ETA {_fmt_time(eta)}  "
              f"{rate:.1f} pairs/s", end="", flush=True)
        return now

    # --- Resume from checkpoint ---
    table = OrderedDict()
    resumed = 0
    out_path = os.path.abspath(args.out)
    if not args.no_resume and os.path.isfile(out_path):
        try:
            with open(out_path, "rb") as f:
                existing = pkl.load(f)
            if isinstance(existing, dict):
                table = OrderedDict(existing)
                resumed = len(table)
                print(f"[resume] loaded {resumed} existing annotations from {out_path}")
        except Exception as e:
            print(f"[warn] could not load existing output for resume ({e}); starting fresh")
            table = OrderedDict()

    total = sum(len(v) for v in cand.values())
    skipped = 0
    done, t0 = 0, time.time()
    last_print = 0  # timestamp of last terminal refresh
    last_save = t0   # timestamp of last checkpoint save
    ckpt_interval = args.checkpoint_interval
    newly_done = 0   # pairs annotated in this run (not counting resumed)

    if resumed:
        print(f"Annotating {total} candidate pairs ({resumed} already done, "
              f"{total - resumed} remaining) ...")
    else:
        print(f"Annotating {total} candidate pairs ...")

    def _save_checkpoint():
        """Atomically save current table to the output pickle."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            pkl.dump(dict(table), f)
        os.replace(tmp, out_path)  # atomic on POSIX

    # --- Build the full list of pending pairs (preserving order) ---
    pending = []       # list of (key, sys, user)
    for doc_id, ijs in cand.items():
        dlg = by_doc[doc_id]
        n = len(dlg["lines"])
        for (i, j) in ijs:
            if not (0 <= i < n and 0 <= j < n):
                continue
            key = (int(doc_id), int(i), int(j))
            if key in table:          # already annotated in a previous run
                skipped += 1
                done += 1
                continue
            sys, user = build_prompt(dlg, i, j)
            pending.append((key, sys, user))

    # Fast-forward progress bar for resumed pairs
    if skipped:
        last_print = _print_progress(done, total, t0, last_print)

    remaining = len(pending)
    if resumed:
        print(f"  {skipped} pairs resumed, {remaining} pairs to annotate")
    elif remaining == 0:
        print("  All pairs already annotated!")
    elif use_batch:
        print(f"  Using batch mode (batch_size={args.batch_size}, "
              f"{remaining} pairs to annotate)")

    # --- Annotate pending pairs ---
    if use_batch and remaining > 0:
        # Concurrent batch mode: process pending in chunks of batch_size
        batch_size = args.batch_size
        for start in range(0, remaining, batch_size):
            chunk = pending[start:start + batch_size]
            keys = [c[0] for c in chunk]
            prompts = [(c[1], c[2]) for c in chunk]
            results = annotator.batch_annotate(prompts)
            for key, samples in zip(keys, results):
                table[key] = aggregate(samples, k_expected=args.k)
                done += 1
                newly_done += 1
            last_print = _print_progress(done, total, t0, last_print)
            # Periodic checkpoint
            if ckpt_interval > 0 and newly_done >= ckpt_interval and newly_done % ckpt_interval < len(chunk):
                _save_checkpoint()
                now = time.time()
                print(f"\n  [checkpoint] saved {len(table)} annotations "
                      f"({_fmt_time(now - t0)} elapsed)")
    elif remaining > 0:
        # Sequential mode (QwenAnnotator / StubAnnotator / batch_size=0)
        for key, sys, user in pending:
            samples = annotator.annotate(sys, user)
            table[key] = aggregate(samples, k_expected=args.k)
            done += 1
            newly_done += 1
            last_print = _print_progress(done, total, t0, last_print)
            # Periodic checkpoint
            if ckpt_interval > 0 and newly_done % ckpt_interval == 0:
                _save_checkpoint()
                now = time.time()
                print(f"\n  [checkpoint] saved {len(table)} annotations "
                      f"({_fmt_time(now - t0)} elapsed)")
                last_save = now

    # Final newline after the progress line
    elapsed = time.time() - t0
    print(f"\n  Finished {_fmt_time(elapsed)} total, "
          f"{done / max(1e-6, elapsed):.1f} pairs/s avg "
          f"({skipped} resumed, {newly_done} newly annotated)")

    _save_checkpoint()
    print(f"[done] wrote {len(table)} annotations -> {args.out}")
    print("Set `llm_anno_path: %s` in src/config.yaml to enable distillation." % args.out)


if __name__ == "__main__":
    main()
