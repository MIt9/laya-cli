import sys
import types

import pytest


def _make_fake_laya(calls=None):
    if calls is None:
        calls = []

    mod = types.ModuleType("laya")

    class FakeAgent:
        def __init__(self, *a, **kw):
            self.device = kw.get("device", "cpu")
            # minimal encoder mock for embed_fn_from_agent
            self.model = types.SimpleNamespace(
                encoder=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=8))
            )
            self.tok = None

        def predict(self, state, questions, **kwargs):
            calls.append((state, list(questions.keys())))
            answers = {}
            for qid, qdef in questions.items():
                t = qdef["type"]
                if t == "choice":
                    crit = qdef.get("criteria", {})
                    if isinstance(crit, dict):
                        keys = list(crit.keys())
                    else:
                        keys = list(crit)
                    # deterministic: first key wins, 0.8 prob
                    answers[qid] = {
                        "type": "choice",
                        "choice": keys[0] if keys else "opt",
                        "probabilities": {
                            k: (0.8 if i == 0 else 0.2 / max(1, len(keys) - 1)) for i, k in enumerate(keys)
                        },
                        "confidence": 0.9,
                        "action": {"act_probability": 0.5},
                    }
                elif t == "score":
                    answers[qid] = {
                        "type": "score",
                        "score": 1.5,
                        "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
                        "confidence": 0.6,
                        "action": {"act_probability": 0.5},
                    }
                else:  # noul
                    # vary by state content for deterministic testing
                    s = str(state)
                    p = 0.85 if "forest" in s or "good" in s else 0.25
                    answers[qid] = {
                        "type": "noul",
                        "noul": p,
                        "confidence": max(p, 1 - p),
                        "action": {"act_probability": 0.5},
                    }
            return {
                "answers": answers,
                "usage": {"input_tokens": 10},
                "routing": {"model": "english", "reason": "test"},
            }

    def load(repo, device=None, subfolder=None, token=None):
        return FakeAgent(device=device)

    class FakeRouter(FakeAgent):
        def __init__(self, preload=True, device=None, **kw):
            super().__init__(device=device)
            self.device = device or "cpu"

    def fake_predict_shortlist(agent, state, questions, embed_fn, k, **kw):
        res = agent.predict(state, questions, **kw)
        # simulate shortlist filtering to k for choice questions
        res["shortlist"] = {
            qid: {
                "labels": list(qdef.get("criteria", {}).keys())[:k]
                if isinstance(qdef.get("criteria"), dict)
                else list(qdef.get("criteria", []))[:k],
                "scores": [0.9] * k,
                "k": k,
                "n": 30,
                "passthrough": False,
            }
            for qid, qdef in questions.items()
            if qdef.get("type") == "choice"
        }
        return res

    def fake_embed_fn(agent, max_length=512, batch_size=32):
        import numpy as np

        def fn(texts):
            return np.zeros((len(texts), 8), dtype=np.float32)

        return fn

    mod.load = load
    mod.Router = FakeRouter
    mod.predict_shortlist = fake_predict_shortlist
    mod.embed_fn_from_agent = fake_embed_fn
    mod.triage_questions = lambda: {
        "intent": {
            "type": "choice",
            "instructions": "What?",
            "criteria": {"refund": "money", "other": "rest"},
        },
        "is_urgent": {"type": "noul", "instructions": "urgent?"},
    }
    mod.email_questions = lambda: {
        "category": {
            "type": "choice",
            "instructions": "which?",
            "criteria": {"billing": "x", "other": "y"},
        }
    }
    mod.guard_questions = lambda: {"jailbreak": {"type": "noul", "instructions": "jail?"}}
    mod.moderation_questions = lambda: {"toxic": {"type": "noul", "instructions": "toxic?"}}
    mod.router_questions = lambda: {
        "difficulty": {"type": "score", "instructions": "hard?", "criteria": ["easy", "hard"]}
    }
    mod.__version__ = "0.3.4"
    return mod, calls


@pytest.fixture
def fake_laya():
    calls = []
    mod, calls = _make_fake_laya(calls)
    orig = sys.modules.get("laya")
    sys.modules["laya"] = mod
    try:
        yield mod, calls
    finally:
        if orig is not None:
            sys.modules["laya"] = orig
        else:
            sys.modules.pop("laya", None)


@pytest.fixture
def fake_laya_calls(fake_laya):
    _, calls = fake_laya
    return calls
