"""Reproduce E023's missing path cheaply, without modifying frozen E023 artifacts."""
import argparse
from pathlib import Path

from .gate import create, read, sha
from .reachability import SEMANTICS, analyze


def circuit(length, positions, heads, mlps):
    return dict(semantics=SEMANTICS, layers=28, heads_per_layer=16,
                sequence_length=length, dynamic_heads=heads, dynamic_mlps=mlps,
                source=dict(layer=0, phase="attention_output", positions=positions),
                targets=[dict(id="MLP17_last", layer=17, phase="mlp_output", positions=[length-1])])


def run(root):
    import numpy as np
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from e023.subject import Subject

    root = Path(root)
    spec = read(root/"runs/E023/simulator_spec.json")["circuit"]
    docs = [d for d in read(root/"runs/E022/dataset.json")["documents"] if d["pair_id"] >= 16]
    structural = [dict(doc_id=d["id"], **analyze(circuit(len(d["ids"]), d["positions"]["subject"],
                                                       spec["heads"], spec["mlps"]))) for d in docs]
    if len(structural) != 128 or any(r["ok"] for r in structural):
        raise AssertionError("Historical unreachable-state regression changed")
    torch.set_num_threads(4)
    torch.manual_seed(23023)
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=1536,
                     num_hidden_layers=28, num_attention_heads=16,
                     num_key_value_heads=8, head_dim=128, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).float().eval()

    class Counter:
        calls = 0

        def reserve(self, kind, label):
            if kind != "forward":
                raise AssertionError("Only synthetic forwards allowed")
            self.calls += 1
            return self.calls

        def finish(self, *args):
            pass

    budget = Counter()
    subject = Subject(budget, {" is": 1, " are": 2}, model=model)
    doc = dict(ids=[3, 4, 5, 6], positions={"subject": [1, 2], "all": [0, 1, 2, 3]})
    try:
        base, natural = subject.run(doc, "regression_R", capture="all", capture_kv=(19,), full=True)
        _, donor = subject.run(dict(doc, ids=[3, 7, 8, 6]), "regression_D", capture="source")
        blocked, states = subject.run(doc, "regression_late_heads", donor=donor,
                                      reference=natural, circuit=spec, capture="all")
        # Nonzero source confirms that this is not a vacuous same-value test.
        source_change = float(np.linalg.norm(states["c_0"]-natural["c_0"]))
        blocked_change = float(np.linalg.norm(np.array(blocked["mlp17_last"])-base["mlp17_last"]))
        connected_spec = dict(heads=[[1, 0], *spec["heads"]], mlps=spec["mlps"])
        connected, _ = subject.run(doc, "regression_early_head", donor=donor,
                                   reference=natural, circuit=connected_spec)
        connected_change = float(np.linalg.norm(np.array(connected["mlp17_last"])-base["mlp17_last"]))
        if not source_change > 0 or blocked_change != 0 or not connected_change > 1e-9:
            raise AssertionError("Real hook reachability positive/negative controls failed")

        # An intervention with a known nonzero V delta must actually replay.
        # Other head/position outputs at the receiver must remain unchanged.
        changed = {k: v.copy() for k, v in natural.items()}
        changed["V_19"][:, 12, [1, 2], :] += np.linspace(-.1, .2, 128, dtype=np.float32)
        _, replay = subject.run(doc, "regression_V_replay", capture="all",
                                path=dict(layer=19, head=12, which="V", states=changed))
        delta = replay["c_19"].reshape(1, 4, 16, 128)-natural["c_19"].reshape(1, 4, 16, 128)
        receiver_change = float(np.linalg.norm(delta[:, -1, 12]))
        delta[:, -1, 12] = 0
        if not receiver_change > 1e-9 or np.any(delta != 0):
            raise AssertionError("Isolated V replay or replacement coordinates failed")
        off_source = {k: v.copy() for k, v in natural.items()}
        off_source["V_19"][:, 12, 0, :] += 10
        negative, _ = subject.run(doc, "regression_off_source", full=True,
                                  path=dict(layer=19, head=12, which="V", states=off_source))
        np.testing.assert_allclose(negative["logits"], base["logits"], atol=1e-6, rtol=0)
    finally:
        subject.close()
    protected = {}
    for freeze in sorted((root/"runs/E023").glob("freeze*.json")):
        protected[freeze.relative_to(root).as_posix()] = sha(freeze)
        for name, expected in read(freeze).get("files", {}).items():
            actual = sha(root/name)
            if actual != expected:
                raise AssertionError("Frozen historical file changed: " + name)
            protected[name] = actual
    return dict(ok=True, historical_unreachable_checkpoints=len(structural),
                structural=structural, synthetic_forwards=budget.calls,
                pretrained_forwards=0, backward_calls=0, api_cost_usd=0,
                source_delta_norm=source_change, unreachable_MLP17_delta_norm=blocked_change,
                positive_control_MLP17_delta_norm=connected_change,
                isolated_V_receiver_delta_norm=receiver_change,
                checks=dict(source_nonzero=True, unreachable_checkpoint_zero=True,
                            reachable_positive_control_nonzero=True,
                            isolated_V_nonzero_and_coordinates_exact=True,
                            outside_source_negative_control=True, historical_freezes_intact=True),
                protected_files=protected,
                limits="Synthetic learned weights are random. Adding L1H0 is a software positive control, "
                       "not a proposed scientific circuit or a repair validated on pretrained Qwen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(Path.cwd())
    create(args.out, result)
    print({k: v for k, v in result.items() if k not in {"protected_files", "structural"}})
