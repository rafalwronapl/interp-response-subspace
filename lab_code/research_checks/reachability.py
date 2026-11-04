"""Necessary structural reachability for a deliberately narrow decoder contract.

Dense causal attention; tokenwise norms/MLPs; additive residuals; no KV cache.
Listed heads/MLPs are dynamic at ALL positions; all other branch outputs are
clamped to the recipient's natural values. This is a possible-dependency check,
not a claim about nonzero learned weights, effect size or mechanistic fidelity.
Other hook semantics need another adapter, not a relabelled contract.
"""

PHASES = ("residual_in", "attention_output", "mlp_input", "mlp_output", "residual_out")
SEMANTICS = "causal_decoder_output_clamping_v1"


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def analyze(spec):
    expected = {"semantics", "layers", "heads_per_layer", "sequence_length",
                "dynamic_heads", "dynamic_mlps", "source", "targets"}
    if set(spec) != expected or spec["semantics"] != SEMANTICS:
        raise ValueError("Unsupported or incomplete circuit semantics")
    layers = integer(spec["layers"], "layers", 1)
    heads = integer(spec["heads_per_layer"], "heads_per_layer", 1)
    length = integer(spec["sequence_length"], "sequence_length", 1)
    dynamic = set()
    for pair in spec["dynamic_heads"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each dynamic head must be [layer, head]")
        layer = integer(pair[0], "head layer", 0, layers-1)
        head = integer(pair[1], "head", 0, heads-1)
        if (layer, head) in dynamic:
            raise ValueError("Duplicate dynamic head")
        dynamic.add((layer, head))
    mlps = [integer(x, "MLP layer", 0, layers-1) for x in spec["dynamic_mlps"]]
    if len(set(mlps)) != len(mlps):
        raise ValueError("Duplicate dynamic MLP")

    def endpoint(item, source=False):
        keys = {"layer", "phase", "positions"} | (set() if source else {"id"})
        if set(item) != keys:
            raise ValueError("Invalid endpoint fields")
        integer(item["layer"], "endpoint layer", 0, layers-1)
        allowed = ("residual_in", "attention_output", "mlp_output") if source else PHASES
        if item["phase"] not in allowed:
            raise ValueError("Unsupported endpoint phase")
        if not isinstance(item["positions"], list) or not item["positions"]:
            raise ValueError("Endpoint must have concrete nonempty positions")
        for pos in item["positions"]:
            integer(pos, "position", 0, length-1)
        if len(set(item["positions"])) != len(item["positions"]):
            raise ValueError("Duplicate position")
        if not source and (not isinstance(item["id"], str) or not item["id"].strip()):
            raise ValueError("Target needs an id")

    source = spec["source"]
    endpoint(source, True)
    if not spec["targets"]:
        raise ValueError("No required checkpoints")
    for target in spec["targets"]:
        endpoint(target)
    if len({x["id"] for x in spec["targets"]}) != len(spec["targets"]):
        raise ValueError("Duplicate checkpoint id")

    states = {}
    residual = set()
    active_layers = {l for l, _ in dynamic}
    for layer in range(layers):
        def inject(phase, support):
            support = set(support)
            if source["layer"] == layer and source["phase"] == phase:
                support.update(source["positions"])
            states[(layer, phase)] = support
            return support

        residual = inject("residual_in", residual)
        # Attention reads the pre-attention residual, never its own output.
        attention = set(range(min(residual), length)) if residual and layer in active_layers else set()
        attention = inject("attention_output", attention)
        after_attention = residual | attention
        states[(layer, "mlp_input")] = after_attention
        mlp = inject("mlp_output", after_attention if layer in mlps else set())
        residual = after_attention | mlp
        states[(layer, "residual_out")] = residual

    checks = []
    for target in spec["targets"]:
        possible = states[(target["layer"], target["phase"])]
        missing = sorted(set(target["positions"])-possible)
        checks.append(dict(id=target["id"], reachable=not missing,
                           unreachable_positions=missing,
                           possible_changed_positions=sorted(possible)))
    return dict(ok=all(c["reachable"] for c in checks), targets=checks,
                scope="Necessary structural condition only; numerical sensitivity still required.")
