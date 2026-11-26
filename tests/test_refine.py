from tlav import catalog, finder, hardware, planner


def _nodes(ram=16):
    return [planner.Node(hardware.simulated(ram), 16.0)]


def test_refine_uses_real_sizes_and_drops_missing():
    nodes = _nodes(8)
    plans = planner.shortlist(planner.candidates(nodes[0].hw, 16.0), 3)
    big = {"Q4_K_M": ("x-Q4_K_M.gguf", 40 * 1024 ** 3)}   # real file far bigger than estimated
    real = {q.name: (f"x-{q.name}.gguf", int(plans[1].model.weights_gb(q) * 1024 ** 3))
            for q in catalog.QUANTS[1:5]}

    def lister(repo):
        if repo == plans[0].model.repo:
            return big
        if repo == plans[1].model.repo:
            return real
        raise OSError("offline")

    out = finder.refine(plans, nodes, lister=lister)
    assert out[0].plan is None and "too big" in out[0].note
    assert out[1].plan is not None and out[1].plan.model.id == plans[1].model.id
    assert out[2].plan is None and out[2].note.startswith("could not check")


def test_named_quants_pass_floor_even_when_files_are_small():
    # hybrid models can ship a Q4_K_M below 4.55 bits/weight: still a Q4_K_M
    q = catalog.quant_from_file(int(4.4 * 8.95e9 / 8), 8.95, "Q4_K_M")
    assert q in planner._quants_for("Q4_K_M", [q])


def test_shortlist_unique_and_ranked():
    nodes = _nodes(16)
    plans = planner.candidates(nodes[0].hw, 16.0)
    s = planner.shortlist(plans + plans, 5)
    assert len({p.model.id for p in s}) == len(s) == 5
    assert planner.rank(s)[0].quality >= s[-1].quality
