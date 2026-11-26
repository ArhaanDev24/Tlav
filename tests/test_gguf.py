from tlav import gguf


def _write(tmp_path, moe=False):
    path = tmp_path / "m.gguf"
    meta = {"general.name": "Tiny", "llama.block_count": 2, "llama.context_length": 4096,
            "llama.embedding_length": 64, "llama.attention.head_count": 4,
            "llama.attention.head_count_kv": 2}
    tensors = [("token_embd.weight", [64, 100], 0), ("blk.0.attn_q.weight", [64, 64], 0),
               ("blk.1.attn_q.weight", [64, 64], 0)]
    if moe:
        meta.update({"llama.expert_count": 8, "llama.expert_used_count": 2})
        tensors.append(("blk.0.ffn_up_exps.weight", [64, 32, 8], 0))
    gguf.write_minimal(str(path), "llama", meta, tensors)
    return path


def test_reads_architecture(tmp_path):
    info = gguf.read(str(_write(tmp_path)))
    assert info.arch == "llama" and info.name == "Tiny"
    assert info.layers == 2 and info.kv_heads == 2 and info.head_dim == 16
    assert info.context_length == 4096
    assert info.n_params == 64 * 100 + 2 * 64 * 64
    m = info.to_model("tiny")
    assert m.layers == 2 and not m.is_moe


def test_moe_active_params(tmp_path):
    info = gguf.read(str(_write(tmp_path, moe=True)))
    experts = 64 * 32 * 8
    assert info.n_expert_params == experts
    assert info.active_params == info.n_params - experts + experts * 2 // 8
    assert info.to_model("tiny").is_moe


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 64)
    try:
        gguf.read(str(p))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
