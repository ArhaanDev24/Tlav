"""`tlav remove` deletes models; `tlav uninstall` removes Tlav but nothing that isn't Tlav's."""
import json

import pytest

from tlav import remove


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "custom-home"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("TLAV_HOME", str(h))
    monkeypatch.setattr(remove.server, "status", lambda: {"running": False})
    return h


def _touch(p, n=10):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    return p


def test_model_bundles_its_helper_files(home):
    m = home / "models"
    _touch(m / "Qwen-f16.gguf", 100)
    _touch(m / "Qwen-f16.imatrix.gguf", 5)
    _touch(m / "Qwen-f16.logits", 7)
    _touch(m / "Qwen-Q4_K_M.gguf", 40)
    _touch(m / "Qwen-Q8_0.gguf.part", 3)
    files = remove.model_files()
    names = [f.name for f in files]
    assert names == ["Qwen-f16.gguf", "Qwen-Q4_K_M.gguf", "Qwen-Q8_0.gguf"]   # A-Z; helpers aren't listed alone
    f16 = files[0]
    assert f16.size == 112 and len(f16.extras) == 2
    assert files[2].unfinished


def test_match_by_number_name_or_part(home):
    for n in ("Qwen-Q4_K_M.gguf", "Qwen-Q8_0.gguf", "Gemma-3-1b-it-Q4_K_M.gguf"):
        _touch(home / "models" / n)
    files = remove.model_files()
    assert [f.name for f in remove.match(files, "2")] == [files[1].name]
    assert [f.name for f in remove.match(files, "gemma")] == ["Gemma-3-1b-it-Q4_K_M.gguf"]
    assert len(remove.match(files, "qwen")) == 2          # ambiguous: the CLI asks for more
    assert remove.match(files, "llama") == []


def test_deleting_the_active_model_forgets_it(home):
    m = home / "models"
    act = _touch(m / "A.gguf")
    _touch(m / "B.gguf")
    (home / "state.json").write_text(json.dumps({"active": {"path": str(act), "draft_path": None}}))
    files = remove.model_files()
    remove.delete_models([f for f in files if f.name == "A.gguf"])
    assert not act.exists() and (m / "B.gguf").exists()
    assert json.loads((home / "state.json").read_text())["active"] is None


def test_deleting_the_draft_keeps_the_main_model(home):
    m = home / "models"
    main, draft = _touch(m / "Big.gguf"), _touch(m / "Small.gguf")
    (home / "state.json").write_text(json.dumps({"active": {"path": str(main), "draft_path": str(draft)}}))
    remove.delete_models([f for f in remove.model_files() if f.name == "Small.gguf"])
    act = json.loads((home / "state.json").read_text())["active"]
    assert act["path"] == str(main) and act["draft_path"] is None


def test_uninstall_in_a_custom_folder_only_deletes_tlavs_files(home, monkeypatch):
    monkeypatch.setattr(remove, "_remove_services", lambda: [])
    for p in ("models/A.gguf", "runtime/bin/llama-server", "chats/last.json", "web/index.html"):
        _touch(home / p)
    (home / "state.json").write_text("{}")
    mine = _touch(home / "my-notes.txt")              # not Tlav's: must survive
    remove.uninstall()
    assert mine.exists()
    assert sorted(p.name for p in home.iterdir()) == ["my-notes.txt"]


def test_uninstall_can_keep_models(home, monkeypatch):
    monkeypatch.setattr(remove, "_remove_services", lambda: [])
    _touch(home / "models" / "A.gguf")
    _touch(home / "runtime" / "bin" / "llama-server")
    notes = remove.uninstall(keep_models=True)
    assert (home / "models" / "A.gguf").exists() and not (home / "runtime").exists()
    assert any("kept" in n for n in notes)


def test_size_labels():
    assert remove.gb(10) == "under 1 MB"
    assert remove.gb(50 * 1024 ** 2) == "50 MB"
    assert remove.gb(int(1.07 * 1024 ** 3)) == "1.07 GB"
