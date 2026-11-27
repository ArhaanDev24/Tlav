"""Regression tests for problems found by the stress test."""
import errno
import urllib.error

import pytest

from tlav import chat, cli, discover, hub, server


def test_shorten_keeps_start_and_end():
    text = "START " + "x" * 50000 + " END"
    out = chat.shorten(text, 1000)
    assert out.startswith("START") and out.endswith("END") and "left out" in out
    assert len(out) < 3200
    assert chat.shorten("short", 1000) == "short"


def test_cut_answer_is_detected():
    a = chat.Answer()
    assert a.cut                      # nothing finished it
    a.finished = True
    assert not a.cut
    b = chat.Answer()
    b.stopped = True                  # the user pressed Ctrl+C: not a cut-off
    assert not b.cut


def test_why_failed_explains(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("0.1 E srv load_model: failed to load model, 'x.gguf'\n")
    assert "damaged or incomplete" in server.why_failed(log)
    log.write_text("E llama_model_load: error loading model: unknown model architecture: 'newarch'\n")
    assert "install-runtime" in server.why_failed(log)
    log.write_text("0.00.004.494 E srv  something odd happened\n")
    assert "something odd happened" in server.why_failed(log)


def test_friendly_errors():
    assert "internet" in cli.friendly_error(urllib.error.URLError("refused"))
    assert "misspelled" in cli.friendly_error(urllib.error.HTTPError("u", 401, "x", {}, None))
    assert "disk is full" in cli.friendly_error(OSError(errno.ENOSPC, "No space left"))
    assert cli.friendly_error(OSError(errno.ENOSPC, "Not enough disk space for x")) == "Not enough disk space for x"


def test_discovery_offline_keeps_old_list(tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    (tmp_path / "discovered.json").write_text('{"updated": 0, "models": [{"id": "keep-me"}]}')

    def offline(url):
        raise urllib.error.URLError("no network")

    with pytest.raises(ConnectionError):          # no publisher reachable: an error, not an empty list
        discover.list_candidates(lister=offline)

    def unreachable():
        raise ConnectionError("Hugging Face can't be reached right now")

    monkeypatch.setattr(discover, "list_candidates", unreachable)
    discover.refresh_if_stale(progress=lambda *a: None)
    assert "keep-me" in (tmp_path / "discovered.json").read_text()


def test_damaged_download_is_fetched_again(tmp_path, monkeypatch):
    bad = tmp_path / "m.gguf"
    bad.write_text("<html>error page</html>")
    calls = []
    monkeypatch.setattr(hub, "_download_once", lambda repo, fn, dest, quiet: calls.append(fn) or dest)
    hub.download("r", "m.gguf", dest_dir=tmp_path, quiet=True)
    assert calls == ["m.gguf"] and not bad.exists()


def test_port_validation():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["worker", "--port", "99999"])
