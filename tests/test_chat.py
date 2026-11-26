from tlav import chat


def test_trim_keeps_system_and_newest():
    msgs = [{"role": "system", "content": "be brief"}]
    for i in range(40):
        msgs += [{"role": "user", "content": "q" * 700}, {"role": "assistant", "content": "a" * 700}]
    msgs.append({"role": "user", "content": "latest question"})
    out, dropped = chat.trim_history(msgs, ctx=2048, reserve=512)
    assert out[0]["role"] == "system" and out[-1]["content"] == "latest question"
    assert dropped > 0 and chat.estimate_tokens(out) <= 2048 - 512
    assert out[1]["role"] == "user"  # pairs are dropped together


def test_trim_noop_when_it_fits():
    msgs = [{"role": "user", "content": "hi"}]
    assert chat.trim_history(msgs, 4096, 1024) == (msgs, 0)


def test_think_splitter_across_chunks():
    s = chat.ThinkSplitter()
    parts = []
    for piece in ["<th", "ink>let me ", "think</thi", "nk>The answer", " is 4."]:
        parts += s.feed(piece)
    think = "".join(t for k, t in parts if k == "think")
    answer = "".join(t for k, t in parts if k == "answer")
    assert think == "let me think" and answer == "The answer is 4."


def test_parse_command():
    assert chat.parse_command("/temp 0.3") == ("temp", "0.3")
    assert chat.parse_command("/EXIT") == ("exit", "")
    assert chat.parse_command("hello") == ("", "hello")
    assert chat.parse_command("//not a command") == ("", "/not a command")


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    c = chat.Chat("http://x", {"model": "m", "ctx": 4096})
    c.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    c.save("my chat/../x")
    d = chat.Chat("http://x", {"model": "m", "ctx": 4096})
    assert d.load("my chat/../x") and d.messages == c.messages
    assert all(p.parent == tmp_path / "chats" for p in (tmp_path / "chats").iterdir())


def test_think_default_follows_speed():
    assert chat.Chat("u", {"est_tps": 2.7}).think is False
    assert chat.Chat("u", {"est_tps": 20}).think is True


def test_mark_cursor_circles_and_cleans_up(capsys):
    import time
    m = chat.Mark(enabled=True, interval=0.01)
    m.start("  ")
    m.write("Hello")
    time.sleep(0.06)
    m.stop()
    m.stop()  # second stop does nothing
    out = capsys.readouterr().out
    assert "Hello" in out and out.endswith("\b \b")
    assert sum(out.count(f) for f in chat.Mark.FRAMES) >= 4   # it moved
    assert out.count("\b \b") == 2                           # one before the text, one at the end


def test_mark_off_writes_plain_text(capsys):
    m = chat.Mark(enabled=False)
    m.start("  ")
    m.write("Hi")
    m.stop()
    assert capsys.readouterr().out == "  Hi"
