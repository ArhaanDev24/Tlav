from pathlib import Path

from tlav import service


def test_systemd_unit_runs_tlav():
    unit = service.systemd_unit("server", service._command("server", "0.0.0.0", 8080))
    assert "ExecStart=" in unit and "-m tlav run" in unit and "--host 0.0.0.0" in unit
    assert "Restart=on-failure" in unit


def test_launchd_plist_for_worker():
    plist = service.launchd_plist("worker", service._command("worker", "", 0), Path("/tmp/x"))
    assert "<string>worker</string>" in plist and "com.tlav.worker" in plist
    assert "<key>KeepAlive</key><true/>" in plist
