"""Start Tlav automatically at login, turning an old PC into an always-on AI box.

macOS: a launchd agent in ~/Library/LaunchAgents
Linux: a systemd --user unit (+ `loginctl enable-linger` hint to run without login)
Windows: a Task Scheduler task that runs at logon
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

LABEL = "com.tlav.{role}"


def _command(role: str, host: str, port: int) -> List[str]:
    base = [sys.executable, "-m", "tlav"]
    if role == "worker":
        return base + ["worker"]
    return base + ["run", "--no-browser", "--foreground", "--host", host, "--port", str(port)]


def _paths(role: str) -> dict:
    home = Path.home()
    return {
        "Darwin": home / "Library" / "LaunchAgents" / f"{LABEL.format(role=role)}.plist",
        "Linux": home / ".config" / "systemd" / "user" / f"tlav-{role}.service",
    }


def launchd_plist(role: str, cmd: List[str], log_dir: Path) -> str:
    args = "\n".join(f"    <string>{a}</string>" for a in cmd)
    env = "".join(f"    <key>{k}</key><string>{v}</string>\n" for k, v in os.environ.items()
                  if k in ("TLAV_HOME", "HF_TOKEN", "PATH"))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL.format(role=role)}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>EnvironmentVariables</key>
  <dict>
{env}  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log_dir / f'service-{role}.log'}</string>
  <key>StandardErrorPath</key><string>{log_dir / f'service-{role}.log'}</string>
</dict>
</plist>
"""


def systemd_unit(role: str, cmd: List[str]) -> str:
    env = "".join(f"Environment={k}={v}\n" for k, v in os.environ.items() if k in ("TLAV_HOME", "HF_TOKEN"))
    return f"""[Unit]
Description=Tlav local AI ({role})
After=network-online.target

[Service]
ExecStart={' '.join(cmd)}
{env}Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""


def install(role: str = "server", host: str = "127.0.0.1", port: int = 8080) -> str:
    from . import config
    cmd = _command(role, host, port)
    system = platform.system()
    if system == "Darwin":
        path = _paths(role)["Darwin"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(launchd_plist(role, cmd, config.home()))
        subprocess.run(["launchctl", "unload", str(path)], capture_output=True)
        try:
            subprocess.run(["launchctl", "load", "-w", str(path)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"macOS refused to load the login item ({(e.stderr or b'').decode()[:120].strip()}). "
                               f"The file is at {path}.") from None
        return f"Installed launchd agent {path}"
    if system == "Linux":
        path = _paths(role)["Linux"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(systemd_unit(role, cmd))
        try:
            for step in (["daemon-reload"], ["enable", "--now", path.name]):
                subprocess.run(["systemctl", "--user"] + step, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"Saved the service file ({path}), but this system has no user service manager running "
                "(common on WSL, containers and minimal installs), so it can't start at login here. "
                "Add `tlav " + ("worker" if role == "worker" else "run --detach") + "` to your startup "
                "programs instead, or just type `tlav` when you need it.") from None
        return (f"Installed systemd user service {path.name}. To keep it running when nobody is logged in: "
                f"sudo loginctl enable-linger {os.environ.get('USER', '$USER')}")
    if system == "Windows":
        task = f"Tlav-{role}"
        quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        try:
            subprocess.run(["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", task, "/TR", quoted], check=True,
                           capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Windows refused to create the startup task (try a terminal opened with "
                               "'Run as administrator').") from None
        return f"Installed scheduled task {task} (starts at logon)"
    raise RuntimeError(f"Unsupported OS: {system}")


def uninstall(role: str = "server") -> str:
    system = platform.system()
    if system == "Darwin":
        path = _paths(role)["Darwin"]
        subprocess.run(["launchctl", "unload", "-w", str(path)], capture_output=True)
        path.unlink(missing_ok=True)
        return f"Removed {path}"
    if system == "Linux":
        path = _paths(role)["Linux"]
        subprocess.run(["systemctl", "--user", "disable", "--now", path.name], capture_output=True)
        path.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        return f"Removed {path.name}"
    if system == "Windows":
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", f"Tlav-{role}"], capture_output=True)
        return f"Removed scheduled task Tlav-{role}"
    raise RuntimeError(f"Unsupported OS: {system}")
