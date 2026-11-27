<p align="center">
  <img src="docs/media/banner.svg" alt="Tlav. Old PCs. New minds." width="100%">
</p>

<p align="center">
  <b>Your old computer can run a private AI.</b><br>
  Tlav picks the best model it can handle, shrinks it to fit and starts a chat. Offline, with one command.
</p>

<p align="center">
  <a href="https://tlav.online"><img src="https://img.shields.io/badge/website-tlav.online-f26b3a?style=flat-square" alt="Website: tlav.online"></a>
  <a href="https://github.com/ArhaanDev24/Tlav/actions/workflows/ci.yml"><img src="https://github.com/ArhaanDev24/Tlav/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-264530?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-264530?style=flat-square" alt="macOS, Linux, Windows">
  <img src="https://img.shields.io/badge/runs-100%25%20offline-f26b3a?style=flat-square" alt="Runs 100% offline">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-264530?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#install"><img src="docs/media/nav/install.svg" alt="Install" height="40"></a>
  <a href="#see-it-work"><img src="docs/media/nav/see.svg" alt="See it work" height="40"></a>
  <a href="#chat"><img src="docs/media/nav/chat.svg" alt="Chat" height="40"></a>
  <a href="#how-it-picks"><img src="docs/media/nav/picks.svg" alt="How it picks" height="40"></a>
  <a href="#cluster"><img src="docs/media/nav/cluster.svg" alt="Cluster" height="40"></a>
  <a href="#commands"><img src="docs/media/nav/commands.svg" alt="Commands" height="40"></a>
  <a href="docs/GUIDE.md"><img src="docs/media/nav/guide.svg" alt="Full guide" height="40"></a>
</p>

<br>

<p align="center">
  <img src="docs/media/setup.gif" alt="A real tlav run: it measures the computer, picks Qwen3.5 9B, runs a speed test and asks where to chat" width="860">
  <br><sub>A real run on a 16 GB, 4-core PC. Only the one-minute speed test is sped up.</sub>
</p>

<table>
  <tr>
    <td width="33%" valign="top"><img src="docs/media/icons/detect.svg" width="40" alt=""><br><b>Knows your PC</b><br><sub>RAM, CPU, disk and real memory speed.</sub></td>
    <td width="33%" valign="top"><img src="docs/media/icons/find.svg" width="40" alt=""><br><b>Finds the best model</b><br><sub>50+ models, checked against the real files.</sub></td>
    <td width="33%" valign="top"><img src="docs/media/icons/shrink.svg" width="40" alt=""><br><b>Shrinks it to fit</b><br><sub>Near-lossless, and quality-checked.</sub></td>
  </tr>
  <tr>
    <td valign="top"><img src="docs/media/icons/speed.svg" width="40" alt=""><br><b>Makes it faster</b><br><sub>Same answers, more words per second.</sub></td>
    <td valign="top"><img src="docs/media/icons/chat.svg" width="40" alt=""><br><b>Opens a chat</b><br><sub>In your terminal or your browser.</sub></td>
    <td valign="top"><img src="docs/media/icons/cluster.svg" width="40" alt=""><br><b>Pools old PCs</b><br><sub>Several computers, one bigger model.</sub></td>
  </tr>
</table>

## Install

**macOS / Linux** (in Terminal)
```bash
curl -fsSL https://tlav.online/install.sh | bash
```

**Windows** (in PowerShell)
```powershell
irm https://tlav.online/install.ps1 | iex
```

Then just type `tlav`. You only need Python 3.8+, and on Windows the installer offers to install it for you.

## See it work

<table>
  <tr>
    <td width="50%" valign="top"><b>Terminal chat</b><br><br><img src="docs/media/chat.gif" alt="Tlav terminal chat answering at 3.4 tokens per second" width="100%"></td>
    <td width="50%" valign="top"><b>Web chat</b><br><br><img src="docs/media/web-chat.gif" alt="Tlav web chat streaming an answer" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="40%" align="center"><img src="docs/media/web-light.png" alt="Web chat, white theme" width="100%"><br><sub>White</sub></td>
    <td width="40%" align="center"><img src="docs/media/web-dark.png" alt="Web chat, dark theme" width="100%"><br><sub>Dark</sub></td>
    <td width="20%" align="center"><img src="docs/media/web-phone.png" alt="Web chat on a phone" width="100%"><br><sub>Phone</sub></td>
  </tr>
</table>

<sub>All real recordings. MP4s: <a href="docs/media/setup.mp4">setup</a> · <a href="docs/media/chat.mp4">terminal</a> · <a href="docs/media/web-chat.mp4">web</a> · <a href="docs/media/cluster.mp4">cluster</a></sub>

## Chat

Type `tlav` and choose **terminal** or **browser**. Everything stays on your computer.

`/new` fresh chat · `/think on` smarter, slower · `/save` keep it · `/stats` speed and memory · `/web` open the browser

## How it picks

The strongest model that still runs smoothly on **your** machine:

| Your RAM | Tlav picks | Speed |
|---|---|---|
| 4 GB | Qwen3.5 2B | ~8 tok/s |
| 8 GB | Qwen3.5 4B | ~5 tok/s |
| 16 GB | Qwen3.5 9B | ~3 tok/s |
| **2 × 16 GB** | **Qwen3.6 35B, split across both** | ~7 tok/s |

<sub>Typical picks for a 2013 iMac. Try yours: <code>tlav recommend</code></sub>

## Cluster

Run `tlav worker` on spare computers, then `tlav setup --cluster` on the main one. Their memory adds up, so you can run a bigger, smarter model.

<p align="center">
  <img src="docs/media/cluster.gif" alt="Tlav connecting two workers and splitting a model across three computers" width="860">
</p>

<sub>Use it on a home network you trust. Wired Ethernet is best.</sub>

## Commands

| Command | What it does |
|---|---|
| `tlav` | Set up, or start chatting |
| `tlav recommend` | What this PC can run |
| `tlav chat` | Terminal chat |
| `tlav run` | Web chat + OpenAI-compatible API |
| `tlav worker` | Lend this PC to a cluster |
| `tlav stop` | Free the memory |

Everything else is in the **[full guide](docs/GUIDE.md)**.

<br>

<p align="center">
  <a href="docs/GUIDE.md"><img src="docs/media/nav/guide.svg" alt="Full guide" height="40"></a>
  <a href="CONTRIBUTING.md"><img src="docs/media/nav/contribute.svg" alt="Contribute" height="40"></a>
  <br><br>
  <sub><b>Old PCs. New minds.</b> · MIT licensed</sub>
</p>
