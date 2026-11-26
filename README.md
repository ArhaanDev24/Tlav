# Tlav

**Turn old PCs and laptops into local AI workstations, alone or together.**

That 2013 iMac in the corner, the ThinkPad in a drawer, the tower under the desk can all run a capable AI assistant, fully offline. And if one isn't enough, Tlav can **combine several old computers** to run a model none of them could hold alone.

```bash
tlav
```

> **Old PCs. New minds.** · Your old computer isn't dead. It's waiting. · Don't throw it out. Wake it up. · Two old laptops. One bigger brain. · Private AI from the computer you already own.

Website: [`site/index.html`](site/index.html)

One command. Tlav:

1. **Detects** your hardware: RAM, CPU cores, AVX/AVX2/AVX-512, GPU, disk.
2. **Benchmarks** memory bandwidth, which decides how fast a model can talk on a CPU, and self-calibrates against real runs.
3. **Finds the best model** for *this* PC: it ranks the catalog by your specs, then checks the top five against the files that really exist on Hugging Face (exact sizes, no weights downloaded) and tells you why the winner won. Any GGUF on Hugging Face works too (`tlav search`).
4. **Makes it use less RAM** with measured, near-lossless compression and a compressed context cache, then **verifies the quality**.
5. **Speeds it up** with speculative decoding, which gives exactly the same answers, faster.
6. **Measures it**: one quick speed test on the installed model; if your PC is slower than expected, Tlav offers a faster pick.
7. **Opens the Tlav chat** in your terminal. A web chat and an OpenAI-compatible API run too, and it can start at boot as an always-on AI box.

Next time, just type `tlav`: it starts your model and opens the chat.

---

## Install

Requires Python 3.8+ and nothing else. On first setup Tlav builds [llama.cpp](https://github.com/ggml-org/llama.cpp), tuned for your exact CPU.

**macOS / Linux**
```bash
git clone https://github.com/ArhaanDev24/Tlavv && cd Tlavv && ./install.sh
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/ArhaanDev24/Tlavv; cd Tlavv; ./install.ps1
```

### Intel Macs (2010–2020), including the 2013 iMac
Tlav builds llama.cpp from source with AVX2 on and Metal off (old Mac GPUs are slower than the CPU for this). You need Apple's command line tools once:
```bash
xcode-select --install
```

## What will my machine run?

```bash
tlav doctor                   # what Tlav sees
tlav recommend                # ranked models for this machine
tlav recommend --ram 16       # simulate a 16 GB machine
tlav recommend --ram 16 --ram 16   # simulate two 16 GB machines working together
```

Typical picks for a 2013 iMac (quad-core Haswell, DDR3-1600):

| Setup | Best pick | RAM used | Speed |
|---|---|---|---|
| 2 GB (light Linux) | Qwen3.5 0.8B @ Q4_K_M | ~0.8 GB | fast |
| 4 GB | Qwen3.5 2B @ Q8_0 | ~2.3 GB | ~8 tok/s |
| 8 GB | Qwen3.5 4B @ Q5_K_M | ~3.8 GB | ~5 tok/s |
| 16 GB | Qwen3.5 9B @ Q4_K_M | ~6.7 GB | ~2.5–3 tok/s |
| **two 16 GB PCs** | **Qwen3.6 35B-A3B (MoE) @ Q4_K_M, split across both** | ~11 GB each | ~7 tok/s on Ethernet |

`tlav setup --min-tps 5` prefers faster models, and `tlav setup --task code` picks coding models.

---

## The Tlav chat

`tlav` (or `tlav chat`) opens a chat in your terminal and starts your model if it isn't running. Answers stream as they are written, with the speed shown after each one. Conversations are kept on your disk only.

```
  ■ Tlav  Qwen3.5 9B  Q4_K_M
    this PC · 8192 tokens of memory · offline · reasoning off
    Type a message. /help for commands, Ctrl+C stops an answer.

› Explain RAM like I'm ten
  RAM is your computer's desk...
  64 tokens · 2.8 tokens/s · 23.1s
```

| Command | |
|---|---|
| `/new` | fresh conversation (the previous one is kept as `last`) |
| `/system TEXT` | how the assistant should behave |
| `/think on\|off` | let reasoning models think first: smarter, slower (on by default only on fast PCs) |
| `/temp X` | 0 = precise, 1 = creative |
| `/save NAME`, `/load NAME`, `/list` | keep and reopen conversations |
| `/copy` | copy the last answer |
| `/stats` | speed, memory, context used |
| `/web` | open the browser chat |
| `/stop` | stop the model and free its memory |
| `"""` | start/end a message with several lines |

Old turns are left out automatically when a long chat no longer fits the model's memory. `tlav chat --model file.gguf` chats with any file; `tlav chat --url http://host:8080` with a Tlav on another PC.

## Combine computers (cluster mode)

Yes, you can pool several old computers into one bigger AI machine. Tlav uses llama.cpp's RPC backend to **split the model's layers across machines**, so their RAM adds up.

```
 ┌──────────── main PC ────────────┐        ┌──── worker PC ────┐
 │ tlav run --cluster              │  LAN   │ tlav worker       │
 │ layers 1–24 + web chat + API    │ ◄────► │ layers 25–48      │
 └─────────────────────────────────┘        └───────────────────┘
```

**On every helper computer:**
```bash
tlav worker
```

**On the main computer:**
```bash
tlav cluster scan            # finds the workers automatically
tlav recommend --cluster     # what the combined machines can run
tlav setup --cluster         # pick, download, split and run it
```

If your network blocks discovery, add a worker by address: `tlav cluster add 192.168.1.23`.

**What to expect (honestly):**
- **It pools memory, not speed.** Each word passes through every machine in turn, so a split model runs at roughly the speed of its parts added together, plus network delay. Use a cluster to run a *bigger, smarter* model than any one machine could hold. Tlav never splits a model that fits on one computer.
- **Answers are identical** to running on one machine. We tested a 3-way split against a single PC and got word-for-word the same output.
- **Use Ethernet.** Gigabit Ethernet works well, Wi-Fi works but is slower. `tlav cluster list` shows the delay to each worker and rates the connection.
- **Only the main PC downloads the model.** Workers receive their share over the network and cache it, so the next start is faster.
- **Security:** a worker accepts work from anyone who can reach it, with no password. Only run it on a home network you trust, never on public Wi-Fi, and never forward its port (50052) on your router. `tlav worker --token WORD` stops accidental joins but is not real security.

---

## Find any model

```bash
tlav search qwen
tlav search "llama 3.2" --task code
tlav setup --hf bartowski/Qwen_Qwen3-8B-GGUF
```

Tlav searches Hugging Face, then reads each model's real architecture straight from its GGUF header. It downloads only a few MB per model, not the whole file, so it can tell how much RAM and speed **every** compression level of **any** model would need on your machine. It works with models newer than Tlav itself. `tlav models --update` refreshes the curated catalog.

---

## Less RAM, same quality

Everything below is **measured**, not assumed.

| Technique | What it saves | Quality impact | How Tlav makes sure |
|---|---|---|---|
| **Weight compression** (quantization) | 16 → ~5 bits per weight, 3–4× smaller | Q8/Q6: practically none; Q5: ~1%; Q4_K_M: ~1–3% | Measures perplexity vs the original; any level over 3% is rejected and Tlav steps up a level |
| **Importance-matrix compression** | Same size, protects the weights that matter most | Measured on a 0.5B model: Q3_K_M loss halved (+8.2% → +3.9%) | Tries standard *and* importance-matrix compression, keeps whichever measures better |
| **Compressed context cache** (q8_0 KV) | Halves the memory for conversation history: 2× longer chats, or a bigger model | Measured: +0.3–0.5% | `--kv-check` measures it on your model; above 0.5% it falls back to full precision |
| **Speculative decoding** | (speed) | **Zero**: the output is identical | Tested: Qwen2.5 3B went **9.6 → 15.7 tok/s** with byte-identical output |
| **Right-sized buffers** | Up to 8 GB: llama-server's default prompt cache and extra chat slots | None | One slot for one user; the prompt cache only gets truly spare RAM |
| **mmap / mlock** | Fast loading, nothing swapped out | None | `mlock` only when there's spare RAM; `--low-ram` profile for tight machines |

Want zero noticeable loss? `tlav setup --max-quality` allows only Q6_K/Q8_0 (measured change under 0.5%).

A bigger model compressed to Q4_K_M almost always beats a smaller model kept at full precision. That trade is what lets Tlav run "a larger, better model" on your machine.

---

## Commands

| Command | What it does |
|---|---|
| `tlav` | Set up on first run, then open the chat |
| `tlav setup` | Everything: detect → benchmark → pick → download/compress → speed test → chat (`--web` for the browser) |
| `tlav setup --cluster` / `--hf REPO` / `--max-quality` | Use workers / any Hugging Face model / near-lossless only |
| `tlav doctor` | Hardware report, RAM budget, cluster support |
| `tlav bench [--real]` | Bandwidth benchmark; `--real` measures true tok/s and calibrates future estimates |
| `tlav recommend` | Ranked models (`--cluster`, `--task code`, `--ram 16 --ram 8` to simulate) |
| `tlav search QUERY` | Find GGUF models on Hugging Face that fit this machine or cluster |
| `tlav models [--update]` | Curated catalog and compression levels |
| `tlav compress --model ID --fit` | Compress as little as possible to fit, verify quality |
| `tlav compress --file X-f16.gguf --to Q4_K_M --kv-check` | Compress your own model |
| `tlav run [--detach] [--host 0.0.0.0]` | Start the server + web chat (0.0.0.0 = usable from phones/other PCs) |
| `tlav status` / `tlav stop` | What's running and how much memory it really uses / stop it |
| `tlav chat` | The Tlav chat (starts your model if needed) |
| `tlav worker` | Share this PC's RAM and CPU with a main computer |
| `tlav cluster scan/add/list/remove/clear` | Manage workers |
| `tlav service install [--worker]` | Start at login (launchd / systemd / Task Scheduler): an always-on AI box |
| `tlav install-runtime` | Install/rebuild llama.cpp tuned for this CPU |
| `tlav list` | Downloaded models |

Everything lives in `~/.tlav` (change it with `TLAV_HOME`).

## Use it from other apps

`tlav run` exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`. Point any app that accepts an OpenAI base URL at it (Open WebUI, Continue, editors, scripts):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello from my old iMac!"}]}'
```

## How the planner decides

For every model × compression level × (one machine or a cluster):

- **memory** = weights + context cache + runtime buffers. It must fit each machine's budget: total RAM minus room for the OS (at least 2 GB or 25%, plus 1 GB on macOS).
- **speed** ≈ usable bandwidth ÷ bytes read per token. Mixture-of-experts models only count their active experts. Clusters add up each machine's share plus network round trips.
- Models slower than `--min-tps` (default 2.5 tok/s) are dropped. For each model Tlav uses the least compression that still runs at a comfortable 5 tok/s, otherwise the fastest allowed level.
- The winner has the highest quality score after compression. A draft model is added when RAM allows.

Measured on the test machine: memory 2.7 GB actual vs 3.0 GB planned; speed 52 tok/s actual vs 55 planned.

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs the unit tests on Linux, macOS and Windows. It also runs an end-to-end job that builds llama.cpp, compresses and quality-checks a model, serves it, and splits it across a 3-machine cluster, checking that the answers match.

## License

MIT
