# Tlav

**Turn old PCs and laptops into local AI workstations.**

That 2013 iMac in the corner, the ThinkPad in a drawer, the tower under the desk can all run a capable AI assistant, fully offline. Tlav does the hard parts for you:

1. **Detects** your hardware: RAM, CPU cores, AVX/AVX2/AVX-512 support, GPU, disk.
2. **Benchmarks** memory bandwidth, which sets how fast a model can talk on a CPU.
3. **Picks the best model** that fits your RAM and still responds at a usable speed.
4. **Compresses** (quantizes) it so a bigger, smarter model fits, then **checks the quality** to make sure it still answers about as well as the original.
5. **Tunes and runs** it: threads, context size and batch size are set for your machine. You get a web chat in your browser and an OpenAI-compatible API.

```bash
tlav setup
```

That's it. One command, and you're chatting with a local AI.

---

## Install

Requires Python 3.8+ (no other Python packages). On first setup Tlav installs [llama.cpp](https://github.com/ggml-org/llama.cpp), the fastest CPU inference engine for old hardware.

**macOS / Linux**
```bash
curl -fsSL https://raw.githubusercontent.com/OWNER/Tlav/main/install.sh | bash
```
or, from a clone:
```bash
git clone https://github.com/OWNER/Tlav && cd Tlav && ./install.sh
```

**Windows (PowerShell)**
```powershell
./install.ps1
```

### Intel Macs (2010–2020), including the 2013 iMac
Old macOS versions aren't supported by prebuilt llama.cpp binaries, so Tlav builds it from source with your CPU's AVX2 instructions switched on and Metal switched off (old Intel/NVIDIA Mac GPUs are slower than the CPU for this). You need Apple's command line tools once:
```bash
xcode-select --install
```
If `cmake` is missing, Tlav installs it with pip automatically.

## What will my machine run?

Check before you download anything:
```bash
tlav doctor              # what Tlav sees
tlav recommend           # ranked models for this machine
tlav recommend --ram 16  # simulate another machine
```

Typical results on a 2013 iMac (quad-core Haswell i5, DDR3-1600):

| RAM   | Best pick                       | RAM used | Speed        |
|-------|---------------------------------|----------|--------------|
| 8 GB  | Qwen2.5 7B @ Q4_K_M             | ~4.9 GB  | ~3–4 tok/s   |
| 16 GB | Qwen3 8B @ Q4_K_M               | ~6.1 GB  | ~3 tok/s     |
| 32 GB | Qwen3 30B-A3B (MoE) @ Q5_K_M    | ~21 GB   | ~6–7 tok/s   |

Want quicker replies? `tlav setup --min-tps 5` prefers faster models (for example Qwen3 4B at ~5–6 tok/s).
For programming help, use `tlav setup --task code`.

> Mixture-of-experts models (like Qwen3 30B-A3B) are perfect for old machines with lots of RAM: they store 30B parameters of knowledge but read only ~3B per word, so they run faster than an 8B model.

## Commands

| Command | What it does |
|---|---|
| `tlav setup` | Everything: detect → benchmark → pick → download/compress → tune → run |
| `tlav doctor` | Hardware report and RAM budget |
| `tlav bench [--real]` | Memory bandwidth benchmark; `--real` measures true tokens/s with llama-bench |
| `tlav recommend` | Ranked model list (`--task code`, `--aggressive`, `--ram 16` to simulate) |
| `tlav models` | Model catalog and compression levels |
| `tlav compress --model qwen3-8b --fit` | Compress a model as little as possible to fit this machine, then verify quality |
| `tlav compress --file my-model-f16.gguf --to Q4_K_M` | Compress your own GGUF file |
| `tlav run` | Start the server + web chat (`--host 0.0.0.0` to use it from your phone or other PCs) |
| `tlav chat` | Chat in the terminal |
| `tlav install-runtime` | Install/rebuild llama.cpp tuned for this CPU |
| `tlav list` | Downloaded models |

Everything lives in `~/.tlav` (change it with `TLAV_HOME`).

## Using it from other apps

`tlav run` exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, so any app that lets you set an OpenAI base URL works (Open WebUI, Continue, editors, scripts):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello from my old iMac!"}]}'
```

## Compression: how close to the original is it?

Tlav compresses with quantization: weights are stored in small blocks with a shared scale and fewer bits per value (16 bits → about 5). It is not mathematically lossless, so Tlav doesn't just assume it worked. It **measures perplexity** (how well the model predicts text it hasn't seen; lower is better) for the original and the compressed model, and rejects any level that loses more than 3% (`--tolerance`). If a level fails, Tlav automatically tries the next, less compressed one.

| Level  | Size vs original | Quality kept | Notes |
|--------|------------------|--------------|-------|
| Q8_0   | 53% | ~99.9% | practically identical |
| Q6_K   | 41% | ~99.6% | practically identical |
| Q5_K_M | 36% | ~99%   | near-lossless |
| Q4_K_M | 30% | ~97–98% | default floor, best size/quality balance |
| Q3_K_M | 24% | ~94%   | only with `--aggressive`, to fit a much better model |

A bigger model compressed to Q4_K_M almost always beats a smaller model kept at full precision. That trade is what lets Tlav run "a larger, better model" on your machine.

By default `tlav setup` downloads files that were already compressed with importance-matrix quantization, which preserves slightly more quality than plain local quantization and saves you hours on an old CPU. Use `--local-compress` to do it all on your machine instead.

## How the planner decides

For every model × compression level:

- **memory** = weights + KV cache for the context length + runtime overhead. It must fit in your RAM budget (total RAM minus room for the OS: at least 2 GB or 25%, plus 1 GB extra on macOS).
- **speed** ≈ usable memory bandwidth ÷ bytes read per token (for MoE models, only the active experts count).
- Models slower than `--min-tps` (default 2.5 tokens/s) are dropped. For each model, Tlav uses the least compression that still runs at a comfortable 5 tok/s, otherwise the fastest allowed level.
- The winner is the highest quality score after compression.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
