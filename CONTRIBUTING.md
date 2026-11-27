# Contributing to Tlav

Thanks for helping old computers think. Tlav is small on purpose: pure Python (standard library only) plus llama.cpp, which Tlav builds for each machine.

## Getting started

```bash
git clone https://github.com/ArhaanDev24/Tlav && cd Tlav
pip install -e ".[dev]"
pytest
```

## Good first contributions

- **Run it on your old machine** and open an issue with `tlav doctor` and `tlav bench --real` output. Real numbers from real old hardware are what make Tlav's picks accurate.
- **Model requests:** if a good new model is missing from `tlav recommend`, open a model request.
- **Bugs:** include your OS, Python version, the command you ran and the full output (`TLAV_DEBUG=1` shows the traceback).

## Ground rules

- No new runtime dependencies: Tlav must install with nothing but Python 3.8+.
- Keep it honest: speeds and quality numbers in docs must be measured, not assumed.
- Add a test for every bug fix (`tests/`), and keep `pytest` green on Linux, macOS and Windows.
- Terminal output must still work with `NO_COLOR=1`, when piped, and on old Windows consoles.
