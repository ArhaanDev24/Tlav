# Tlav website

A single self-contained page (`index.html`): CSS, a canvas pixel field and vanilla JavaScript, no build step. The only external request is Google Fonts (Inter Tight, JetBrains Mono, Silkscreen).

**Preview locally:** open `index.html`, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:** Netlify / Cloudflare Pages / Vercel (drag in the `site` folder, no build command), or GitHub Pages from `/site` (needs a public repo or a paid plan for private repos).

## What's on it
A minimal page in one sage palette:
- **Hero:** letter-by-letter headline, blinking pixel cursor, and a pixel sculpture that assembles into a chip, an upward arrow and "AI" in turn (pixels dodge the cursor; click to scatter them), over a pixel field that ripples on each change.
- **Watch it work:** a scroll-driven story where a Tlav window measures the machine, picks a model, compresses it and answers.
- **What can your computer run?** memory slider using the same catalog and formulas as `tlav/planner.py`.
- **Feel the speed:** pick a question and watch an example reply stream at the measured speed of a 16 GB desktop or a 4 GB laptop.
- Cluster layer split, three measured results and a one-line install.

## Colours
One palette, Sage Bright: light sage background, forest green and a signal-orange pixel accent. Sections stay light; the results and cluster sections get a pulsing pixel field in one corner instead of a solid colour band.

## Keeping the tool in sync
The catalog in the page is generated from `tlav/data/catalog.json`. After changing the catalog, regenerate the `MODELS` and `QUANTS` arrays in the script.
