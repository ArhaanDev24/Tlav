# Tlav website

A single self-contained page (`index.html`): CSS, a canvas pixel field and vanilla JavaScript, no build step. The only external request is Google Fonts (Inter Tight, JetBrains Mono, Silkscreen).

**Preview locally:** open `index.html`, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:** Netlify / Cloudflare Pages / Vercel (drag in the `site` folder, no build command), or GitHub Pages from `/site` (needs a public repo or a paid plan for private repos).

## What's on it
A deliberately minimal page in one sage palette: a hero with an animated pixel field and pixel arrow, a scroll-driven type band, a **"What can your computer run?"** memory slider (same catalog and formulas as `tlav/planner.py`, assuming a typical Windows PC; the real tool measures your machine), three steps, the cluster layer split, three measured results and a one-line install.

## Keeping the tool in sync
The catalog in the page is generated from `tlav/data/catalog.json`. After changing the catalog, regenerate the `MODELS` and `QUANTS` arrays in the script.
