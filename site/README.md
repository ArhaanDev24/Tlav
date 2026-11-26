# Tlav website

A single self-contained page (`index.html`): CSS, a canvas pixel field and vanilla JavaScript, no build step. The only external request is Google Fonts (Inter Tight, JetBrains Mono, Silkscreen).

**Preview locally:** open `index.html`, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:** Netlify / Cloudflare Pages / Vercel (drag in the `site` folder, no build command), or GitHub Pages from `/site` (needs a public repo or a paid plan for private repos).

## What's on it
- Pixel dot-grid hero with a stepped pixel arrow.
- Scroll-driven "your old pc can think" type wall.
- **"What can your computer run?"** tool: pick RAM, OS and one or two computers, and it shows the model Tlav would choose, its compression, speed, context and a memory map. It uses the same catalog and formulas as `tlav/planner.py` (checked to give identical picks); typical speeds per RAM size are assumptions, and the real tool measures them.
- How it works, cluster layer split, measured results, a terminal example, slogans and install commands.

## Colours
Three palettes, switchable in the footer or by link: `#palette-aluminium` (default), `#palette-sage`, `#palette-night`.

## Keeping the tool in sync
The catalog in the page is generated from `tlav/data/catalog.json`. After changing the catalog, regenerate the `MODELS` and `QUANTS` arrays in the script (see the comment there).
