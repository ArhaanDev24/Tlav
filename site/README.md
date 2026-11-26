# Tlav website

A single self-contained page (`index.html`): CSS, a canvas pixel field and vanilla JavaScript, no build step. The only external request is Google Fonts (Inter Tight, JetBrains Mono, Silkscreen).

**Preview locally:** open `index.html`, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:** Netlify / Cloudflare Pages / Vercel (drag in the `site` folder, no build command), or GitHub Pages from `/site` (needs a public repo or a paid plan for private repos).

## What's on it
- **Hero:** dark forest background, Space Grotesk headline that builds letter by letter, a halftone pixel wave (green to orange, swells near the cursor), and a pixel sculpture that morphs chip → arrow → AI.
- **Type band:** orange, scroll-driven "your old pc can think".
- **Watch it work:** four step cards (Measures, Chooses, Compresses, Answers); each plays its small animation once when scrolled into view.
- **What can your computer run?** memory slider using the same catalog and formulas as `tlav/planner.py`.
- **Feel the speed:** example replies streamed at measured speeds.
- Cluster layer split, measured results and install commands.

## Colours
Sage palette with a dark forest hero and an orange accent; headlines in Space Grotesk, body in Inter Tight. Sections stay light; the results and cluster sections get a pulsing pixel field in one corner instead of a solid colour band.

## Keeping the tool in sync
The catalog in the page is generated from `tlav/data/catalog.json`. After changing the catalog, regenerate the `MODELS` and `QUANTS` arrays in the script.
