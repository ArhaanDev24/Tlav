# Tlav website

A single self-contained page (`index.html`): inline SVG artwork, CSS and vanilla JavaScript, no build step. The only external request is the Inter / Inter Tight / JetBrains Mono fonts from Google Fonts.

**Preview locally:** open `index.html` in a browser, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:**
- **GitHub Pages:** Settings → Pages → deploy from branch `main`, folder `/site`. (Pages on a private repo needs a paid GitHub plan; otherwise make the repo public or use one of the options below.)
- **Netlify / Cloudflare Pages / Vercel:** drag the `site` folder in, or point the project at it with no build command.

**Five style directions** are built in. A picker at the bottom of the page switches them live, and a link like `index.html#style-3` opens a specific one:

| # | Name | Palette |
|---|---|---|
| 1 | Lavender Grid | soft lilac, indigo ink, magenta pixels |
| 2 | Cloud Cobalt | pale sky, cobalt type, rose pixels |
| 3 | Mint Static | fresh mint, deep teal, orchid pixels |
| 4 | Soft Dusk | gentle dark mode, periwinkle, pink pixels |
| 5 | Sorbet | blush ground, violet type, magenta pixels |

Once one is chosen, set it as the default in the script (`apply(... "1" ...)`) and remove the dock.

Sections: pixel dot-grid hero with a stepped pixel arrow, scroll-driven "your old pc can think" type wall, sticky how-it-works story on a pixel-art iMac, pixel-art cluster diagram, measured results, terminal demo, slogans and install commands. Animations respect `prefers-reduced-motion`.
