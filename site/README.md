# Tlav website

A single self-contained page (`index.html`): inline SVG artwork, CSS and vanilla JavaScript, no build step. The only external request is the Inter / Inter Tight / JetBrains Mono fonts from Google Fonts.

**Preview locally:** open `index.html` in a browser, or run `python3 -m http.server -d site` and visit http://localhost:8000.

**Host it:**
- **GitHub Pages:** Settings → Pages → deploy from branch `main`, folder `/site`. (Pages on a private repo needs a paid GitHub plan; otherwise make the repo public or use one of the options below.)
- **Netlify / Cloudflare Pages / Vercel:** drag the `site` folder in, or point the project at it with no build command.

Sections: electric-blue opening with the headline over a scroll-driven "your old pc can think" type wall, sticky how-it-works story, cluster diagram, measured results, terminal demo, slogans and install commands. Scroll animations respect `prefers-reduced-motion`.
