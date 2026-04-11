# SDG Hub Documentation Site

Next.js + Markdoc site that renders documentation from `../docs/`.

## Development

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Regenerate API Reference

```bash
cd .. && uv run python website/scripts/generate_api_reference.py
```

## Build

```bash
npm run build
```

Static output goes to `out/`. Deployed to GitHub Pages via `.github/workflows/deploy-docs.yml`.
