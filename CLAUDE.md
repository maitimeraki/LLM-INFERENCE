# SparseLLM Development Guidelines

## Design System
Always read `DESIGN.md` before making any visual or UI decisions.

All font choices, colors, spacing, and aesthetic direction are defined there. The system is built around one north star: **fastest path to inference with zero friction**.

Do not deviate from DESIGN.md without explicit user approval. This includes:
- Typography: Geist Sans (UI), IBM Plex Mono (data), JetBrains Mono (code/logs)
- Color palette: Blue accent only (no secondary colors unless approved)
- Layout: Left sidebar (180px fixed) + sticky top bar (64px) + main content
- Spacing: 8px base unit, comfortable density
- Motion: Minimal, functional only

In code review and QA, flag any UI that doesn't match DESIGN.md values (wrong font, wrong spacing, gratuitous animations, etc.).

## Product Positioning
SparseLLM's web interface is for **self-hosted teams who want to run MoE inference locally with minimal setup friction**.

Users are business users and integration partners, not ML researchers. The interface should:
- Abstract away MoE complexity
- Get users to "model running" in under 5 minutes
- Provide clear status/observability without information overload
- Never require deep technical knowledge

## Architecture Notes
- Backend: Python (FastAPI recommended for async inference + WebSocket support)
- Frontend: React/TypeScript (recommended for component reuse and type safety)
- Deployment: Self-hosted (users deploy in their own infrastructure)
- No SaaS billing or authentication needed (unless user adds it later)

## Key Interfaces to Build
1. **Model loader:** Browse/upload models, click "Load," see progress
2. **Inference playground:** Text input → Generate → See output + metrics
3. **Metrics dashboard:** Tokens/sec, cache hit %, VRAM usage, uptime
4. **Settings:** Model configuration, cache size, device selection (GPU/CPU/mixed)
5. **Status monitor:** Real-time inference health, error logs

Keep each interface focused. No feature creep. When in doubt, ask: "Does this speed up the path to inference or add friction?"
