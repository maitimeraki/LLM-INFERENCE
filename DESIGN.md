# Design System — SparseLLM Interactive Interface

## Known Production Issues & Workarounds

### Device Mapping + Offload Folder Alignment Bug
**Issue:** When using `device_map="auto"` with `offload_folder`, transformers' accelerate library triggers memory-mapping that doesn't respect 16-byte alignment for MoE expert weights, causing `RuntimeError: expected data_ptr to be aligned to 16 bytes`.

**Workaround (implemented in `sparse_llm/models/adapters.py`):**
- Use `max_memory` dict instead of `device_map="auto"` when offloading
- `max_memory` maps device names to memory budgets, letting accelerate handle offloading without broken memory-mapping code path
- Works across all architectures (Qwen, Mixtral, Llama, etc.)
- See `TransformersCausalLMAdapter.load()` and `_compute_max_memory()` for implementation

**When to apply:** Any feature that combines device mapping with SSD/CPU offloading should use this pattern.

---

## Product Context
- **What this is:** A self-hosted web interface for SparseLLM that lets business users and integration partners run MoE inference models with minimal setup friction.
- **Who it's for:** Business users, integration partners, and teams deploying SparseLLM in their own infrastructure. Non-technical stakeholders who need reliable inference without complexity.
- **Space/industry:** Infrastructure / AI tooling. Peers: Ollama, vLLM, text-generation-webui, but positioned as the fastest path to inference for business users.
- **Project type:** Dashboard / infrastructure management UI
- **North Star:** Users can deploy SparseLLM, load a model, and start generating inference in under 5 minutes. Zero friction, zero learning curve.

## Aesthetic Direction
- **Direction:** Utilitarian Industrial
- **Decoration level:** Minimal
- **Mood:** Modern infrastructure software that prioritizes clarity over ornament. Serious, confident, built by engineers who care. Think Stripe's API dashboard or Vercel's deployment interface.
- **Philosophy:** Function-first. No gradients, no decorative elements. Contrast, typography, and spacing do all the work. Every pixel serves a purpose.

## Typography
- **Display/Hero:** Geist Sans (Medium, 28px) — Clean, modern, technical credibility. Sets tone as "serious engineering."
- **Body/UI:** Geist Sans (Regular, 14px primary / 13px secondary) — Consistent family throughout. Excellent readability. Supports tabular figures for metrics.
- **Data/Tables:** IBM Plex Mono (Regular, 12px) — Monospace signals data credibility. All metrics, token counts, and model names render here. Fixed-width alignment.
- **Code/Logs:** JetBrains Mono (Regular, 11px) — When users see logs or model errors. Familiar to developers. Warm undertone, excellent screen rendering.
- **Loading:** Google Fonts (Geist) or Bunny Fonts (self-hosted alternative)
- **Scale:** 
  - xs: 11px (labels, captions)
  - sm: 12px (secondary text)
  - base: 14px (body text)
  - lg: 18px (section headers)
  - xl: 24px (page headers)
  - 2xl: 28px (hero/display)

**Rationale:** One font family (Geist) + monospace (Plex Mono) reduces cognitive load. No system fonts or overused defaults (Inter, Roboto). Monospace for numbers builds instant trust. Geist reads as "built by engineers."

## Color
- **Approach:** Restrained. Neutral foundation, single vibrant accent. Color is sparse and purposeful.
- **Primary Accent:** `#2563EB` (Electric blue) — Actions, primary CTAs, emphasis. Confidence without aggression.
- **Secondary Accent:** `#10B981` (Emerald green) — Success states, valid operations.
- **Warning/Caution:** `#F59E0B` (Amber) — Warnings, attention-needed states.
- **Error/Critical:** `#EF4444` (Red) — Errors, failures, critical states.
- **Info/Neutral:** `#06B6D4` (Cyan) — Information, secondary emphasis.

**Neutrals:**
- Surface (background): `#FAFAFA` (Almost-white, paper-like)
- Elevated (cards, modals): `#FFFFFF` (Pure white)
- Text primary: `#0F0F0F` (Near-black, high contrast, easier than pure black)
- Text secondary: `#666666` (Muted gray, for labels and helper text)
- Border/divider: `#E5E5E5` (Light gray, structure without noise)
- Disabled: `#D1D5DB` (Medium gray for inactive elements)

**Dark Mode Strategy:**
- Surface: `#0F0F0F` (invert to dark)
- Elevated: `#1A1A1A` (slightly lighter)
- Text primary: `#FAFAFA` (invert to light)
- Text secondary: `#A0A0A0` (lighter gray)
- Border/divider: `#333333` (darker gray)
- Accent colors: Desaturate by 15% to reduce eye strain (e.g., blue becomes `#3B82F6`, green becomes `#34D399`)

**CSS Variables:**
```css
--color-surface: #FAFAFA;
--color-elevated: #FFFFFF;
--color-text-primary: #0F0F0F;
--color-text-secondary: #666666;
--color-border: #E5E5E5;
--color-accent-primary: #2563EB;
--color-accent-success: #10B981;
--color-accent-warning: #F59E0B;
--color-accent-error: #EF4444;
--color-accent-info: #06B6D4;
```

**Rationale:** Blue + green + neutrals feels technical and trustworthy (not playful). No secondary accent color forces clarity about what matters. Monochromatic neutrals with semantic accents prevent visual chaos.

## Spacing
- **Base unit:** 8px
- **Density:** Comfortable (breathing room around content, but data-efficient)
- **Scale:**
  - 2xs: 4px (tight spacing, text line-height adjustments)
  - xs: 8px (element margins, tight grouping)
  - sm: 16px (card padding, section margins)
  - md: 24px (major section spacing, component grouping)
  - lg: 32px (large gaps between major sections)
  - xl: 48px (page-level spacing)
  - 2xl: 64px (edge cases, hero sections)

**Component-level:**
- Card padding: 16px
- Form input height: 40px (vertical)
- Button padding: 12px vertical, 16px horizontal
- Section margins: 24px between major blocks
- Sidebar width: 180px (fixed)
- Top bar height: 64px (fixed)

**Rationale:** 8px is the infrastructure standard (Stripe, Vercel, Railway). Comfortable density balances readability with data efficiency. Users see what matters without excessive scrolling.

## Layout
- **Approach:** Grid-disciplined. Structure is always consistent.
- **Grid:** 12-column responsive grid
  - Desktop (1024px+): Full width with 1200px max-width constraint
  - Tablet (640px-1024px): Collapses to 8-column
  - Mobile (<640px): 4-column single-column layout

**Page Structure (persistent):**
1. **Left sidebar** (180px, fixed, sticky) — Navigation (Models, Deployments, Settings, Docs, Status)
2. **Top bar** (64px, fixed, sticky) — Logo, quick status indicator, settings/profile menu
3. **Main content** (right of sidebar, respects max-width) — Full page content with padding
4. **Optional right panel** (320px) — Deployment details, logs, or status (context-dependent)

**Breakpoints:**
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: 1024px+

**Max content width:** 1200px

**Rationale:** Left sidebar + sticky top bar ensures navigation is always visible. No hidden complexity. Content is the focus. Grid discipline makes the interface predictable.

## Motion
- **Approach:** Minimal-functional. Only transitions that aid comprehension, never decorative.
- **Easing:**
  - Entrance: `ease-out` (150ms) — Objects arriving feel snappy
  - Exit: `ease-in` (100ms) — Objects leaving feel light
  - State changes: `ease-in-out` (200ms) — Feels considered
- **Duration:**
  - Micro: 50-100ms (hover states, simple changes)
  - Short: 150-250ms (page transitions, modal opens)
  - Medium: 250-400ms (complex animations, state transitions)
  - Long: 400-700ms (only for loading states, never busy-work)

**Specific patterns:**
- Tab/section transitions: Fade in (100ms, ease-out)
- Hover states: Subtle background color shift, no transform scaling
- Loading states: Spinner or progress bar (no notification pop-ups)
- Status badge changes: Instant (status is factual, not performative)
- Modal entrance: Fade in + slight scale (150ms, ease-out)
- Modal exit: Fade out (100ms, ease-in)

**Rationale:** Inference is time-critical. Motion should never make users wait or question what's happening. Minimalism builds confidence and feels fast.

## Components & Interactions

**Primary CTA:** Large blue button, 40px height, 16px horizontal padding. Text: "Load Model" or "Start Inference" or "Deploy."

**Form inputs:** Vertical stacking. Label above input. Helper text below. Validation errors inline, in red. No floating labels.

**Metric cards:** 4-column grid at desktop. Each card shows one key metric (Tokens/sec, Cache Hit %, VRAM Used, Uptime). Layout changes to 2-column at tablet, 1-column on mobile.

**Status badges:** Pill-shaped. Colors match semantic palette. States: "Running" (green), "Loading" (blue), "Error" (red), "Stopped" (gray).

**Navigation:** Sidebar links use left border accent (blue) when active. Hover state: subtle background shift.

**Data tables:** IBM Plex Mono for numeric content. Row hover: `#F3F4F6` background (light gray). Striping optional but recommended for > 10 rows.

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-02 | Initial design system created | Created by /design-consultation. Self-hosted MoE interface for business users. North Star: fastest path to inference. Aesthetic: Utilitarian Industrial. |
| 2026-09-02 | Single accent color (blue only) | Forces ruthless clarity. Reduces visual noise. Users focus on actions, not competing colors. |
| 2026-09-02 | Geist Sans + IBM Plex Mono stack | Geist reads as "serious engineering," not defaults (Inter/Roboto). Plex Mono for data credibility. One font family for UI reduces cognitive load. |
| 2026-09-02 | Left sidebar + sticky top bar | Navigation always visible. No hidden state. Predictable, reduces cognitive load for non-technical users. |
| 2026-09-02 | Minimal motion (functional only) | Inference is time-critical. No decorative animations. Every animation should aid comprehension or feel fast. |
| 2026-09-02 | Comfortable spacing density | Breathing room (not compact) balanced with data efficiency. Users see what matters without excessive scrolling. |

## Design Anti-Patterns (Never Use)
- ❌ Purple/violet gradients as default accent
- ❌ 3-column feature grid with icons in colored circles
- ❌ Centered everything with uniform spacing
- ❌ Uniform bubbly border-radius on all elements
- ❌ Gradient buttons as the primary CTA
- ❌ System fonts (system-ui, -apple-system) as primary display or body
- ❌ Decorative animations that delay understanding
- ❌ "Built for X" marketing copy on infrastructure UIs
- ❌ Hidden complexity (modals instead of inline options where possible)

## Next Steps
1. Implement this design system in CSS variables and component library
2. Build the dashboard layout (sidebar + top bar + main content)
3. Create reusable components (buttons, cards, forms, badges, tables)
4. Test with business users for friction points (especially model loading and inference triggering)
5. Iterate on dark mode if it's used in evening/night deployments
