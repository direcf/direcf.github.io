# Private Portfolio Page — Design Spec

**Date:** 2026-09-04
**Author:** Seungjun Lee (with Claude)
**Target file:** `src/pages/portfolio/index.astro` (rewrite existing)
**Route:** `/portfolio/` (already exists, private area alongside `/property/`)

---

## 1. Goal

A single-page, visually premium personal portfolio/resume for **Seungjun Lee**, targeting
Big Tech AI roles. Replaces the current placeholder resume with accurate, professionally
framed content and a refined "Light Editorial" design. Includes a PDF-resume download.
Stays behind the existing password lock (private).

## 2. Design Direction — "Light Editorial / Print" (Direction B)

- **Palette:** off-white `#f7f5f0` bg, ink `#1a1a1a` text, single amber accent `#b5762a`,
  hairline `#e2dccd`, card `#fffdf8`.
- **Type:** Georgia (serif) for display/name/section school names; Inter / system-ui sans
  for body and labels; monospace only for the behavior-class chips.
- **Feel:** generous whitespace, confident type scale, single accent, hairline rules,
  subtle hover (project left-border turns amber). No skill bars, no gradients, no stock icons.
- Content language: **English** (Big Tech target).

## 3. Page Structure (order confirmed as v3)

Sticky top bar (name left, "Download PDF Resume" button right).

1. **Hero** — label "AI · Vision · Multimodal", name, role ("AI / Vision Research Engineer ·
   SK Telecom"), rule, summary paragraph (highlights 15 patents / CVPR 3rd / Minister's Award),
   two CTAs (View Work, Download PDF), contact meta line.
2. **Education**
   - KAIST — M.S. Electrical Engineering · RITL Lab (Adv. Jong-Hwan Kim) · 2020.09–2022.08 ·
     GPA 4.15/4.3 · Thesis: CAL-ODWC — Continual Adaptation Learning for Object Detection
     under Weather Changes.
   - Hanyang University — B.S. Electronic Engineering · 2014.03–2020.08 · GPA 4.04/4.5 · Cum Laude.
3. **Experience** — SK Telecom · Vision Engineer · AI R&D Center / Vision Algorithm Team ·
   Jan 2023 – Present. Three project sub-blocks (end-date desc):
   - **CareVia** (2023.05–2026.05): multi-camera action-recognition system; 9 behavior classes
     (push, kick, push-pull, lie-down, hit-head, falldown, run, loitering, jump) shown as
     monospace chips; real-time FR + ReID pipeline on AI Box; pose-estimation model trained on
     3M+ images; Minister's Award (ICT Award Korea 2025); 12 welfare centers nationwide since 2023.
   - **SynapsEgo** (2025.06–2025.09): video semantic segmentation (event localization); LoRA
     fine-tuned Qwen3-MoE / Qwen2.5-VL to surpass SOTA (Gemini-2.5-Pro) for National Police
     Agency video summary & report; Recursive Agentic LLM pipeline; SOP-based work-procedure
     recognition (Yonsei Nursing College PoC).
   - **SK Hynix Industrial Safety** (2025.04–2025.06): 4 safety AI functions (fall, collapse,
     fire/smoke, intrusion); false alarms cut 96%; early qualification for the main contract.
4. **Awards & Challenges**
   - Minister's Award — ICT Award Korea (Ministry of Science & ICT), 2025 — for CareVia.
   - CVPR AVA Challenge — 3rd Prize (Keypoint Track), 2024.
   - SKT AI Frontier (4th cohort) — Grand Prize / 1st place (Vision Retrieval System), 2024.
5. **Patents** — headline "15 filed domestic patents (co-inventor)" + 4 areas with counts and
   one-line descriptions: Video Understanding (5), Visual Perception (4), Agentic/Multimodal (4),
   Industrial Safety (2).
6. **Certifications** — AWS Certified Generative AI Developer – Professional (AIP-C01), 2026.09.
7. **Skills** — LLM/VLM Agentic Pipelines; ML/DL; Languages & Infra (tag groups).
8. **Footer** — "Let's build something." + contact line (email · github · site) + updated date.

Contact: email `direcf1520@naver.com`, `github.com/direcf`, `direcf.github.io`, Seoul.
LinkedIn omitted for now (no public profile URL yet; trivial to add a line later).

## 4. PDF Download — print stylesheet

- Button (top bar + hero CTA) calls `window.print()`.
- `@media print` reformats the page to a clean A4 resume: hide top bar / lock screen / CTA
  buttons; force light background; remove sticky positioning and excess padding; keep amber
  accent subtle; avoid section page-break splitting (`break-inside: avoid` on blocks).
- Single source of truth — the page IS the resume; no separate PDF file to maintain.

## 5. Privacy / Lock (reuse existing, unchanged)

- Keep the current mechanism verbatim: `#lockScreen` / `#resume` toggle, SHA-256 password hash
  `99bbf103da194bbcab3a2b855888fbe0ee8432d83d1dde4bc6a55ffb4edd1416`, `sessionStorage['port:auth']`.
- Password is unchanged (hash preserved). Lock screen keeps its existing dark styling; the
  unlocked resume adopts the new Light Editorial design.
- Note: this is a client-side gate on a static site (deters casual viewing, not a hard secret).

## 6. Implementation Notes

- Rewrite `src/pages/portfolio/index.astro`, keeping `BaseLayout` usage
  (`activeSlug="portfolio"`, crumbs Personal → Portfolio) and the lock `<script>` block.
- All new styles scoped within the page's `<style>`; class prefixes avoid collision with
  BaseLayout global styles.
- Ensure Inter (or acceptable system sans) renders; fall back to system-ui if not bundled.
- Responsive: single column already; collapse header/date rows and reduce name size under 600px.

## 7. Out of Scope

- Changing the password or lock UX.
- A separately designed/exported PDF file.
- Adding LinkedIn/Scholar links (none available yet).
- Any change to `/property/` or other routes.
