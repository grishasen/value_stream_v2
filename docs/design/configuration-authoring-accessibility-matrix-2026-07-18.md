# Configuration authoring accessibility matrix

**Date:** 2026-07-18
**Scope:** Build, Configuration Builder, and AI Configuration Studio
**Release rule:** Automated checks are necessary but do not replace the named
keyboard and screen-reader journeys below.

The matrix tests behavior, not pixel similarity. YAML and IDs may be visually
collapsed, but their controls still need accessible names and keyboard access.
Diff meaning must be written in text; green/red color alone is never the
contract.

## Current evidence status

On 2026-07-18, the in-app Chromium browser on macOS completed the light-theme
Build, Builder, and deterministic Studio journeys at the default desktop
viewport and at 390×844. The narrow pass had no document-level horizontal
overflow, Studio controls reflowed to one column, every Builder editor opened
without a false draft, and the final actions remained reachable. A 720-pixel
layout surrogate also reflowed without horizontal overflow. Live metadata
inspection confirmed the title **Value Stream** and the configured analytics
shortcut icon.

This evidence does not close CA-203. Manual Tab/Shift+Tab traversal,
VoiceOver/NVDA, true browser 200% zoom, and a live dark-theme browser pass
remain release checks. The 720-pixel surrogate is useful reflow evidence, not
a substitute for browser zoom or assistive technology.

| Mode | Viewports / zoom | Required checks | Release evidence |
|---|---|---|---|
| Keyboard only | 1440×900, 390×844, browser 200% | Skip in logical task order; visible focus; Build choice; jump outline; Back/Continue; schema mapping; review choice; Apply; outcome handoff; no keyboard trap in popovers/expanders. | Browser journey plus manual Tab/Shift+Tab/Enter/Space pass. |
| Screen reader | Desktop at 100% and 200% | Page/section headings form a useful outline; current step and validation object/revision are named; status and errors are announced as text; controls have unique names; tables expose headers; collapsed technical detail names its contents. | Accessibility-tree snapshot plus VoiceOver or NVDA journey. |
| Low vision | Light/dark, desktop/narrow, 200% | Text and interactive contrast meet WCAG AA; content reflows without two-dimensional page scrolling; focused control remains visible; no clipped primary action or status. | Token contrast unit checks and browser screenshots. |
| Non-color meaning | Light/dark | Add/remove/change/rejected/invalid states have text or icons with names; success, warning, and error copy is explicit. | Review fixtures and DOM text assertions. |
| Reduced motion | System reduced-motion preference | Nonessential transitions/animations are disabled; progress remains understandable in text. | Theme guardrail and browser emulation. |

## Journey assertions

### Cold Build entry

1. The first heading is **Build**.
2. **Start from sample** and **Configure manually** are buttons with distinct
   descriptions in reading order.
3. Focus indication is visible against both light and dark surfaces.

### Builder object revision

1. The compact outline identifies current step and position.
2. Back precedes the current task and Continue follows it in focus order.
3. Editing creates a text-labelled draft revision and one enabled primary
   **Apply to workspace** action.
4. Discard is separately named and never masquerades as navigation.
5. Outcome copy states either **Report ready** or **Data refresh required**.

### Studio draft and review

1. Sample origin, preview-only meaning, and production source plan are read as
   separate concepts.
2. Required-field selectors expose field name and type and cannot accept an
   arbitrary value.
3. Consent states provider/model and whether examples are included.
4. Operation stages and failure recovery are textual, not spinner-only.
5. Review bundles expose summary, consequence, validity, and removal state;
   exact YAML is optional collapsed detail.
6. Invalid revisions have no acceptance or apply control.

## Automated guardrails

- Native Streamlit components only outside the established theme layer.
- Visible `:focus-visible` outline and `prefers-reduced-motion` treatment.
- Light and dark token pairs meet WCAG AA for body text and primary controls.
- Browser metadata retains the configured title and icon.
- App tests cover cold entry, navigation, disclosure, invalid apply blocking,
  and outcome handoff.

Manual assistive-technology evidence must name the browser, OS, reader/version,
viewport, and date. A screenshot alone cannot close a screen-reader row.
