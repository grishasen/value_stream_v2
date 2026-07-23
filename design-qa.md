# Design QA

## Evidence

- Source visual truth: `/Users/gregory/Downloads/Online Bike Shopping App (Community)-2/Online Bike Shopping App (Community).png`
- Reference role: visual language only. The mobile commerce layout is intentionally not copied into the desktop analytics product.
- Implementation captures:
  - `artifacts/ui-audit/dark-theme/01-home.png`
  - `artifacts/ui-audit/dark-theme/02-configuration-builder.png`
  - `artifacts/ui-audit/dark-theme/03-reports.png`
  - `artifacts/ui-audit/dark-theme/04-builder-1100.png`
  - `artifacts/ui-audit/dark-theme/05-reports-1100.png`
  - `artifacts/ui-audit/dark-theme/06-reports-filters-1100.png`
- Full-view comparisons:
  - `artifacts/ui-audit/comparisons/reference-vs-builder.png`
  - `artifacts/ui-audit/comparisons/baseline-vs-dark-reports.png`

The source is 1800 × 1200 pixels. The primary implementation captures are
1280 × 720 pixels at a 1280 × 720 CSS viewport and 1× device-pixel ratio.
Responsive captures are 1100 × 800 pixels at a 1100 × 800 CSS viewport and 1×
device-pixel ratio. Density normalization is therefore 1:1 for implementation
captures. The source and implementation use different product form factors, so
comparison is based on palette, surface hierarchy, material language,
typography, radii, and action emphasis rather than coordinate matching.

## State

- Application: local Streamlit UI
- Home: initial state
- Configuration Builder: Select Template step
- Reports: Overview report with dashboard presentation
- Reports narrow view: toolbar, tiles, and Filters dialog open
- Data: example workspace without generated aggregates; error cards are expected
  product state and were included in the review

## Comparison history

1. Initial baseline comparison identified a flat light hierarchy, weak active
   navigation affordance, low differentiation between editor controls and
   content, and repetitive low-emphasis report errors. These were treated as P2
   visual-usability findings.
2. The first dark implementation added a deep-navy surface ladder, royal-blue
   actions, cyan signal accents, raised cards, stronger borders, and visible
   focus treatment. The active navigation state initially depended on
   Streamlit's URL state and did not remain reliable after navigation.
3. The navigation state was replaced with a keyed page container and verified
   in the rendered browser. The active item now has a cyan rail and icon, a
   raised blue surface, and computed 3.5 px accent border.
4. Final 1280 × 720 and 1100 × 800 captures were compared side by side. No P0,
   P1, or P2 mismatch remained. The only residual P3 observation is that one
   long template name abbreviates visually at 1100 px; its full accessible name
   and dropdown remain available.

## Verification

- Browser-rendered screenshots were captured, not inferred from source code.
- Home, Configuration Builder, Reports, and Reports Filters states were opened.
- Reports presentation changed from Dashboard to Inspect and the status banner
  updated accordingly.
- Filters dialog opened with time presets, dimensions, advanced filters, and
  Clear action available.
- Browser console errors and warnings: none.
- At 1100 px, `scrollWidth` equaled `clientWidth`; no horizontal overflow or
  clipped buttons, inputs, or comboboxes were found.
- Token contrast and UI guardrail tests passed.

Final result: passed
