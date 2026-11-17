# Single-page redesign verification

Branch: `design/single-page-predictor`. The branch was created before website edits. The supplied `DESIGN.md` defines the midnight canvas, neutral actions, restrained coral brand detail and floating navigation. The site uses its permitted system-font fallback and introduces no frontend runtime dependencies.

## Preserved evidence and behaviour

| Rubric area | Evidence on the single page |
| --- | --- |
| Problem formulation and scope | Research introduction: one recorded session, Revenue target, prediction-time limits |
| Data understanding | Dataset source, attribution, feature inventory, quality checks and training-only exploration |
| Preprocessing and statistical reasoning | Pipeline rationale, training-fold fitting, encoding, scaling and imbalance decisions |
| Baseline and method selection | Majority baseline, Logistic Regression, Decision Tree and Random Forest; bounded searches and assumptions |
| Evaluation design and metrics | Saved 70/15/15 split, training-only CV, validation selection, frozen testing, purchase-focused metrics |
| Results, errors and limitations | Full result tables, error examples, subgroups, PageValues removal, duplicate sensitivity and uncertainty |
| Reproducibility and integrity | Executed notebooks, source bundle, exact split/results downloads, checksum and saved model-run provenance |
| Useful website interaction | Real single-session/CSV predictions in the hero; separate validation threshold lab |
| Individual understanding | Method explanations and executed cell-by-cell notebook support preparation; an individual's viva performance cannot be verified by a website test |

The six-input predictor remains separate from the full-feature research benchmark. No training data, split, model parameters, decision cutoff, notebook or saved prediction metrics were changed for this redesign. The research run identifier records source at the time of fitting; the presentation renderer has subsequently changed.

## Automated verification

- Python suite: 15 tests passed. Six existing Pandas deprecation warnings are non-failing.
- Python/browser prediction parity: all 3,703 cases passed, maximum score difference 0; attribution sums and CSV validation/export checks passed. The 10,000-row batch check passed.
- Threshold parity: six Python/JavaScript fixtures passed, including equality and zero predicted positives.
- Generated HTML: balanced nesting, unique IDs, no unresolved template tokens, local assets/anchors and source ZIP integrity checked by `scripts/verify_artifacts.py`.
- Accesslint 0.21.0: 94 rules, zero skipped rules and zero reported violations in the initial rendered state at 764 × 485. A readiness-selector attempt timed out; the successful scan used the loaded page without that optional selector. Three earlier structural findings were fixed and the same scan was rerun.
- JavaScript syntax checked for the workspace, research interaction and compatibility redirect.

## Browser checks

Local preview used the in-app browser. Checked widths: 320, 390, 768, 1024 and 1440 CSS pixels; no page-level horizontal overflow was observed. Wide evidence tables retain their own scroll regions, including keyboard scrolling.

- Fresh input: product pages 23, product seconds 620, information pages 2, information seconds 40, new visitor, weekend yes → purchase predicted, 23.3/100.
- Empty submission: six fields identified; linked error summary receives focus. Corrected inputs can be scored.
- Editing a scored input marks the result stale and disables its download until rescoring.
- Historical single-session example is explicitly labelled as a training example and produces a real model score.
- CSV example: five rows scored, one purchase predicted and four no-purchase predictions, with no invalid rows. Ranked output and row notes remained available.
- Keyboard tab selection uses arrow keys and roving tabindex. The skip link moves focus into the predictor. Native inputs and expandable evidence remain operable.
- Validation slider: at 1.00, TP 0 / FN 286 / FP 0; at 0.00, TP 286 / FN 0 / FP 1564. Reset restores 0.50.
- `research.html#workspace-model` redirects to the single page, keeps the fragment and opens the containing evidence disclosure.
- Missing model file: prediction and CSV file controls stay disabled, an actionable error appears, and the separately loaded research lab remains usable. No substitute scores are produced.
- Successful-page console check reported no warnings or errors.

This is bounded development QA, not a claim of full WCAG conformance. Actual screen-reader announcements, assistive-technology user testing, browser zoom/text-spacing changes and other browsers were not exercised. Reduced-motion support is implemented in CSS; its operating-system preference was not changed during testing. The project author has confirmed that human verification is complete. This report records the bounded automated and browser checks above; that confirmation does not add unperformed assistive-technology tests to this record.

The branch is reviewable separately from `main`; its GitHub Pages workflow continues to publish `main` only.
