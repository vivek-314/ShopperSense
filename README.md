# ShopperSense

A reproducible undergraduate ML investigation and a complete static research report. The supplied specification is preserved in [PRD.md](PRD.md).

**Question:** How well can recorded browsing-session characteristics identify sessions that end in a purchase, compared with always predicting no purchase?

One row is one historical browsing session; Revenue=1 means purchase. This study uses recorded session summaries. It does not establish early-session performance, causal conversion uplift, customer motives, or price/quality preferences.

## Start here

The complete generated report is `docs/index.html`. Run from this repository's root directory (the directory containing this README):

```powershell
.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8765/>. If your machine blocks the Python preview executable, the dependency-free Node fallback is `node scripts/serve.cjs`, at <http://127.0.0.1:8766/>. A local preview is not a public submission URL. Double-clicking HTML does not reliably support `fetch()` for the threshold lab; use HTTP.

Repository: [vivek-314/ShopperSense-study](https://github.com/vivek-314/ShopperSense-study). The published site URL is recorded in `site_config.json` after a successful Pages deployment. No placeholder website is presented as a working link.

## What was actually run

The supplied CSV has **12,330 rows, 17 predictors, 1,908 purchases**, matching UCI's dimensions and class counts. It contains no missing cells, infinite numeric values, negative numeric values, or detected invalid category/range values. All **125 excess complete duplicates** were preserved. Raw SHA-256:

```text
b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9
```

This identifies the supplied bytes. No newly downloaded UCI archive was used to claim canonical byte identity. See [data/README.md](data/README.md) and `results/data_manifest.json` for provenance, the full schema, versions and licensing.

Six prespecified variants were fitted: majority Dummy, Logistic Regression, Decision Tree and Random Forest with full features; LR and RF without PageValues. All 20 search candidates and their three fold scores are saved. A separate group-aware sensitivity reused the selected model's frozen parameters and threshold on different rows with identical inputs kept together.

**Validation selected Random Forest**, max_depth=None, min_samples_leaf=2, n_estimators=150. Validation AP=0.725437. The validation F1 threshold grid selected **0.24**. Held-out test, full features:

| Condition | AP | Precision | Recall | F1 | Accuracy | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Majority baseline, 0.50 | 0.154595 | 0* | 0 | 0 | 0.845405 | 1564 / 0 / 286 / 0 |
| Selected forest, 0.50 | 0.725975 | 0.762431 | 0.482517 | 0.591006 | 0.896757 | 1521 / 43 / 148 / 138 |
| Selected forest, frozen 0.24 | 0.725975 | 0.567430 | 0.779720 | 0.656848 | 0.874054 | 1394 / 170 / 63 / 223 |

\*Precision is undefined with no predicted positives; zero_division=0 displays zero. Accuracy alone obscures the baseline's missed purchases. AP is **average precision**, not trapezoidal PR-AUC. Scores are not claimed calibrated.

Removing PageValues changes test AP from 0.649399 to 0.344051 for LR and from 0.725975 to 0.354664 for RF. This quantifies dependence, not proof of leakage or real-time availability. The group-aware forest test has 1,850 sessions / 287 purchases, AP=0.751015 and F1=0.687316 at the reused 0.24 threshold. It is a separate sensitivity estimate with different rows and support.

## Install and reproduce

Tested Python: **3.12.14**. Node 20+ is needed only for the small JavaScript parity test and optional fallback server; the deployed website has no server, external runtime APIs, npm dependencies or build framework.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.run_all --data data/raw/online_shoppers_intention.csv
python -m pytest -q
python -m tests.export_threshold_fixtures
node tests/check_threshold.cjs
python -m nbconvert --to notebook --execute notebooks/01_data_and_reasoning.ipynb --output executed_01.ipynb
python -m nbconvert --to notebook --execute notebooks/02_models_and_results.ipynb --output executed_02.ipynb
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

`python -m nbconvert` directly uses the active environment. Each notebook explicitly adds the project root to `sys.path`, works from `notebooks/` or the root, and imports shared source functions. The training command fits models; the notebooks inspect evidence, repeat structural checks and reproduce validation decisions without retuning test outcomes. Both must run top to bottom without hidden state. Source notebooks contain executed outputs after verification; the commands above produce new executed copies.

`requirements.txt` pins the directly used packages. `requirements-lock.txt` records the exact full tested Windows environment; for exact Windows restoration use `pip install -r requirements-lock.txt`. The direct dependency list is preferable across platforms because the full lock contains Windows-specific dependencies. Network is needed to install packages, not to train or view the local report afterward.

The implementation machine's initial training/evaluation stage took **44.9 seconds** (52.7 seconds including report generation); see `results/run.json` for the current rerun. Hardware, platform and dependencies affect runtime. Seeds, saved row IDs and the fixed parameter grids make the decisions deterministic in the tested environment. Across platforms use a numerical tolerance of **1e-10** for metrics/scores; CSV timing columns, timestamps, ZIP metadata, library rendering and notebook outputs are not byte-identical reproduction targets. Identical tested environments should reproduce selected parameters and threshold. Never change the frozen decision in response to test outcomes.

On restrictive Windows runners, set TEMP, TMP, MPLCONFIGDIR, JUPYTER_RUNTIME_DIR and IPYTHONDIR to project-local writable directories if the default sandbox directories reject writes. Kernel execution requires local process/socket access. The implementation used these overrides; they do not affect model inputs or split IDs.

## Evaluation procedure

1. Assign stable zero-based source row IDs. Preserve raw bytes and all source rows; structurally audit the whole file. Training-only EDA informs interpretation.
2. Stratify with random_state=42: hold out 15% test, then use validation fraction 15/85 on the rest. Exact sizes: **8,630 train / 1,850 validation / 1,850 test**, with **1,336 / 286 / 286 purchases**. Assert complete/disjoint coverage and save IDs with the source hash.
3. Within training-only shuffled three-fold StratifiedKFold (seed=42), use GridSearchCV(scoring='average_precision', refit=True). LR: C=[0.1,1,10], max_iter=2000. Tree: depth=[3,5,8], min leaf=[5,20]. RF: 150 trees, depth=[8,None], min leaf=[2,10]. No class weighting, resampling or outlier removal. Dummy has no tuning. Save all candidate/fold scores and warnings; unresolved convergence warnings stop a run.
4. Fit transformations inside every training fold: numeric median if needed, categorical constant missing token, OneHotEncoder(handle_unknown='ignore'). LR alone uses numeric StandardScaler. Numeric-coded categories are categorical; Weekend is explicit 0/1. Target and ID are never inputs.
5. Compare full-feature learned models by **validation AP** at 0.50. Any AP within 0.005 of the highest is eligible; prefer LR, Tree, Forest in that order. This is a practical rule, not a significance test. The no-PageValues variants are prespecified secondary conditions, not feature-selection candidates.
6. For the selected full-feature model maximize validation purchase F1 over 0.05..0.95 in 0.01 steps, breaking ties toward the larger threshold. Persist `selection.json` before any test predictions. Models remain fitted on training only; no validation refit.
7. Report every frozen variant on test at 0.50 and the selected model at its chosen threshold. Test rankings cannot change selection. Persist row-level scores, exact thresholds, support, AP, ROC-AUC, accuracy, purchase precision/recall/F1 and confusion counts.
8. Define ProductRelated low/medium/high bins with training tertiles (**10 and 29**), reuse them for test error analysis. Deterministic FP/FN examples use the first four source row IDs per error category. Report supports and denominators; do not infer visitor motives.
9. Also report the prespecified duplicate sensitivity: 20 StratifiedGroupKFold groups, 14/3/3 fold allocation, approximate 70/15/15. Identical full predictor hashes never straddle its partitions. Clone/refit the primary selected model with frozen hyperparameters and reuse its threshold. No separate group-test tuning; do not relabel this as the original split.

The Decision Tree depth diagnostic uses fixed min_samples_leaf=5 and threshold=0.50 on train versus validation; it does not change model selection. All cross-partition repeated predictor vectors in this supplied file have zero observed purchases; overlap counts remain visible and the secondary analysis is still reported.

## Outputs and code

| Location | Purpose |
| --- | --- |
| `src/data.py`, `preprocessing.py` | Explicit schema/target handling, quality, deterministic rows, fold-local transforms |
| `src/train.py`, `evaluate.py`, `run_all.py` | Estimators/grids, pure selection/metrics, ordered offline workflow |
| `src/report.py`, `report_template.html` | Generate SVG/PNG figures, static narrative, JSON and source bundle from artifacts |
| `results/data_manifest.json`, `quality.json` | Byte identity, schema, actual quality, attribution and environment |
| `results/splits.json`, `group_splits.json` | Exact primary/secondary row IDs and class counts |
| `results/cv_results.csv`, `parameters.json` | Every candidate/fold score and fitted configuration |
| `results/selection.json`, `events.json` | Frozen validation-only decision and order relative to first test evaluation |
| `results/*_metrics.csv`, `*_predictions.csv` | Full precision scores, stable row order, model, features, partition, threshold and labels |
| `results/pagevalues_comparison.csv`, `group_sensitivity.csv` | Prespecified secondary experiments |
| `results/error_analysis.json` | Test errors, training-derived bin boundaries, subgroup denominators, deterministic examples |
| `results/figures/`, `eda_*.csv`, `tree_depth_comparison.csv` | Generated figures and accessible numeric evidence |
| `results/models/` | Trusted local joblib pipelines, ignored in Git and regenerated by run_all; never load untrusted joblib files |
| `notebooks/` | Executable data/reasoning and results notebooks importing shared source |
| `docs/` | Responsive static report, threshold lab, relative assets, JSON and downloadable source/evidence ZIP |
| `tests/` | Schema/split isolation, hand-checked metrics, frozen selection, exports and exact threshold parity |
| `.github/workflows/pages.yml` | GitHub Actions deployment of docs/ only; no Python on the web server |

After a text/style/config-only edit, regenerate the report without training using `python -m src.report`. The run identifier identifies the Python source at the full pipeline run, plus the data hash; rerun the pipeline after Python-source changes to refresh that provenance. Rebuild the source ZIP after README/notebook changes with the same report command.

## Website interaction and accessibility

The threshold lab loads the frozen selected model's actual **validation** truth/scores from `docs/data/threshold_validation.json`. Range 0.00..1.00, step 0.01; rule score >= threshold, including equality. It starts/resets at 0.50, marks the chosen 0.24, and instantly updates predicted buyers, TP/FP/TN/FN, precision, recall and F1. Keyboard arrow keys work. It neither retrains nor changes test results. Missing/invalid JSON produces an explanatory error and disabled controls, never fake scores.

All essential analysis is readable in static HTML. Charts have captions/alternative text and adjacent numeric evidence. Navigation, skip link, focus states, labelled controls and scrollable metric tables support keyboard use and small screens. The website uses relative URLs to work under a Pages repository subpath.

## Verification

Executed checks and any limitations are recorded in `results/verification.json` after final verification. The first scientific pass completed **11 pytest checks** and **six JS/Python threshold fixtures**, including 0, 0.50, 1, selected threshold, equality and no predicted positives. Five pandas deprecation warnings about the explicitly requested round-trip CSV parser are nonconvergence warnings; no estimator convergence warnings occurred. Notebook execution and browser checks are separate gates; consult the verification record for their actual outcome.

The automated checks verify raw identity/schema, complete stable disjoint IDs, training-only fitted means/vocabularies, unknown-category isolation, every required variant and bounded search, finite aligned scores, hand-checkable confusion metrics, selection/threshold replay from validation only, freeze/test event ordering, group isolation, and equality between website exports and result artifacts. These do not replace human understanding or external validity.

## Deploy with GitHub Pages

The specification targets GitHub Pages. The user selected the new public [ShopperSense-study repository](https://github.com/vivek-314/ShopperSense-study). The older public `vivek-314/ShopperSense` repository is preserved. The workflow publishes this version's `docs/` folder.

1. Create/select the intended GitHub repository, add these project files, and push its `main` branch. Do not commit `.venv`, `.tmp`, trusted-local model binaries, or browser QA caches.
2. Set its actual `repository_url` and `site_url` in `site_config.json`; run `python -m src.report`. Add actual contributor identities only when known.
3. In repository **Settings → Pages → Build and deployment**, select **GitHub Actions**. The included workflow uploads `docs/` and deploys on changes pushed to main or manual dispatch.
4. Observe the Pages workflow complete successfully. Open the real returned Pages URL; verify the JSON, figures, ZIP, repository link and keyboard threshold controls under the repository subpath. A localhost address is not the deployed URL.

## Boundaries, credits and sources

This is one historical source and one primary random split, with small bounded grids and descriptive subgroup/feature/duplicate sensitivities. There is no external or future-period validation, significance test, confidence interval, calibrated-probability claim, known business cost ratio, or intervention experiment. Session timing is unresolved; dropping PageValues does not solve that. No product quality/price preferences or motives are observable. The threshold is an illustrative F1 balance, not an established business optimum.

The requester supplied the PRD and CSV. **Codex authored and executed** the pipeline, tests, report, notebooks and documentation. Automated execution is accurately distinguished from human verification: no human review, personal contributor names, partner roles or grading criteria are invented. Authors should add their real contributions and verify methodology/submission links before submission.

Sources consulted on 16 November 2026:

- Sakar, C. & Kastro, Y. (2018). [Online Shoppers Purchasing Intention Dataset](https://doi.org/10.24432/C5F88Q). UCI, CC BY 4.0. [Dataset record, definitions and license](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- [Scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html). Metric terminology and average precision.
- [GitHub: What is GitHub Pages?](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages). Static repository website hosting.

All figures are generated locally from the actual supplied dataset and saved model outputs. No external visual assets, trackers, chart CDNs or fabricated model results are used.
