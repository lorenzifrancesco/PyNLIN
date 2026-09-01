# Superseded notes

These notes are kept for provenance and are **excluded from the Sphinx build**
(`exclude_patterns = ['stale']` in `../conf.py`). Nothing here should be quoted
as a current result. Retired 2026-09-01.

## Superseded by an active note

| Note | Superseded by | Reason |
|---|---|---|
| `pcfm_td_scientific_spec.md` | `../theory.md` §2–3 | Applies the workflow-level $16/9$ TD factor that the active path no longer uses; its only code reference, `analysis/pcfm/td.py`, has been deleted. |
| `noise_calculations.md` | `../theory.md` §2–3 | Same $16/9$ factor; built on `analysis/pcfm/workflow.py` and `analysis/pcfm/td.py`, both deleted with the `analysis/pcfm/` package. |
| `fwm_high_mu_oscillations.md` | `../fwm_single_tuple_scaling.md` | Diagnostic sub-topic of that note; should be folded in as an appendix. |

## No longer reproducible

| Note | Reason |
|---|---|
| `calculation_correctness_audit.md` | Audits revision `3397d18` *plus an uncommitted working tree* as of 2026-07-14. The state it describes cannot be recovered, and the code has since been reorganized. |
| `lorenzi_fast_cost_anatomy.md` | Its own header states that the timings, survivor counts, branch shares and seam tables predate the mask-aware tube selector and must not be quoted as current measurements. Regenerate before reinstating. |

## Planning and tracking, not documentation

`repository_consolidation_todo.md`, `publication_novelty.md`, `PROBLEMS.md`
(formerly `logical_analysis/PROBLEMS.md`). These remain useful as working
records but do not belong on the published site.

`overview.md` is here because its content was generic and four months stale;
`../index.md` currently has no orientation page.

## Outstanding salvage

Content that exists only in these notes and has no home in the active set:

1. **TD vs PCFM comparison.** `pcfm_td_scientific_spec.md` ("Matched TD vs PCFM
   Comparison in the Idealized Regime") and `noise_calculations.md` §4 are the
   only places this comparison is written down. `../theory.md` has no such
   section. Lift it, with the $16/9$ factor removed and the polarization
   convention stated.
2. **Orientation page.** `../index.md` needs a short replacement for
   `overview.md` describing the repository layout.
3. **Prefactor convention.** The $64/27$ vs $128/27$ single-polarization
   discrepancy recorded in `calculation_correctness_audit.md` (lines 363–366)
   is unresolved; it needs one audited equation in `../theory.md`.
