# MP Autoresearch — Program

Autonomous research loop for mass photometry particle classification.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun17`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**:
   - `PROGRAM.md` — this file.
   - `prepare.py` — frozen constants, data generation, evaluation harness. **Do not modify.**
   - `test.py` — frozen evaluation runner. **Do not modify.**
   - `sim.py` — frozen simulation library. **Do not modify.**
   - `train.py` — the file you modify. Pipeline: peak picking → feature extraction → classification.
4. **Verify data exists**: check that `~/.cache/autoresearch-mp/train/` and `test/` contain
   `.npy` movies and `.pkl` ground-truth files. If not, ask the human to run `python prepare.py`.
5. **Initialize results.tsv**: create with just the header row. The baseline will be recorded
   after the first run.
6. **Confirm and go**.

## The scientific problem

Particles (proteins) diffuse in a supported lipid bilayer and are observed by iSCAT mass
photometry.  Each particle produces a small dark contrast spot shaped like a jinc (Airy)
PSF.  The contrast is proportional to the particle's mass.

**Task**: classify each detected peak event into one of 7 classes:
  - `0` noise  (ghost peak — no real particle)
  - `1` monomer  (55 kDa)
  - `2` dimer    (110 kDa)
  - `3` trimer   (165 kDa)
  - `4` tetramer (220 kDa)
  - `5` pentamer (275 kDa)
  - `6` hexamer  (330 kDa)

**The hard problem**: the monomer is near the noise floor.  Ghost peaks and monomers overlap
in contrast space.  The classifier must reject ghosts (low FDR) while accepting monomers
(high monomer recall).  Higher oligomers are easier.

## Metrics

The primary metrics, reported by `test.py`, are:

| Metric | Direction | Description |
|--------|-----------|-------------|
| `monomer_recall` | **maximize** | TP_monomer / (TP_monomer + FN_monomer) — key metric |
| `fdr` | **minimize** | ghost detections / all detections |
| `macro_f1` | maximize | unweighted average F1 across all 7 classes |
| `accuracy` | maximize | correct / total matched detections |

**A result is `keep` if it improves `monomer_recall` without worsening `fdr` by more than
0.02 absolute, OR if it improves `fdr` without worsening `monomer_recall` by more than 0.02.**
Everything else is `discard`.

## What you CAN do

Modify `train.py` — this is the **only** file you edit.  Everything in it is fair game:
- Peak detection method, thresholds, filter parameters
- Feature engineering (what to compute per detected peak)
- Classifier architecture and hyperparameters
- Data augmentation, class weighting, resampling
- Post-processing: per-class thresholds, Platt scaling, FDR-controlled rejection

## What you CANNOT do

- Modify `prepare.py`, `test.py`, or `sim.py`.
- Install packages not in `pyproject.toml`.
- Modify `results.tsv` except to append rows.
- Use the test data during training.

## Experimentation loop

LOOP FOREVER after initial setup:

1. Read git state (current branch/commit).
2. Modify `train.py` with one experimental idea.
3. `git commit -am "brief description"`
4. Run: `python train.py > run.log 2>&1`
5. If training succeeds, evaluate: `python test.py >> run.log 2>&1`
6. Read results: `grep "^monomer_recall\|^fdr\|^macro_f1\|^accuracy" run.log`
7. If the grep is empty, the run crashed. `tail -50 run.log` and attempt a fix.
   If the idea is fundamentally broken, log `crash` and skip it.
8. Log results to `results.tsv` (do NOT commit this file).
9. If result is `keep`: advance the branch (keep the commit).
10. If result is `discard`: `git reset --hard HEAD~1`.

**Timeout**: if a run exceeds 20 minutes, kill it and treat as crash.

**Crashes**: fix obvious bugs (typo, missing import) and retry once.
If the approach is broken, skip it.

**NEVER STOP**: once the loop starts, do not pause to ask the human.
Run indefinitely until manually stopped.

## Output format

`test.py` prints a summary ending with lines like:

```
monomer_recall: 0.412000
fdr:            0.183000
macro_f1:       0.651000
accuracy:       0.821000
```

Extract them with:
```
grep "^monomer_recall:\|^fdr:\|^macro_f1:\|^accuracy:" run.log
```

## Logging results

`results.tsv` has 6 tab-separated columns (no commas in descriptions):

```
commit  monomer_recall  fdr  macro_f1  status  description
```

- `commit`: 7-char git hash
- `monomer_recall`: float e.g. `0.412000`
- `fdr`: float e.g. `0.183000`
- `macro_f1`: float e.g. `0.651000`
- `status`: `keep`, `discard`, or `crash`
- `description`: brief description of the experiment

Example:
```
commit	monomer_recall	fdr	macro_f1	status	description
a1b2c3d	0.412000	0.183000	0.651000	keep	baseline RF with DoG features
b2c3d4e	0.441000	0.175000	0.668000	keep	add PSF width and asymmetry features
c3d4e5f	0.388000	0.191000	0.632000	discard	lower DoG threshold — too many ghosts
```

## Simplicity criterion

All else equal, simpler is better.  A marginal improvement that doubles code complexity
is not worth it.  Deleting code that achieves equal or better results is a win.

## Research directions to explore

Starting ideas (not exhaustive — you should invent your own):
- DoG threshold tuning (threshold_factor, sigma range)
- PSF shape features: fitted width, asymmetry, residual, Fourier-Bessel coefficients
- Temporal features: frame-to-frame contrast change, persistence
- Contrast normalization by local noise
- Class-weighted training to up-weight monomer
- Cascaded classifier: first noise-vs-real, then species classification
- Probability calibration + per-class decision thresholds optimized for FDR constraint
- Contrast-based prior: use MW calibration to constrain class probabilities
