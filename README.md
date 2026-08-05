# LyoXL digital twin — Track A pipeline

Everything below runs on your machine. Nothing in this pipeline
uploads or transmits your historian data anywhere.

## Files

- `ingest.py` — cleans a raw historian export, converts units, flags
  the Pirani/capacitance-manometer convergence point (physically
  observed end of primary drying).
- `geometry.py` — PCR tube frustum geometry: constant base area for
  heat entry vs. shrinking front area for mass transfer as the
  sublimation front recedes. Supports empirical fill-height
  calibration (recommended) or a theoretical frustum formula
  (fallback, less reliable at your fill volumes — see below).
- `pikal_model.py` — the physics model, single-segment fit, and
  **joint fit across multiple segments** (`fit_parameters_joint`),
  which is what you actually need given your setpoint structure.
- `run_pipeline.py` — **the script you run.** Takes your raw CSV,
  segments it by (Cycle, Step) inside your Primary Drying phase, and
  runs the joint fit. Prints and saves a text report — only that
  report is something you'd need to share back, never the CSV.

## Before running: calibrate the fill height

At 6–20 µL, product sits entirely in the tube's tip region, where the
real geometry (rounded transition, not a clean point) isn't published
by the manufacturer. Measure it directly:

1. Pipette 6 µL, 12 µL, and 20 µL of water into empty tubes, one at a
   time.
2. Measure meniscus height from the tip with calipers or a fine ruler.
3. Pass the three points to `run_pipeline.py` as `--calib`, e.g.
   `--calib 6:1.8 12:2.9 20:4.1` (µL:mm pairs).

If you skip `--calib`, the script falls back to `geometry.py`'s
theoretical frustum formula with placeholder tube dimensions — treat
any fit from that fallback as illustrative only, not trustworthy.

## Running it

```
python run_pipeline.py your_historian_export.csv \
    --primary-phase-code <the numeric code for Primary Drying> \
    --fill-volume-ul 12 \
    --calib 6:1.8 12:2.9 20:4.1
```

### Handling fully-immersed probes (ice-front temperature)

When thermocouples are completely immersed in the liquid with metal parts exposed to different sections of the freezing front, the **lowest probe reading** best approximates the ice-front temperature. Use:

```
python run_pipeline.py your_historian_export.csv \
    --primary-phase-code 6 \
    --fill-volume-ul 12 \
    --calib 6:1.8 12:2.9 20:4.1 \
    --use-min-temp
```

This uses `prod_min_k` (minimum of TP01-TP04) instead of `prod_avg_k`.

### Handling multi-step ramps (4-5 ramp steps)

The steady-state assumption breaks down during frequent setpoint changes. Enable transient mode:

```
python run_pipeline.py your_historian_export.csv \
    --primary-phase-code 6 \
    --fill-volume-ul 12 \
    --calib 6:1.8 12:2.9 20:4.1 \
    --use-transient
```

This adds a heat accumulation term to account for non-steady conditions during ramp steps.

### Using both features together

For fully-immersed probes with multi-step ramps:

```
python run_pipeline.py your_historian_export.csv \
    --primary-phase-code 6 \
    --fill-volume-ul 12 \
    --calib 6:1.8 12:2.9 20:4.1 \
    --use-min-temp \
    --use-transient
```

### Hybrid mode: quasi-steady for Hold steps, transient for Ramp steps

When your recipe has alternating Hold and Ramp steps (e.g., Step 1=Hold, Step 2=Ramp, Step 3=Hold, etc.), use hybrid mode. This automatically applies:
- **Quasi-steady state** for odd-numbered steps (1, 3, 5, ...) — assumed to be Hold steps
- **Transient mode** for even-numbered steps (2, 4, 6, ...) — assumed to be Ramp steps

```
python run_pipeline.py your_historian_export.csv \
    --primary-phase-code 6 \
    --fill-volume-ul 12 \
    --calib 6:1.8 12:2.9 20:4.1 \
    --use-min-temp \
    --hybrid-mode
```

This is ideal when you have 4-5 ramp steps interleaved with hold steps, as it uses the appropriate physics model for each step type without manual configuration. The `--hybrid-mode` flag overrides `--use-transient` if both are provided.

You know your `Phase` column's numeric coding (freezing / primary /
secondary / storage) — I don't, so pass whichever code corresponds to
Primary Drying. The script segments by `(Cycle, Step)` within that
phase; each Step is treated as a separate setpoint segment and fit
jointly.

Output: a printed and saved (`fit_report.txt`) summary — fitted
`{Kv, Rp0, A1, A2}`, overall RMS error in Kelvin, and a per-segment RMS
breakdown so you can spot a segment that doesn't belong (e.g. a
mislabeled phase transition) before it drags down the joint fit.

**That report is what's useful to share back with me** if you want a
second read on the numbers — not the CSV.

## What the joint fit does and doesn't fix

Fitting jointly across Steps with different shelf-temp/pressure
setpoints resolves the Kv/Rp0 identifiability problem from a single
constant-setpoint segment. It does **not** automatically identify
`A1`/`A2` (how resistance grows with dried-layer thickness) — those
only become identifiable once the dried layer has grown substantially,
regardless of setpoint diversity. If `A1`/`A2` come back visibly
unstable across different runs while `Kv`/`Rp0` stay consistent,
that's the data telling you something true, not a bug. Report `Kv`
and `Rp0` with more confidence than `A1`/`A2` unless your primary
drying segments run long enough to substantially deplete the ice.

## Why the SS plate + aluminium block stack isn't modeled separately

`Ts` is the shelf's own sensor reading; `Tp` is the product
thermocouple reading. Every conduction layer between them (shelf to
plate, plate to block, block to tube) is absorbed into the fitted
`Kv` — it's defined as the effective transfer coefficient between
those two *measured* points, not a textbook shelf heat-transfer
coefficient. An explicit resistance network for the plate/block stack
would need contact-resistance data you don't have and wouldn't
improve the fit or the eventual MPC. Revisit only if you specifically
need to diagnose where thermal resistance concentrates.

## What I still need from you

- The numeric `Phase` code for Primary Drying (and ideally the
  Freezing/Secondary/Storage codes too, for later phases of this
  project)
- Your three calibration points once measured
- The `fit_report.txt` output once you run it on a real batch
