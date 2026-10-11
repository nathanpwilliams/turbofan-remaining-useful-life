# Data

The three FD001 files are committed here (~5.5 MB total), so the repository
clones and runs without a download step:

```
data/FD001/train_FD001.txt    20,631 rows, 100 engines run to failure
data/FD001/test_FD001.txt     13,096 rows, 100 engines truncated mid-life
data/FD001/RUL_FD001.txt      100 lines, true RUL at each test engine's last cycle
```

## Source

NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set, from the NASA
Prognostics Center of Excellence Data Repository. The full archive contains
four subsets (FD001–FD004); this project uses FD001 only.

- <https://data.nasa.gov/dataset/simulating-degradation-data-for-prognostic-algorithm-development>
- Mirror: <https://www.kaggle.com/datasets/behrad3d/nasa-cmaps>

## Citation

> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation
> Modeling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of the
> 1st International Conference on Prognostics and Health Management (PHM08)*,
> Denver, CO, 1–9.

> Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation
> Data Set*. NASA Prognostics Data Repository, NASA Ames Research Center,
> Moffett Field, CA.

## File layout

Whitespace-separated, no header, 26 columns in this order:

| # | Column | Meaning |
|---|--------|---------|
| 1 | `ID` | engine unit number |
| 2 | `Cycle` | time, in operational cycles |
| 3–5 | `Op S1`–`Op S3` | operational settings |
| 6–26 | `S1`–`S21` | sensor measurements |

Sensors cover temperatures, pressures and speeds through the engine: fan inlet
temperature (S1), LPC outlet temperature (S2), HPC outlet temperature (S3),
LPT outlet temperature (S4), fan inlet and bypass pressures (S5, S6), HPC
outlet pressure (S7), physical and core speeds (S8, S9), engine pressure ratio
(S10), HPC outlet static pressure (S11), fuel flow ratio (S12), corrected
speeds (S13, S14), bypass ratio (S15), burner fuel-air ratio (S16), bleed
enthalpy (S17), required fan speeds (S18, S19), and turbine cooling air flows
(S20, S21).

Separators are runs of spaces rather than single spaces, so the files need
`sep=r"\s+"` to read correctly.

## FD001 specifics

One operating condition, one fault mode (HPC degradation). In the training
set, engines fail between 128 and 362 cycles, mean 206.31.

Eight columns carry no signal in this subset and are dropped: `Op S3`, `S1`,
`S5`, `S10`, `S16`, `S18` and `S19` hold a single value throughout, and `S6`
holds one of two values on 98% of rows.
