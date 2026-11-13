# Error handlers

Edit `inp.txt` to select `Heston` or `Bates` and enable the P-space and Q-space
analyses. Then run, from the repository root:

```bash
python3 error_handlers/run_error_handlers.py
```

The launcher compiles the selected C++ P-space analyzer into the system
temporary directory and runs the corresponding Python Q-space analyzer.

## Paths

Heston reads:

- `Data/Real_Stock/S_path.csv`
- `Output/Baseline_Stock/`
- `Output/Real_Stock/`

and writes under both `Results/Real_Stock/{P_errors,Q_errors}/` and
`Results/Real_Stock_LMS/{P_errors,Q_errors}/`.

Bates reads:

- `Data/Real_Crypto/S_path.csv`
- `Output/Baseline_Crypto/`
- `Output/Real_Crypto/`

and writes under `Results/Real_Crypto/{P_errors,Q_errors}/`.

The Bates analyzers preserve date-hour alignment. They can be compiled and
syntax-checked before `Output/Baseline_Crypto/` exists, but running them
requires the completed Bates baseline files.
