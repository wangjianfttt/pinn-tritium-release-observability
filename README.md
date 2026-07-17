# Minimal Ceramic-Breeder Transport Code

This repository contains one basic, reduced one-dimensional transport example
for lithium-ceramic breeder beds.

The repository does not distribute manuscript files, supplementary material,
figures, literature datasets, generated research results, submission files, or
advanced inverse-analysis workflows.

## Contents

- `code/li_ceramic_bed_1d.py`: reduced one-dimensional ceramic-breeder
  transport example.
- `requirements.txt`: minimal Python dependencies.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 code/li_ceramic_bed_1d.py --material li2tio3 --out output/li2tio3
```

Generated output and figures are ignored by Git.
