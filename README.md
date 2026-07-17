# Ceramic-Breeder Transport Research Code

This repository contains the source-code portion of a lithium-ceramic breeder
transport study.

The repository does not distribute manuscript files, supplementary material,
figures, literature datasets, generated research results, or submission
materials.

## Contents

- `code/`: numerical transport, inverse-analysis, sensitivity and supporting
  calculation scripts.
- `requirements.txt`: Python dependencies used by the source-code package.

## Basic Run

```bash
python3 -m pip install -r requirements.txt
python3 code/li_ceramic_bed_1d.py --material li2tio3 --out output/li2tio3
```

Generated output and figures are ignored by Git. Some supporting scripts need
locally generated inputs that are not distributed in this public repository.
