# Tetramer Structural Comparison

A Python Shiny application for comparing sequence-resolved DNA structural
coordinates across X-ray crystallography, molecular dynamics (MD) simulation,
and optionally coarse-grained DNA (cgDNA) datasets.

---

## Scientific purpose

DNA structural parameters — such as Twist, Roll, Tilt, Slide, Propeller, or
Buckle — vary depending on the nucleotide sequence context.  The finest
practically useful resolution is the DNA **tetramer**: the 4-base subsequence
centred on the dinucleotide step of interest.  All 4^4 = 256 possible tetramers
can be assigned coordinates from different experimental or computational sources.

This application enables side-by-side comparison of those 256 tetramer
coordinates from up to three datasets, using:

- Pearson correlation to assess linear agreement
- Cosine similarity to measure vector orientation
- Regression analysis, filtering by central-step class, heatmaps, and more

---

## Input format

Each dataset must be a plain CSV file with the following structure:

| Tetramer | Coord_1 | Coord_2 | … | Coord_18 |
|----------|---------|---------|---|----------|
| AAAA     | 0.123   | 35.4    | … | 3.38     |
| AAAC     | 0.087   | 34.8    | … | 3.35     |
| …        | …       | …       | … | …        |
| TTTT     | −0.456  | 36.1    | … | 3.41     |

Requirements:
- The **first column must be named `Tetramer`**.
- Every tetramer must be exactly 4 characters from the set A, C, G, T.
- The recommended number of tetramers is 256 (all possible 4-mers).
- Remaining columns must be numeric. Non-numeric columns are dropped automatically.
- Column names beyond `Tetramer` are detected automatically — there is no
  requirement for any specific naming convention, though names containing
  `intra`/`inter` (or common parameter keywords such as `roll`, `twist`,
  `buckle`) are automatically labelled in the UI.

---

## Central-step classification

The **central dinucleotide** of a tetramer is formed by positions 2 and 3
(1-indexed).  For example, in `ACGT`, the central dinucleotide is `CG`.

Each base is classified as a purine (A or G) or a pyrimidine (C or T),
giving four **step classes**:

| Class | Meaning              | Example central steps |
|-------|----------------------|-----------------------|
| PP    | Purine / Purine      | AA, AG, GA, GG        |
| PY    | Purine / Pyrimidine  | AC, AT, GC, GT        |
| YP    | Pyrimidine / Purine  | CA, CG, TA, TG        |
| YY    | Pyrimidine / Pyrimidine | CC, CT, TC, TT     |

Each class therefore contains 4 × 4 × 4 = 64 tetramers (4 choices of
flanking base-pair × 4 choices on each side × 16 dinucleotides divided into
four equal groups of four).

---

## Metrics

### Pearson correlation (r)

Pearson r measures **linear association** after centering both vectors:

```
r = cov(x, y) / (σ_x · σ_y)
```

It is insensitive to additive offsets or multiplicative scaling, making it
appropriate for comparing structural parameter trends across datasets that
may have systematic biases.

### Cosine similarity

Cosine similarity measures the **angle** between the two raw (un-centred)
vectors:

```
cos(θ) = (x · y) / (||x|| · ||y||)
```

Unlike Pearson r, cosine similarity is sensitive to the mean level of both
vectors.  A high cosine with a low Pearson r indicates good shape agreement
but a mean-level difference (systematic bias).

**These metrics are not interchangeable.**  Inspecting both simultaneously
(see the *Similarity analysis* tab) provides a more complete picture.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running locally

```bash
shiny run --reload app.py
```

Then open `http://localhost:8000` in a browser.

---

## Example datasets

Three pre-generated synthetic datasets are included in `data/`:

| File             | Description                                              |
|------------------|----------------------------------------------------------|
| example_xray.csv | "Reference" dataset with small Gaussian noise           |
| example_md.csv   | Slightly scaled + biased + moderate noise                |
| example_cgdna.csv| Different scale + bias + noise level                     |

All three contain all 256 tetramers × 18 structural coordinates with
physically plausible parameter ranges (Twist ~35°, Rise ~3.36 Å, etc.).
They are generated with a fixed random seed (42) for full reproducibility.

To regenerate:
```bash
python generate_data.py
```

Expected Pearson r range (X-ray vs MD): approximately 0.67–0.97 across
the 18 coordinates, providing meaningful variation for exploring the UI.

---

## Application tabs

| Tab                  | Content                                                    |
|----------------------|------------------------------------------------------------|
| Scatter comparison   | Main X vs Y scatter plot for one coordinate                |
| Coordinate overview  | Pearson r and cosine for all 18 coordinates (bar chart + table) |
| Similarity analysis  | Pearson r vs cosine scatter — one point per coordinate     |
| Correlation heatmap  | Pearson r or cosine across coordinates × step classes      |
| Difference plot      | Bar chart of Y − X per tetramer                            |

---

## Export functionality

| Format | Contents                                   |
|--------|--------------------------------------------|
| PNG    | Current scatter plot at 2× pixel density  |
| PDF    | Vector PDF of the current scatter plot     |
| SVG    | Vector SVG of the current scatter plot     |
| CSV    | Currently filtered comparison data table  |

PNG/PDF/SVG export uses [Kaleido](https://github.com/plotly/Kaleido) for
high-fidelity static rendering of Plotly figures.

---

## Interpretation guide

1. **Start** on the *Scatter comparison* tab with your X-ray and MD datasets.
2. Use the **Coordinate** dropdown to inspect each of the 18 parameters.
3. Use **Central-step filter** (PP/PY/YP/YY) to check whether agreement is
   step-class-dependent.
4. Switch to **Coordinate overview** for a global r profile across all 18 coords.
5. Use **Similarity analysis** to identify coordinates where Pearson r and
   cosine diverge (indicative of systematic offsets).
6. Use **Correlation heatmap** to see which step classes drive agreement or
   disagreement for each parameter.
7. Use **Difference plot** to identify individual tetramers with unusually
   large discrepancies.

---

## Project structure

```
tetramer_comparison/
├── app.py              Main Shiny application
├── requirements.txt    Python dependencies
├── README.md           This file
├── generate_data.py    Script to regenerate synthetic datasets
├── data/
│   ├── example_xray.csv
│   ├── example_md.csv
│   └── example_cgdna.csv
└── utils/
    ├── __init__.py
    └── metrics.py      Pure-Python analysis functions (no UI)
```

---

## Dependencies

- **shiny / shinywidgets** — reactive UI framework for Python
- **plotly** — interactive publication-quality figures
- **pandas / numpy** — data handling
- **scipy** — Pearson correlation
- **kaleido** — static image export from Plotly
