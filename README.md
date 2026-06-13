# foam — Spacetime Foam Spectroscopy Papers

This repository contains two papers on spectroscopic probes of quantum-gravitational spacetime foam:
- **g1**: "Spectroscopy as a Probe of Spacetime Foam" (arrival-time jitter via GRB and CMB)
- **g2**: "Siegert Relation Violation as a Probe of Spacetime Foam" (temporal coherence effects)

## Repository Structure

```
foam/                          # Main monorepo (shared infrastructure)
├── .gitmodules              # Submodule definitions
├── g1/                       # Paper 1 (separate git repo, linked as submodule)
├── g2/                       # Paper 2 (separate git repo, linked as submodule)
├── references.bib            # Shared bibliography (used by both papers)
├── aas_macros.sty            # Shared LaTeX macros
├── refs/                      # Shared reference PDFs
├── refs.md                    # Reference catalog
└── bibliography.md           # Bibliography documentation
```

## Setup

### Clone with Submodules

To clone foam with both papers:

```bash
git clone --recurse-submodules git@github.com:AlexGKim/foam.git
cd foam
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Working with Individual Papers

### Working on Paper g1

```bash
cd foam/g1
# Edit files, build, commit
git add .
git commit -m "Your message"
git push origin main
```

Then update the submodule reference in foam:

```bash
cd foam
git add g1
git commit -m "Update g1 submodule to latest"
git push origin main
```

### Building Papers Locally

Each paper builds independently with its own LaTeX setup:

```bash
cd foam/g1
pdflatex g1.tex
bibtex g1
pdflatex g1.tex
pdflatex g1.tex
```

The bibliography at `../references.bib` is used automatically.

## Shared Assets

- **references.bib**: Master bibliography used by both papers
- **aas_macros.sty**: AAS LaTeX macros (symlinked from each paper's directory)
- **refs/**: PDF copies of reference papers
- **refs.md**: Annotated index of references organized by topic

### Updating Bibliography

Edit `foam/references.bib` directly. Changes are immediately visible to both papers.

### Adding New Reference Papers

1. Add the PDF to `foam/refs/` with a descriptive name
2. Update `foam/refs.md` with the new entry
3. Add the citation to `foam/references.bib`

## Publishing Papers to GitHub

The papers g1 and g2 are maintained as separate repositories on GitHub:

- [AlexGKim/foam-g1](https://github.com/AlexGKim/foam-g1)
- [AlexGKim/foam-g2](https://github.com/AlexGKim/foam-g2)

These are linked to the foam monorepo as submodules. When pushing:

1. Push changes from within each paper's directory:
   ```bash
   cd foam/g1 && git push origin main
   cd foam/g2 && git push origin main
   ```

2. Update the foam monorepo to record the new submodule commits:
   ```bash
   cd foam && git add g1 g2 && git commit -m "Update submodule references"
   ```

## File Symlinks

Each paper has a symlink to the shared aas_macros.sty:

```bash
g1/aas_macros.sty -> ../aas_macros.sty
g2/aas_macros.sty -> ../aas_macros.sty
```

These allow `\usepackage{aas_macros}` to work in each paper's LaTeX files without duplication.

## Troubleshooting

### Submodule Not Cloned

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Stale Submodule References

If the submodule pointer is behind:

```bash
cd foam/g1
git fetch origin
git checkout main  # or git pull origin main
cd ..
git add g1
git commit -m "Update g1 to latest"
```

## Future: Adding Paper g3 and Beyond

When adding a new paper:

1. Create a new repository: `foam-g3.git`
2. Add to foam as a submodule:
   ```bash
   git submodule add git@github.com:AlexGKim/foam-g3.git g3
   ```
3. Create a symlink to shared assets:
   ```bash
   ln -s ../aas_macros.sty g3/
   ln -s ../references.bib g3/
   ```
4. Update `.gitmodules` in the foam commit

---

For more details on project organization, see [CLAUDE.md](./CLAUDE.md).
