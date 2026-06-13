# Git Restructuring Complete: foam → Submodule-Based Monorepo

## What Changed

The foam project has been restructured from a single monolithic repository to a **submodule-based monorepo**, isolating g1 and g2 into independent git repositories while maintaining a shared bibliography and reference infrastructure.

### Before
```
foam/
  .git/ (single repo with all history)
  docs/
    aas_macros.sty
    references.bib
    refs.md
    g1/
    g2/
    ... other files
```

### After
```
foam/
  .git/ (main repo, orchestrates papers)
  .gitmodules
  g1/ (separate git repo via submodule)
  g2/ (separate git repo via submodule)
  aas_macros.sty (shared)
  references.bib (shared)
  refs.md (shared)
  refs/ (shared reference PDFs)
```

## Completed Steps

✅ **Phase 1**: Extracted g1 and g2 into separate git repositories  
✅ **Phase 2**: Moved shared assets (bibliography, references, styles) to foam root  
✅ **Phase 3**: Registered g1 and g2 as git submodules in foam  
✅ **Phase 4**: Created symlinks to shared aas_macros.sty in each paper  
✅ **Phase 5**: Verified g1 builds independently (`pdflatex g1.tex` works)  
✅ **Phase 6**: Created documentation (README.md)

## Next Steps

### 1. Create GitHub Repositories (One-Time Setup)

Create two empty repositories on GitHub:
- `AlexGKim/foam-g1` (public or private)
- `AlexGKim/foam-g2` (public or private)

### 2. Push Submodule Repos to GitHub

```bash
cd /Users/akim/Projects/foam/g1
git remote set-url origin git@github.com:AlexGKim/foam-g1.git
git push -u origin main

cd ../g2
git remote set-url origin git@github.com:AlexGKim/foam-g2.git
git push -u origin main

cd ..
```

### 3. Verify Submodule Configuration

```bash
cd /Users/akim/Projects/foam
git status  # Should show clean

# Test fresh clone of the setup
cd /tmp
git clone --recurse-submodules git@github.com:AlexGKim/foam.git foam-test
cd foam-test/g1
pdflatex g1.tex  # Verify it builds
```

### 4. Update Local Remotes (Optional Cleanup)

The g1 and g2 repos currently point to local temporary bare repos at `/tmp/foam-repos/`. After pushing to GitHub, you can safely delete those:

```bash
rm -rf /tmp/foam-repos
```

The local clones of g1 and g2 in foam/ will work fine even after deletion as long as they have GitHub configured as origin.

## Working with the New Structure

### Typical Workflow for g1

```bash
cd /Users/akim/Projects/foam/g1
# Make edits to g1.tex, run LaTeX, add figures, etc.
git add .
git commit -m "Your commit message"
git push origin main

# Propagate to foam monorepo
cd ..
git add g1
git commit -m "Update g1 submodule to latest"
git push origin main
```

### Typical Workflow for g2

Same as g1, but with `cd g2` instead of `cd g1`.

### Sharing Bibliography

Edit directly in foam/:

```bash
cd /Users/akim/Projects/foam
# Edit references.bib
git add references.bib
git commit -m "Add new reference"
git push origin main

# Both g1 and g2 immediately see the updated bibliography
```

## Key Advantages of This Structure

1. **Independent git history**: g1 and g2 have separate commit histories — you can fetch/view their work independently
2. **Parallel development**: Different people can work on g1 and g2 without stepping on each other's toes
3. **Shared infrastructure**: Bibliography, reference PDFs, and styles are centralized and versioned
4. **Cleaner working directory**: When you `cd g1`, you're in a focused paper repo, not wading through unrelated files
5. **Future-proof**: Easy to add g3, g4, etc. by creating new submodules

## Troubleshooting

### Q: I cloned foam without submodules. How do I get g1 and g2?

```bash
cd /Users/akim/Projects/foam
git submodule update --init --recursive
```

### Q: g1 or g2 shows an outdated commit in git status. How do I update?

```bash
cd /Users/akim/Projects/foam/g1
git pull origin main

cd ..
git add g1
git commit -m "Update g1 to latest"
git push origin main
```

### Q: I want to check out an old version of g1. Can I do that?

Yes! Each paper is a full git repo with its complete history. You can rewind, branch, tag, etc. within `foam/g1` independently.

### Q: How do I compare changes between g1 and g2?

They have independent histories. Use standard git commands within each directory:

```bash
cd /Users/akim/Projects/foam/g1
git log --all --graph --oneline

cd ../g2
git log --all --graph --oneline
```

## File Locations Summary

| Asset | Location | Scope |
|-------|----------|-------|
| g1 paper | `foam/g1/g1.tex` | Paper-specific |
| g2 paper | `foam/g2/paper.tex` | Paper-specific |
| Bibliography | `foam/references.bib` | Shared (both papers) |
| LaTeX macros | `foam/aas_macros.sty` | Shared (symlinked in each paper) |
| Reference PDFs | `foam/refs/` | Shared |
| g1 figures | `foam/g1/` (*.pdf, *.png) | Paper-specific |
| g2 figures | `foam/g2/` (*.pdf, *.png) | Paper-specific |
| Python scripts | `foam/g1/`, `foam/g2/` | Paper-specific |

---

**Status**: Restructuring complete. Awaiting push to GitHub (see Step 1-2 above).
