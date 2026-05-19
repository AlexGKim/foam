# Foam Cosmology Reorganization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the paper by moving the interpretive prose from Section III.D into the Conclusions, while retaining Table I at the beginning of Section IV.

**Architecture:** Modifies `paper.tex`. The content of subsection "Why Temporal Second-Order Correlations are Less Sensitive to First-Order Effects" will be split:
1. The mathematical ratio and Table I will be moved to become a new motivating subsection at the start of Section IV.
2. The remaining paragraphs discussing observational limits and sensitivity (the prose) will be moved into Section VI (Conclusions).
3. The original Section III.D header will be removed to restore the mathematical flow of Section III.

**Tech Stack:** LaTeX

---

### Task 1: Move Table I and mathematics to Section IV

**Files:**
- Modify: `paper.tex`

- [ ] **Step 1: Extract and move Table I to Section IV**

Extract the subsection `\subsection{Why Temporal Second-Order Correlations are Less Sensitive to First-Order Effects}` (lines 467-499) and move it to immediately follow `\label{sec:siegert}` (around line 529).

Change the subsection title to:
`\subsection{Motivation: Sensitivity of Temporal Second-Order Correlations}`

### Task 2: Move Interpretive Prose to Conclusions

**Files:**
- Modify: `paper.tex`

- [ ] **Step 1: Extract and move prose to Conclusions**

Extract the remaining text from the original III.D (lines 501-524, which begins "For $\alpha=1/2$ at $D=1$~Gpc..." and ends "...positive-detection strategy.").

Move this text to the `Conclusions` section, inserting it immediately after the `\label{sec:conclusions}` command (around line 1012), before the existing paragraph about cosmological evolution.

### Task 3: Verify and Commit

- [ ] **Step 1: Recompile the paper**

Run: `pdflatex paper.tex`
Expected: Compile successfully without errors. Run a second time to resolve references.

- [ ] **Step 2: Commit**

```bash
git add paper.tex paper.pdf
git commit -m "docs: move section III.D prose to conclusions and table I to section IV"
```
