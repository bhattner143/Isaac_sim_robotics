---
name: write-latex-notes
description: "Write or extend LaTeX research notes for this project. Use when: documenting new derivations, adding EOM, control theory proofs, experiment results, or creating new notes files in notes_all/."
argument-hint: "Describe the topic (e.g., 'derive SEA dynamics for tendon manipulator' or 'document LQR results')"
---

# Write LaTeX Research Notes

## When to Use
- Documenting new mathematical derivations (EOM, Jacobians, control laws)
- Creating experiment result summaries with equations
- Adding to existing notes in `notes_all/`

## Notes Organization
```
notes_all/
  notes_control/              → CT, LQR, contraction theory, OFC
  notes_cup/                  → Cart-pendulum EOM (1D, 2D, spherical)
  notes_cup_manipulator_tendon/ → Tendon kinematics, Jacobians, pulley routing
  notes_cart_pendulum_manipulator/ → Combined cart-pendulum-manipulator
  notes_ball_plate/           → Ball-plate balancing
```

## New Note Template

```latex
% =============================================================================
%  <Title>
%  <Optional one-line description>
% =============================================================================

\documentclass[11pt, a4paper]{article}

\usepackage{amsmath, amssymb, bm}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{cleveref}
\usepackage[backgroundcolor=blue!5, linecolor=blue!40, linewidth=1pt,
            innertopmargin=6pt, innerbottommargin=6pt,
            innerleftmargin=7pt, innerrightmargin=7pt]{mdframed}

\geometry{margin=2.2cm}

% ── Standard project macros ──────────────────────────────────────────────────
\newcommand{\bq}{\bm{q}}
\newcommand{\bp}{\bm{p}}
\newcommand{\bu}{\bm{u}}
\newcommand{\btau}{\bm{\tau}}
\newcommand{\bF}{\bm{F}}
\newcommand{\bJ}{\bm{J}}
\newcommand{\bM}{\bm{M}}
\newcommand{\bC}{\bm{C}}
\newcommand{\bG}{\bm{G}}

\title{\textbf{<Title>}\\
       \large \texttt{<related\_script.py>}}
\date{<Month Year>}
\author{Isaac-sim robotics notes}

\begin{document}
\maketitle

\section{Overview}
% Brief description of the system/problem

\section{Derivation}
% Main mathematical content

\end{document}
```

## Style Rules
- Use `\bm{x}` for vector quantities (bold italic), not `\mathbf{x}`
- Always state generalized coordinates explicitly: "Let $\bq = [q_1, q_2]^\top$"
- Include numeric values with units: `$L_1 = 342.47\;\text{mm}$`
- Reference Python scripts in titles: `\texttt{script\_name.py}`
- Use `align` environment for multi-line derivations
- Use `mdframed` boxes for definitions, remarks, and key results
- Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) for tables

## Adding to Existing Topic
If a `compile_notes.tex` master file exists, add your note as a new `\input{}`:
```latex
\input{notes_computed_torque.tex}
\newpage
\input{notes_lqr_both_cart_manip.tex}
\newpage
\input{your_new_note.tex}       % ← add here
```

For standalone compilation, keep `\documentclass` in your file.
For inclusion via `\input{}`, wrap content in sections only (no `\documentclass`).

## Compilation
```bash
cd notes_all/<topic_folder>
pdflatex <filename>.tex
# For bibliography: pdflatex → bibtex → pdflatex × 2
```
