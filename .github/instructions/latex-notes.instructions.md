---
description: "Use when writing, editing, or reviewing LaTeX research notes, derivations, equations, or technical documentation. Covers project LaTeX conventions, common macros, document structure, and compilation."
applyTo: ["notes_all/**/*.tex", "papers/**/*.tex"]
---

# LaTeX Research Notes Conventions

## Project Structure
Notes are organized by topic under `notes_all/`:
```
notes_all/
  notes_control/              # CT, LQR, contraction theory
    compile_notes.tex         # Master document (inputs sub-files)
    notes_computed_torque.tex
    notes_lqr_both_cart_manip.tex
    notes_lqr_cart_manip_following.tex
    notes_contraction_theory.tex
  notes_cup/                  # Cart-pendulum EOM derivations
  notes_cup_manipulator_tendon/   # Tendon IK, Jacobians, pulley routing
    hybrid_jacobian_derivation.tex
    cup_manipulator_modes_and_pulley_routing.tex
    computed_torque_controller_dynamics.tex
  notes_ball_plate/           # Ball-plate system
  notes_cart_pendulum_manipulator/  # Combined system
```

## Standard Preamble Packages
```latex
\usepackage{amsmath, amssymb, bm}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{listings}  % For Python code blocks
\usepackage{cleveref}
\usepackage{mdframed}  % Blue-tinted definition/remark boxes
```

## Project-Specific Macros
Use bold italic vectors (consistent across all notes):
```latex
\newcommand{\bq}{\bm{q}}       % joint vector
\newcommand{\bp}{\bm{p}}       % position vector
\newcommand{\bu}{\bm{u}}       % control input
\newcommand{\btau}{\bm{\tau}}   % torque vector
\newcommand{\bF}{\bm{F}}       % force vector
\newcommand{\bJ}{\bm{J}}       % Jacobian
\newcommand{\bM}{\bm{M}}       % mass matrix
\newcommand{\bC}{\bm{C}}       % Coriolis matrix
\newcommand{\bG}{\bm{G}}       % gravity vector
\newcommand{\bl}{\bm{l}}       % cable length vector
\newcommand{\bomega}{\bm{\omega}} % angular velocity
```

## Document Style
- **Article class**: `\documentclass[11pt, a4paper]{article}` or `[12pt]{article}`
- **Margins**: `\geometry{margin=2.2cm}` to `\geometry{margin=2.5cm}`
- **Definition boxes**: Use `mdframed` with blue tint: `backgroundcolor=blue!5, linecolor=blue!40`
- **Code listings**: Python with `\lstset{language=Python, basicstyle=\ttfamily\small}`
- **Title format**: `\title{\textbf{Topic}\\  \large \texttt{script\_name.py}}`
- **Date**: month + year format (e.g., `\date{March 2026}`)

## Compilation
Individual notes are self-contained (`\documentclass` + `\begin{document}`).
Some topics have a master file (e.g., `compile_notes.tex`) that `\input{}`s sub-files.

```bash
cd notes_all/notes_control
pdflatex compile_notes.tex
# or for a single note:
pdflatex notes_computed_torque.tex
```

## Equations
Use `align` for multi-line derivations, `bmatrix` for matrices:
```latex
\begin{align}
  \btau &= \bM(\bq)\, \ddot{\bq}_{\text{des}} + \bC(\bq, \dot{\bq})\,\dot{\bq} + \bG(\bq) \\
  \ddot{\bq}_{\text{des}} &= \ddot{\bq}_{\text{ref}} + K_d\,(\dot{\bq}_{\text{ref}} - \dot{\bq}) + K_p\,(\bq_{\text{ref}} - \bq)
\end{align}
```

## Conventions
- Robot parameters: always include numeric values with units (e.g., `L_1 = 342.47\;\text{mm}`)
- Reference scripts by `\texttt{script\_name.py}` in titles/text
- Figures: place PNGs in the same directory as the `.tex` file
- When deriving EOM, clearly state the generalized coordinates and their physical meaning
- Use `\toprule`, `\midrule`, `\bottomrule` (from booktabs) for tables
