# Meeting Notes — Progress Update 2: Exosuit
**Date:** 20 May 2026  
**Duration:** ~45 minutes  
**Participants:**
- **Dipankar Bhattacharya** — presenter (PhD student, Imperial)
- **Thrishantha Nanayakkara** — PI / supervisor
- **Dagmar Sternad** — external collaborator (Northeastern University)
- **Salah Bazzi** — collaborator

---

## 1. Agenda Overview

Dipankar gave a progress update on:
1. Simulation platform (PyDrake + Isaac Sim, digital twin pipeline)
2. Hardware design (2-DOF cable-driven manipulator + exosuit)
3. Control architecture (CT + SEA + exo co-contraction)
4. Simulation results (exo OFF vs ON under sinusoidal disturbance)
5. Parameter learning (probe → identify → adapt)
6. Contraction theory (CCM vs LQR study)
7. Research gaps and future directions

---

## 2. Simulation Platform

### What was presented
- **CAD → URDF pipeline**: Onshape → URDF → OBJ/USD for both PyDrake and Isaac Sim
- The same CAD models are used for simulation AND manufactured hardware — creating a **digital twin**
- Two parallel simulators are maintained:
  - **PyDrake** (Meshcat): analytical dynamics, LQR/CCM, fast iteration
  - **Isaac Sim** (Isaac Lab): GPU-accelerated, for future ML/RL integration
- Reasoning for dual-simulator approach: "If we need to do AI/ML moving forward, Isaac Sim is much better for that"

### Questions raised
- **Dagmar**: Why keep both simulators running in parallel?  
  **Answer (Dipankar)**: Isaac Sim needed for RL/ML down the line; PyDrake better for analytical control design now.

---

## 3. Hardware Description (shown via video)

### Robot description
- **2-DOF planar manipulator** (shoulder joint = link1_base, elbow joint = link2_link1)
- **Shoulder**: directly actuated — CT torque applied via drive motor (AK60-6, 9 Nm peak)
- **Elbow**: cable-driven via Series Elastic Actuator (SEA)
  - Drive motor wraps cables around a pulley → cable tension controls elbow torque
  - Spring in series between motor and cable = Series Elastic Actuator
  - Spring introduces compliance and lag (important for biological analogy)
  - Encoders at both motor and joint to measure spring extension → spring force
- **Exosuit** (Method B): two additional cables route over an elbow pulley (radius 30 mm) on link2
  - Two exo motors, antagonistic — one cable goes slack when other is taut
  - When both cables pre-tensioned (co-contraction), net torque from exo ≈ 0 but **effective stiffness increases**

### Key clarification (Dagmar's question)
> *"The motor on its own can provide these torques, but the spring is not controllable in the sense that the co-contraction is a given... the exo only modulates the stiffness?"*

**Answer (Dipankar + Thrishantha)**: Correct. The exo's primary effect is **stiffness modulation**, not direct torque assistance. When activated, it applies a passive antagonistic co-contraction that increases the effective joint impedance at the elbow:
$$K_{\text{eff}} = 2 k_{\text{exo}} r_{\text{exo}}^2$$

### Dagmar's clarification on cable design
> *"How common is that cable design for a two-jointed arm?"*

**Thrishantha**: "We made this to join them. I haven't seen that before, no." — This is a **novel mechanical contribution**.

---

## 4. Control Architecture

### Layered control (as presented)
```
Trajectory (circle/rect/figure8/line)
    ↓
Computed Torque Controller (Kp=400, Kd=80)
    ↓ τ_des [shoulder, elbow]
Shoulder: direct torque → τ_1
Elbow: SEA cable → spring extension δ → cable force F → τ_2
    ↓
Exo (transparent OR co-contraction mode):
    τ_exo = K_eff · (q2 - q2_anchor)  [when activated]
```

- Exo is **transparent** (zero stiffness, zero torque contribution) when not active
- Exo **activates** at a trigger time (e.g., t=4 s) and applies passive antagonistic stiffness
- Key parameter: **Δθ = 0.5 rad** (pre-tension angle difference between left and right exo cable motor)

### Salah's question
> *"Is there a torque also to the elbow?"*

**Answer (Dipankar)**: Yes — CT computes desired torques for BOTH shoulder and elbow, but the elbow torque is mediated through cable dynamics AND exo spring torque, not directly.

---

## 5. Simulation Results Shown

### Exo OFF (baseline)
- Manipulator tracking a circle trajectory
- Sinusoidal disturbance injected at t=8 s (τ_ext = 2.0·sin(2π·2Hz·t) Nm for 1.5 s)
- Result: large spikes in tracking error and joint torque during disturbance window

### Exo ON (co-contraction activated at t=4 s)
- Same trajectory + disturbance
- Result: **tracking error spikes significantly reduced**, elbow becomes stiffer
- Metrics logged: peak EE distance, RMS EE distance, peak q2 error, RMS q2 error, peak τ_exo, peak τ_drive

### Confusion about plot interpretation (live during meeting)
**Dagmar**: *"If I look at the units, Newton meter — the disturbance amplitude is actually higher when XO is on... the title of the figures is the same in both plots which confused me."*

**Clarification (Salah)**: The left figure shows exo OFF (τ_exo ≈ 0), right shows exo ON. The larger torque in the right plot is the **exo torque itself** (τ_exo ≈ 2 Nm), not the disturbance. The disturbance torque is the same in both cases. The drive torque (motor) *decreases* when the exo is ON because the exo is doing part of the work.

**Action item**: Improve plot titles/labels to distinguish τ_exo from τ_disturbance more clearly.

---

## 6. Parameter Learning (Probe → Identify → Adapt)

### What was done
- **Scenario**: Manipulator's elbow joint has unknown impedance parameters (K_eff, B_eff initially set wrong/zero)
- A **probe signal** (sinusoidal excitation) is injected at the elbow
- From the joint trajectory response (q2 vs time), run **least-squares regression** to identify K_eff and B_eff
- Once identified, update the CT controller gains
- Result: tracking improves progressively as parameters converge

**Ground truth**: K_eff = 36.48 Nm/rad (from exo hardware). Identified value converged to ~45 Nm/rad (close but not exact — sample size limited to one simulation run).

### Questions raised

**Dagmar**: *"How does it learn?"*  
**Answer**: Linear regression / least-squares on sampled (q2, q̇2, q̈2) trajectories.

**Salah**: *"How dependent is that on the trajectory? Do you need a sinusoidal input for a considerable time?"*  
**Answer (Dipankar)**: Only tested with one simulation sinusoidal probe. Generality to other trajectories not yet verified.

**Thrishantha**: *"If it is very linear... it's a linear regression, right? But if it has complex oscillations, maybe it won't work."*  
**Agreed**: Linear regression is limited to simple/linear regimes. More complex scenarios would need nonlinear learning.

**Salah**: *"What about learning when the robot is performing a task with an object that's adding its own dynamics? Are those parameters even learnable in the same framework?"*  
**Thrishantha's answer**: *"Layered learning"* — linear regression captures some percentage; a neural network on top handles the complex residual, potentially reinforcement-learning based.

### Key research hypothesis (Thrishantha)
> *"Our hypothesis is the brain will learn better with the exosuit. Without the exosuit, it's very random and the brain doesn't see a pattern. So the exo makes it a learnable problem — can the existence of the exo just make it simple enough for the brain to learn?"*

This is a core testable hypothesis for the paper.

### Gap identified
> **Dipankar**: *"What I've learned here is for the manipulator. It's our brain that needs to learn those impedance parameters. So how we will do it for real humans — that's one of the doubts I have."*

---

## 7. Contraction Theory (CCM)

### What was presented
- Implemented the **Manchester & Slotine (2017) CCM** formulation in simulation
- Tested three controllers: **LQR**, **approximate CCM (M=P constant metric)**, **full state-dependent CCM**
- Key result from script_11a (polynomial system): LQR diverges for large initial conditions; CCM stabilizes globally

### Gap acknowledged
> **Dipankar**: *"I still couldn't really grasp how to use the CCM contraction-based controller for the manipulator+exo project. The question is: why not use the contraction-based controller directly? That's a question I still need to think about."*

**Salah's philosophical answer**:
> *"If you go back to when you were learning K and B via regression — you're finding gains that work for one specific scenario. But when you're interacting with uncertain environments... a Control Contraction Metric would guarantee that given some assumptions about the range of K's and B's, these are going to guarantee contraction/convergence/attenuation, no matter what the disturbance magnitude."*

In other words: CCM provides **global certified robustness** — the gain set works for a whole family of disturbances, not just the one you identified for.

**Thrishantha's neuroscience analogy**:
> The brain likely prefers a "learnable" representation. If it goes into full state-space, it becomes complex. The Purkinje cell / red nucleus system in the cerebellum uses error signals from climbing fibers to update simple gains — more like CCM (certified over a region) than full nonlinear optimal control.

---

## 8. Research Gaps Identified

| Gap | Description |
|-----|-------------|
| **CCM ↔ Exo connection** | How to formally incorporate exo stiffness K_eff into CCM synthesis (scripts 14e/14f partially address this) |
| **Linear regression limits** | Works only for simple sinusoidal probing; fails for complex task dynamics |
| **Human-robot experiment** | No human data yet — all work is purely simulation/robot |
| **Plot clarity** | τ_exo vs τ_disturbance confusion in current plots |
| **Sim-to-real** | Controllers designed in simulation, not yet validated on hardware |
| **Learning generalisation** | Parameters learned for sine probe may not transfer to circle/figure-8/task trajectories |

---

## 9. Future Experiment & Paper Strategy

### Dagmar's framing question
> *"For one chunk of achievement for a paper — the novel mechanical design (2-joint cable + exo), the XO acting on elbow, and then how/when does the comparison to a human become necessary?"*

### Agreed next steps

**Stage 1 (Near-term, ~2-4 weeks)**:
- Finish hardware assembly (placement student helping, ETA ~2 weeks)
- Calibrate motors and encoders on actual manipulator + exo
- Implement existing CT + SEA + exo controllers on real hardware
- Verify sim ↔ hardware consistency

**Stage 2 (Robot-only experiments)**:
- Apply sinusoidal end-effector perturbations on hardware
- Record encoder data (reaching trajectory response)
- Compare exo OFF vs ON metrics on real hardware
- Potentially add wireless EMG to monitor muscle co-contraction patterns

**Stage 3 (Human experiments, requires ethics approval)**:
- Human holds end effector of the robot (robot applies perturbation)
- Record arm reaching trajectories + exo ON/OFF conditions
- Compare human EMG patterns to model predictions
- **Thrishantha**: "We can use a sling on the ceiling (elbow support) to isolate elbow dynamics — same paradigm as previous perturbation experiments"

### Paper target (Dipankar's goal)
- **TRO or IROS** — targeting before contract end (October next year)
- **Salah's suggestion**: First paper could be the **mechanical design + novel cable routing + initial hardware results** — that alone is publishable as a novel mechanism that emulates antagonistic muscle co-contraction at the elbow

### Dagmar's recommendation
> *"The contraction theory part — CCM — I don't know what is the state of the art there, but maybe there is something publishable with that. In the meantime, collect human data because that is what guides everything."*

---

## 10. Decisions Made

| Decision | Detail |
|----------|--------|
| **Hardware first** | Finish assembly + calibration before starting human experiments |
| **EMG proxy** | Use wireless EMG kit to capture muscle co-contraction during perturbation as proxy for human learning |
| **Ethics approval** | Start ethics application process for human experiments now (can be delayed) |
| **Plot labels** | Fix figure titles to distinguish τ_exo, τ_drive, τ_disturbance clearly |
| **Learning study** | Test parameter identification with impulse probe (not just sinusoid), document limits of linear regression |
| **CCM integration** | Deepen understanding of how K_eff(exo) modifies the CCM SDP to formally prove exo helps (scripts 14e/14f direction) |

---

## 11. Action Items

| Owner | Action | Priority |
|-------|--------|----------|
| Dipankar | Finish hardware assembly + motor calibration | HIGH |
| Dipankar | Fix plot labels (τ_exo vs τ_disturbance disambiguation) | HIGH |
| Dipankar | Run impulse probe experiments + document regression limits | MEDIUM |
| Dipankar | Connect CCM to manipulator+exo system (scripts 14e/14f) | MEDIUM |
| Dipankar | Add ground truth K_eff vs identified K_eff comparison to learning plots | MEDIUM |
| Dipankar | Draft 1-page paper outline (mechanism + control + results) | MEDIUM |
| Thrishantha | Share sling/perturbation experimental paradigm reference | LOW |
| Team | Start ethics approval process for human study | LOW |

---

## 12. Open Questions (for next meeting)

1. **CCM synthesis**: Can the exo's K_eff term be embedded into the SDP LMI directly, and how much does it improve λ_max? (scripts 14e vs 14f comparison)
2. **Learning generalisation**: If we learn K_eff from a sinusoidal probe, does it transfer to a circle or figure-8 task?
3. **Biological question**: Does the brain use something like CCM (certify over a set) or something simpler (tune for a specific scenario)? How do we test this experimentally?
4. **Paper scope**: Hardware + mechanism paper only, or include simulation + CCM analysis?
5. **Human protocol**: End-effector hold + sinusoidal perturbation vs. elbow sling + perturbation — which is simpler for ethics approval?

---

## 13. Notable Quotes

> **Thrishantha**: *"Our hypothesis is the brain will learn better with the exosuit. Without the exosuit it's very random and the brain doesn't see a pattern. The exo makes it a learnable problem."*

> **Salah**: *"A control contraction metric would guarantee that these K's and B's are going to guarantee contraction, no matter what the disturbance magnitude. That's my philosophical answer to the 'why CCM'."*

> **Dagmar**: *"That is new, right? An elegant way of manipulating a 2-joint arm with this cable design. The added XO acting on the stiffness of the elbow — that is new."*

> **Dipankar**: *"I believe from my career perspective, which will end next year end of October, I need something prestigious. I got a lot of support from the lab and I think we did it pretty fast."*

---

*Notes compiled from transcript: `transcripts/2026_05_20_Progress Update 2_ Exosuit .vtt`*
