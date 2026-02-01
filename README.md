# COMP 4360 — Group 6

## Full Project Proposal (Slide Deck Plan)

This presentation is our **technical plan** for the term project. The goal is to show that our idea is **doable**, that we understand the paper we chose, and that we have a realistic plan to adapt it to a new dataset.

### What the proposal presentation is for
- Show a clear **paper + dataset pairing** and why it makes sense
- Explain what we will **build**, what we will **change**, and what might be **hard**
- Show that we have a plan for:
  - data preparation
  - training and evaluation
  - ablations / sensitivity checks
  - teamwork and timeline

### Presentation format
- **Length:** 5–10 minutes  
- **Q&A:** Each student may be asked **1–2 questions** to confirm:
  - what they worked on
  - what they understand
  - what their responsibilities are

### Submission
- **Due:** **5:00 PM (Monday) February 2, 2026**
- **Deliverable:** Slide deck uploaded to UM Learn

---

## Slide-by-Slide Outline

### Slide 1 — Title Slide
Include:
- Project name (short + clear)
- Group member names
- Course: COMP 4360
- Instructor name
- Date (Feb 2026)

Why this slide matters:
- It makes the presentation easy to identify, and sets context immediately.

---

### Slide 2 — Problem Domain & Dataset
Goal: Explain **what problem we are solving** and **what data we will use**.

Include:
- A simple description of the task (1–2 lines)
- What the model will predict / output (e.g., class label, segmentation mask, etc.)
- Dataset details (be specific):
  - **Size:** number of samples (and train/val/test split if known)
  - **Resolution:** image size or data shape (if not images, describe the input format clearly)
  - **File format:** PNG/JPEG/CSV/NIfTI/etc.
  - **Class distribution:** counts or proportions per class (and whether the data is imbalanced)
- A clear statement:
  - **This dataset is different from the dataset used in the original paper.**

Optional (if space allows):
- A quick note on why this dataset is a good fit (availability, labels, difficulty, etc.)

---

### Slide 3 — Source Paper & Method Overview
Goal: Show we understand the paper we are implementing.

Include:
- Paper info:
  - Title
  - Venue + year (must meet the 2018+ rule)
- A high-level method summary:
  - What the model takes as input
  - What it produces as output
  - The key idea (in plain language)
- If possible, include a **simple architecture sketch** (not a huge diagram)
  - Just the main blocks and data flow

---

### Slide 4 — Adaptation Strategy (Domain Shift)
Goal: Explain what we must change to make the paper’s method work on our dataset.

Include:
- **What stays the same**
  - core architecture idea
  - main training objective (if applicable)
- **What changes**
  - input format differences (channels, resolution, 2D vs 3D, etc.)
  - preprocessing steps and why they are needed
  - label format changes (if applicable)
- **Core technical challenges**
  - the 1–2 things most likely to cause issues (and how we will deal with them)

Examples (use only what applies to your case):
- changing input channels (RGB → multi-channel)
- handling different image sizes
- rewriting dataloader for different formats
- dealing with class imbalance
- adjusting loss function (if label structure differs)

---

### Slide 5 — Development Plan (Environment + Training Plan + Ablations)
Goal: Explain *how* we will build and test the system.

#### Environment
- Framework: (e.g., PyTorch)
- Hardware:
  - Primary: GTX 1080 Ti
  - Backup: Google Colab / University GPU if required
- Tools we will use (examples):
  - experiment tracking (optional)
  - version control (Git)

#### Training plan
- Baseline plan:
  - implement model
  - run a small overfit test (tiny subset) to confirm correctness
  - scale up to full training
- Evaluation plan:
  - what metric(s) we will report (accuracy, IoU, F1, etc.)
  - how we will compare results fairly

#### Ablation / sensitivity plan (minimum 1)
We will include at least one experiment that shows what happens when we remove or weaken a component.

Examples of good ablations:
- remove a module (attention block, refinement stage, etc.)
- turn off a preprocessing step
- modify a key hyperparameter to test stability (learning rate, threshold, augmentation level)

---

### Slide 6 — Project Management (Gantt Chart + Milestones)
Goal: Show a timeline that makes the work feel realistic.

Include a Gantt chart with clear milestones like:
- Dataset downloaded + cleaned
- Data loader working
- Baseline model training runs
- Main model training runs
- Ablation experiments
- Final analysis + write-up

Tip:
- Put dates or week numbers so the schedule feels real.

---

### Slide 7 — Role Division (Who does what)
Goal: Make responsibilities clear so the instructor knows the work is split fairly.

Core responsibilities:
- **Implementation:** Gabe  
- **Training:** Rudy  
- **Ablation studies + analysis:** Satyam  

Also mention collaboration expectations:
- everyone will review results
- everyone will help with debugging / presentation polishing when needed

---

### Slide 8 — EDI Considerations
Goal: Show we thought about fairness, representation, and real-world impact.

Include:
- What EDI issue could exist in our dataset or labels
  - class imbalance
  - representation gaps
  - possible bias from how the data was collected
- How we will check or reduce risk
  - report class distribution
  - use balanced sampling or class-weighted loss (if needed)
  - include subgroup performance reporting if relevant/possible

Keep it honest:
- If we cannot fully address a bias, we should still acknowledge it clearly.

---

### Slide 9 — References (IEEE)
Goal: Provide clean and correct citations.

Include IEEE-style references for:
- the source paper
- the dataset source
- any external tools or papers referenced in slides

---

## Work Split for Slide Creation

To keep slide writing efficient, we will split slide ownership like this:

- **Rudy:** Slides 2, 7, 8  
- **Satyam:** Slides 5, 6  
- **Gabe:** Slides 1, 3, 9  

---
