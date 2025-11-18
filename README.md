# ATC Clarity Console — Synthetic Demo

> ** Safety disclaimer:**
> This is a **research prototype** and uses **fully synthetic data**.
> It is **not** connected to any live ATC systems and is **not for operational use**.

Most “AI for ATC” concepts jump straight to full autonomy.
This project explores something simpler and more conservative:

> A **clarity console** that helps humans see sector risk, conflicts, workload, and comms
load at a glance, while keeping humans firmly in the loop.

The app is built in **Python + Streamlit** and runs a small synthetic airspace sector to
exercise the UI and logic.

---

## What it does

The ATC Clarity Console demo:

- Generates a **mock sector** with ~20 synthetic aircraft:
- ID, altitude, speed, lat/lon, destination.
- Runs a **basic conflict detector**:
- Vertical separation threshold.
- Small lat/lon box for lateral proximity.

- Keeps a **rolling history of conflicts** and:
- Estimates short-term **predicted conflicts** based on recent trend.
- Computes a **workload index** that combines:
- Traffic count.
- Current conflict count.
- Estimates a **comms load** as a simple random fraction (how “busy” the frequency feels
in this toy model).
- Rolls everything into a single **clarity score** (0–100) that drops as:
- Current conflicts increase.
- Predicted conflicts increase.
- Workload rises.
- Comms fraction rises.
- Adds a small **Bayesian confidence layer** with four high-level conditions:
- `STABLE`
- `ELEVATED`
- `HIGH_LOAD`
- `CRITICAL`
- Provides a **human-gated action panel**:
- System can suggest actions like *“Hold all departures”* or *“Request altitude
separation”*.
- An operator must explicitly confirm the choice.
- Every confirmed action is **logged with a timestamp**.

Again: this is a **thinking tool / UI exploration**, not a certified safety system.

---

## Tech stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) for the UI
- [pandas](https://pandas.pydata.org/) for table views
- [NumPy](https://numpy.org/) for basic numeric handling

---

## Running it locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/atc-clarity-console.git
cd atc-clarity-console
