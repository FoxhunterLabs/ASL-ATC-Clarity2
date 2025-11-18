# ATC Clarity Console — Research Prototype

A small, self-contained demo of an ATC “clarity console” — a side UI that helps humans
see sector risk, conflicts, workload, and comms load at a glance.

Built in **Python + Streamlit**, using **fully synthetic** telemetry.
No live data. Not connected to any operational system. Not for real-world use.

---

## What this demo does

The app spins up a toy ATC sector and continuously moves a small fleet of aircraft around a
map while computing a handful of heuristics.

### Synthetic traffic + motion

- Generates ~20 mock aircraft with:
- `id`, altitude, speed, heading
- lat / lon in a small box around Indianapolis
- destination airport (IND / ORD / SDF / CVG)
- Every tick:
- Aircraft advance along their heading using a simple kinematics step
- Headings drift slightly so the pattern doesn’t look perfectly scripted

### Conflict detection

Deliberately simple, just to drive the UI:

- Vertical threshold: **< 800 ft**
- Lateral box: **~0.02° lat / 0.02° lon**
- Any pair inside those bounds is flagged as a conflict

### Workload + clarity score

From the current sector state, the console computes:

- **Traffic load** – number of active aircraft
- **Workload index** – mixes traffic volume + conflict count
`idx = min(1.0, (count/40) + conflicts * 0.15)`
- **Comms fraction** – random fraction between 0.05–0.25 to stand in for “how busy the
frequency feels”
- **Clarity score (0–100)** – starts at 100 and gets penalized by:
- current conflicts
- predicted conflicts (short trend over recent history)
- workload index
- comms fraction

> The clarity score is a **vibe / spice metric**, not a safety metric.

### Bayesian condition layer

On top of the raw numbers, a tiny Bayesian layer classifies the sector into four high-level
conditions:

- `STABLE`
- `ELEVATED`
- `HIGH_LOAD`
- `CRITICAL`

Priors + evidence are hand-tuned; the goal is to show how you might map noisy telemetry
into a small, interpretable state machine — not to claim statistical rigor.

### UI surfaces

The Streamlit UI shows:

- **Top metrics**
- Clarity %
- Active conflict count
- Predicted conflicts (5 min)
- Traffic load, workload index, comms fraction
- **Bayesian confidence bar chart**
- Probability mass over STABLE / ELEVATED / HIGH_LOAD / CRITICAL
- Call-out for the most likely condition
- **Sector map**
- Plotly scatter of aircraft positions (moving dots)
- Conflict aircraft highlighted
- **Active aircraft telemetry table**
- One row per aircraft

- **Conflict table**
- List of current conflict pairs (if any)
- **Human-gated action panel**
- Operator can choose:
- “Hold all departures”
- “Issue spacing instructions”
- “Request altitude separation”
- “Do nothing (monitor only)”
- Choice is confirmed in-UI with a timestamp

The system **never acts on its own**. It only suggests; a human has to choose and
confirm.

---

## Controls

Left sidebar:

- **Auto-run** – when checked, the sim advances automatically
- **Update interval (sec)** – how often to step the sim when auto-run is on
- **Step once** – advance a single tick when auto-run is off

---

## Running the demo locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/ASL-ATC-Clarity2.git
cd ASL-ATC-Clarity2
