import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import math
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="ATC Clarity Console", layout="wide")

# ---------------------------
# Helper Generators (Mock Telemetry)
# ---------------------------

def generate_aircraft(n=20):
planes = []
for i in range(n):
planes.append(
{
"id": f"AC{i+1}",
"alt": random.randint(2000, 38000),
"speed": random.randint(250, 520), # kts-ish
"heading": random.uniform(0, 360), # deg
"lat": 39 + random.uniform(-0.3, 0.3),
"lon": -86 + random.uniform(-0.3, 0.3),

"destination": random.choice(["IND", "ORD", "SDF", "CVG"]),
}
)
return planes

def step_aircraft(planes, dt_sec=1.0):
"""Very simple motion model: move each plane dt seconds along its heading."""
if planes is None:
return None

dt_min = dt_sec / 60.0
for p in planes:
speed = p["speed"] # knots
dist_nm = speed * dt_min / 60.0
dist_deg = dist_nm / 60.0 * 0.25 # shrink factor so they stay on-screen

theta = math.radians(p["heading"])
dlat = dist_deg * math.cos(theta)
dlon = dist_deg * math.sin(theta)

p["lat"] += dlat
p["lon"] += dlon

# tiny heading drift so patterns don’t look too synthetic
p["heading"] += random.uniform(-2, 2)

return planes

def conflict_detection(planes):
conflicts = []
for i in range(len(planes)):
for j in range(i + 1, len(planes)):
dy_alt = abs(planes[i]["alt"] - planes[j]["alt"])
if dy_alt < 800: # vertical threshold
dx = abs(planes[i]["lat"] - planes[j]["lat"])
dy = abs(planes[i]["lon"] - planes[j]["lon"])
if dx < 0.02 and dy < 0.02:
conflicts.append((planes[i]["id"], planes[j]["id"]))
return conflicts

def predict_conflicts(history):
if len(history) < 4:
return 0
recent = [len(x) for x in history[-4:]]
slope = (recent[-1] - recent[0]) / 3
pred = max(0, int(recent[-1] + slope))
return pred

def compute_workload(planes, conflicts):
return {
"count": len(planes),
"idx": min(1.0, (len(planes) / 40) + (len(conflicts) * 0.15)),
}

def compute_communications():
return {"fraction": random.uniform(0.05, 0.25)}

def compute_clarity(conflicts, pred_conf, workload_idx, comms_frac):
base = 100
base -= len(conflicts) * 20
base -= pred_conf * 10
base -= workload_idx * 15
base -= comms_frac * 20
return max(0, min(100, base))

# ---------------------------
# Bayesian Confidence Model
# ---------------------------

def bayesian_confidence(prior, evidence):
posterior = {}

for state in prior:
posterior[state] = prior[state] * evidence[state]
total = sum(posterior.values()) + 1e-9
for state in posterior:
posterior[state] /= total
return posterior

def compute_evidence(clarity, conflicts_now, pred_conflicts, workload_idx, comms_frac):
return {
"STABLE": max(
0.01, (clarity / 100) * (1 - conflicts_now * 0.4) * (1 - pred_conflicts * 0.1)
),
"ELEVATED": max(0.01, (clarity / 100) * (0.4 + workload_idx * 0.4)),
"HIGH_LOAD": max(0.01, workload_idx * 0.7 + comms_frac * 0.3),
"CRITICAL": max(
0.01, conflicts_now * 0.6 + pred_conflicts * 0.4 + (1 - clarity / 100)
),
}

# ---------------------------
# Streamlit State
# ---------------------------

if "history" not in st.session_state:

st.session_state.history = []

if "planes" not in st.session_state:
st.session_state.planes = None

if "last_update" not in st.session_state:
st.session_state.last_update = time.time()

# ---------------------------
# UI – Header & Controls
# ---------------------------

st.title(" ATC Clarity Console (Research Prototype)")
st.caption("Human-gated, auditable, physics-first decision support for airspace safety.")

st.sidebar.subheader("Simulation Controls")
auto_run = st.sidebar.checkbox("Auto-run", value=True)
update_interval = st.sidebar.slider("Update interval (sec)", 0.5, 5.0, 1.0, 0.5)
step_once = st.sidebar.button("Step once")

# ---------------------------
# Simulation Step (auto-move)
# ---------------------------

# init planes
if st.session_state.planes is None:
st.session_state.planes = generate_aircraft()

now = time.time()
dt = now - st.session_state.last_update

if auto_run and dt >= update_interval:
# advance and immediately rerun
st.session_state.planes = step_aircraft(st.session_state.planes, dt_sec=dt)
st.session_state.last_update = now
st.experimental_rerun()
elif step_once:
st.session_state.planes = step_aircraft(
st.session_state.planes, dt_sec=update_interval
)
st.session_state.last_update = now

planes = st.session_state.planes
conflicts = conflict_detection(planes)

pred_conf = predict_conflicts(st.session_state.history)
work = compute_workload(planes, conflicts)
comms = compute_communications()
clarity = compute_clarity(conflicts, pred_conf, work["idx"], comms["fraction"])

st.session_state.history.append(conflicts)

# ---------------------------
# Bayesian Layer
# ---------------------------

priors = {
"STABLE": 0.45,
"ELEVATED": 0.30,
"HIGH_LOAD": 0.15,
"CRITICAL": 0.10,
}

evidence = compute_evidence(
clarity, len(conflicts), pred_conf, work["idx"], comms["fraction"]
)

posterior = bayesian_confidence(priors, evidence)
best_state = max(posterior, key=posterior.get)

# ---------------------------
# LAYOUT
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
st.metric("Clarity", f"{clarity:.1f}%", delta=None)
st.metric("Active Conflicts", len(conflicts))
st.metric("Predicted Conflicts (5 min)", pred_conf)

with col2:
st.metric("Traffic Load", work["count"])
st.metric("Workload Index", f"{work['idx']:.2f}")
st.metric("Comms Fraction", f"{comms['fraction']:.2f}")

with col3:
st.subheader("Bayesian Confidence")
bayes_df = pd.DataFrame(
{
"State": list(posterior.keys()),
"Confidence (%)": [round(v * 100, 1) for v in posterior.values()],
}
)
st.bar_chart(bayes_df.set_index("State"))
st.write(
f"**Most likely condition:** {best_state} "
f"({posterior[best_state] * 100:.1f}% confidence)"
)

# ---------------------------
# Sector Map (moving dots)

# ---------------------------

st.subheader("Sector Map (Synthetic)")

def make_sector_figure(planes, conflicts):
df = pd.DataFrame(planes)

conflict_ids = set()
for a, b in conflicts:
conflict_ids.add(a)
conflict_ids.add(b)
df["in_conflict"] = df["id"].isin(conflict_ids)

fig = px.scatter(
df,
x="lon",
y="lat",
text="id",
color="in_conflict",
labels={"lon": "Longitude", "lat": "Latitude", "in_conflict": "Conflict"},
)
fig.update_traces(textposition="top center")
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(
xaxis_showgrid=False,
yaxis_showgrid=False,

xaxis_title=None,
yaxis_title=None,
margin=dict(l=10, r=10, t=10, b=10),
)
return fig

st.plotly_chart(make_sector_figure(planes, conflicts), use_container_width=True)

# ---------------------------
# TABLE OF AIRCRAFT
# ---------------------------

st.subheader("Active Aircraft Telemetry")
st.dataframe(pd.DataFrame(planes))

# ---------------------------
# CONFLICT TABLE
# ---------------------------

if conflicts:
st.subheader(" Current Conflicts")
st.dataframe(pd.DataFrame(conflicts, columns=["Plane A", "Plane B"]))
else:
st.subheader(" No current conflicts detected")

# ---------------------------

# HUMAN-GATED ACTION PANEL
# ---------------------------

st.subheader("Human-Gated Interventions")
st.write("System will **never** act alone. Operator approval required.")

act = st.radio(
"Action:",
[
"Hold all departures",
"Issue spacing instructions",
"Request altitude separation",
"Do nothing (monitor only)",
],
)

confirm = st.button("Confirm Action")

if confirm:
st.success(f"Action logged: {act} at {datetime.now().strftime('%H:%M:%S')}")
