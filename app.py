# app.py
# ATC Clarity Console — Synthetic, Human-Gated, Predictive Demo
#
# ⚠️ WARNING: Synthetic demo. NOT FOR REAL-WORLD AIR TRAFFIC USE. ⚠️

import math
import random
import time
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------------------
# Streamlit page setup
# ---------------------------

st.set_page_config(
    page_title="ATC Clarity Console — Synthetic Demo",
    layout="wide",
)

# ---------------------------
# Session state init
# ---------------------------

def init_state():
    s = st.session_state
    if "tick" not in s:
        s.tick = 0
    if "running" not in s:
        s.running = False
    if "last_update" not in s:
        s.last_update = time.time()
    if "flights" not in s:
        s.flights = []
    if "history" not in s:
        s.history = []
    if "events" not in s:
        s.events = []
    if "proposals" not in s:
        s.proposals = []
    if "gate_open" not in s:
        s.gate_open = False
    if "sector_id" not in s:
        s.sector_id = "ZAB-42"
    if "audit_log" not in s:
        s.audit_log = []
    if "prediction_horizon_min" not in s:
        s.prediction_horizon_min = 8
    if "sector_center_lat" not in s:
        s.sector_center_lat = 39.0
    if "sector_center_lon" not in s:
        s.sector_center_lon = -97.0
    if "sector_lat_span" not in s:
        s.sector_lat_span = 3.0
    if "sector_lon_span" not in s:
        s.sector_lon_span = 4.0
    if "saved_scenarios" not in s:
        s.saved_scenarios = []
    if "playback_mode" not in s:
        s.playback_mode = False
    if "playback_index" not in s:
        s.playback_index = 0
    if "playback_data" not in s:
        s.playback_data = []
    if "sectors" not in s:
        s.sectors = {
            "ZAB-42": {"center_lat": 39.0, "center_lon": -97.0, "lat_span": 3.0, "lon_span": 4.0, "active": True},
            "ZAB-43": {"center_lat": 42.0, "center_lon": -97.0, "lat_span": 3.0, "lon_span": 4.0, "active": False},
            "ZAB-44": {"center_lat": 39.0, "center_lon": -93.0, "lat_span": 3.0, "lon_span": 4.0, "active": False},
        }
    if "active_sectors" not in s:
        s.active_sectors = ["ZAB-42"]
    if "conflict_resolution_strategy" not in s:
        s.conflict_resolution_strategy = "altitude_separation"
    if "selected_sector" not in s:
        s.selected_sector = "ZAB-42"
    if "playback_events" not in s:
        s.playback_events = []
    if "last_selected_sector" not in s:
        s.last_selected_sector = "ZAB-42"
    if "bayesian_state" not in s:
        s.bayesian_state = {}
    if "action_log" not in s:
        s.action_log = []

init_state()

MAX_HISTORY = 600

# ---------------------------
# Synthetic sector geometry
# ---------------------------

MIN_HORIZONTAL_SEP_NM = 5.0
MIN_VERTICAL_SEP_FT = 1000.0

def get_sector_bounds(sector_id=None):
    if sector_id and sector_id in st.session_state.sectors:
        sector = st.session_state.sectors[sector_id]
        return {
            "center_lat": sector["center_lat"],
            "center_lon": sector["center_lon"],
            "lat_span": sector["lat_span"],
            "lon_span": sector["lon_span"],
        }
    return {
        "center_lat": st.session_state.sector_center_lat,
        "center_lon": st.session_state.sector_center_lon,
        "lat_span": st.session_state.sector_lat_span,
        "lon_span": st.session_state.sector_lon_span,
    }

def get_active_sector_bounds():
    return get_sector_bounds(st.session_state.selected_sector)

# ---------------------------
# Utilities
# ---------------------------

def nm_to_deg_lat(nm: float) -> float:
    return nm / 60.0

def nm_to_deg_lon(nm: float, lat_deg: float) -> float:
    return nm / (60.0 * math.cos(math.radians(lat_deg)) + 1e-6)

def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_nm * c

def log_event(level: str, msg: str, extras: Dict[str, Any] = None):
    payload = {
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
        "level": level,
        "msg": msg,
        "sector": st.session_state.sector_id,
    }
    if extras:
        payload.update(extras)
    st.session_state.events.append(payload)
    st.session_state.events = st.session_state.events[-200:]

# ---------------------------
# Synthetic flight model
# ---------------------------

CALLSIGNS = [
    "DAL123", "UAL455", "SWA208", "AAL902", "JBU77", "FDX301", "UPS441",
    "RYR9001", "BAW52", "AFR221", "QFA11", "ACA315"
]

def random_flight(callsign: str) -> Dict[str, Any]:
    bounds = get_active_sector_bounds()
    lat = bounds["center_lat"] + random.uniform(-bounds["lat_span"]/2, bounds["lat_span"]/2)
    lon = bounds["center_lon"] + random.uniform(-bounds["lon_span"]/2, bounds["lon_span"]/2)
    alt = random.choice([240, 260, 280, 300, 320, 340]) * 100  # flight levels
    heading = random.uniform(0, 360)
    speed_kt = random.uniform(380, 470)
    return {
        "callsign": callsign,
        "lat": lat,
        "lon": lon,
        "alt_ft": alt,
        "heading_deg": heading,
        "speed_kt": speed_kt,
        "squawk": random.randint(1000, 7777),
        "status": "NORMAL",
    }

def init_flights(n: int = 14) -> List[Dict[str, Any]]:
    flights = []
    used = set()
    for _ in range(n):
        cs = random.choice(CALLSIGNS)
        while cs in used:
            cs = random.choice(CALLSIGNS)
        used.add(cs)
        flights.append(random_flight(cs))
    return flights

def step_flights(flights: List[Dict[str, Any]], dt_sec: float = 10.0):
    """Simple straight-line motion with some heading noise."""
    bounds = get_active_sector_bounds()
    for f in flights:
        # random little heading wiggle
        f["heading_deg"] += random.uniform(-1.5, 1.5)
        hdg_rad = math.radians(f["heading_deg"])
        dist_nm = f["speed_kt"] * (dt_sec / 3600.0)
        dlat = nm_to_deg_lat(dist_nm * math.cos(hdg_rad))
        dlon = nm_to_deg_lon(dist_nm * math.sin(hdg_rad), f["lat"])
        f["lat"] += dlat
        f["lon"] += dlon

        # simple "keep inside sector" reflection
        if f["lat"] > bounds["center_lat"] + bounds["lat_span"]/2 or f["lat"] < bounds["center_lat"] - bounds["lat_span"]/2:
            f["heading_deg"] += 180
        if f["lon"] > bounds["center_lon"] + bounds["lon_span"]/2 or f["lon"] < bounds["center_lon"] - bounds["lon_span"]/2:
            f["heading_deg"] += 180

        # sometimes change altitude slowly
        if random.random() < 0.05:
            f["alt_ft"] += random.choice([-1000, 1000])

# ---------------------------
# Conflict & workload metrics
# ---------------------------

def detect_current_conflicts(flights: List[Dict[str, Any]]):
    conflicts = []
    n = len(flights)
    for i in range(n):
        for j in range(i+1, n):
            a = flights[i]
            b = flights[j]
            dist_nm = haversine_nm(a["lat"], a["lon"], b["lat"], b["lon"])
            dalt = abs(a["alt_ft"] - b["alt_ft"])
            if dist_nm < MIN_HORIZONTAL_SEP_NM and dalt < MIN_VERTICAL_SEP_FT:
                conflicts.append({
                    "pair": (a["callsign"], b["callsign"]),
                    "dist_nm": dist_nm,
                    "dalt_ft": dalt,
                })
    return conflicts

def compute_workload(flights: List[Dict[str, Any]]):
    # Crude: workload index based on count and maneuvering rate
    count = len(flights)
    base = min(1.0, count / 18.0)  # normalized
    maneuvering = 0.0
    for f in flights:
        if random.random() < 0.2:
            maneuvering += 1
    maneuvering = maneuvering / max(1, count)
    return {
        "count": count,
        "index": min(1.0, 0.7 * base + 0.3 * maneuvering)
    }

def estimate_comms_load(flights: List[Dict[str, Any]]):
    # fake: some proportion of flights "talking"
    talking = 0
    for _ in flights:
        if random.random() < 0.35:
            talking += 1
    fraction = talking / max(1, len(flights))
    return {
        "active": talking,
        "fraction": fraction
    }

# ---------------------------
# Clarity metric (0–100)
# ---------------------------

def compute_clarity(conflicts, workload, comms, predicted_conflicts_count):
    # Base clarity starts high, penalized by real & predicted conflicts + workload + comms
    conflict_penalty = min(30, 7 * len(conflicts))
    predicted_penalty = min(20, 3 * predicted_conflicts_count)
    workload_penalty = 25 * workload["index"]
    comms_penalty = 15 * comms["fraction"]

    clarity = 100 - (conflict_penalty + predicted_penalty + workload_penalty + comms_penalty)
    clarity = max(0, min(100, clarity))
    return clarity

def classify_state(clarity: float, conflicts_count: int):
    if clarity >= 90 and conflicts_count == 0:
        return "STABLE"
    elif clarity >= 80:
        return "ELEVATED"
    elif clarity >= 65:
        return "HIGH_LOAD"
    else:
        return "CRITICAL"

# ---------------------------
# Bayesian Confidence Model
# ---------------------------

def bayesian_confidence(prior, evidence):
    """Compute posterior probabilities using Bayesian inference."""
    posterior = {}
    for state in prior:
        posterior[state] = prior[state] * evidence[state]
    total = sum(posterior.values()) + 1e-9
    for state in posterior:
        posterior[state] /= total
    return posterior

def compute_evidence(clarity, conflicts_now, pred_conflicts, workload_idx, comms_frac):
    """Compute evidence (likelihood) for each state based on observed metrics."""
    return {
        "STABLE": max(
            0.01,
            (clarity / 100)
            * (1 - conflicts_now * 0.4)
            * (1 - pred_conflicts * 0.1),
        ),
        "ELEVATED": max(
            0.01,
            (clarity / 100) * (0.4 + workload_idx * 0.4),
        ),
        "HIGH_LOAD": max(
            0.01,
            workload_idx * 0.7 + comms_frac * 0.3,
        ),
        "CRITICAL": max(
            0.01,
            conflicts_now * 0.6 + pred_conflicts * 0.4 + (1 - clarity / 100),
        ),
    }

# ---------------------------
# Predictive layer
# ---------------------------

def simulate_future(flights: List[Dict[str, Any]], horizon_min: int = 8, step_sec: int = 30):
    """Fast forward each flight and detect conflicts in the future window."""
    sim_flights = [f.copy() for f in flights]
    total_conflicts = 0
    worst_min_sep = 999.0

    sim_steps = int((horizon_min * 60) / step_sec)
    for _ in range(sim_steps):
        step_flights(sim_flights, dt_sec=step_sec)
        future_conflicts = detect_current_conflicts(sim_flights)
        total_conflicts += len(future_conflicts)
        for c in future_conflicts:
            worst_min_sep = min(worst_min_sep, c["dist_nm"])

    return {
        "future_conflicts": total_conflicts,
        "worst_min_sep_nm": worst_min_sep if worst_min_sep < 999 else None,
        "horizon_min": horizon_min,
    }

# ---------------------------
# Advanced Conflict Resolution Strategies
# ---------------------------

def generate_resolution_strategies(conflicts, flights):
    """Generate multiple conflict resolution strategies based on the current situation."""
    strategies = []
    
    if not conflicts:
        return strategies
    
    strategy_type = st.session_state.conflict_resolution_strategy
    
    if strategy_type == "altitude_separation":
        strategies.append({
            "name": "Altitude Separation",
            "description": "Assign different flight levels to conflicting aircraft",
            "actions": [f"Assign {c['pair'][0]} to FL{int(random.choice([260, 280, 300]))} and {c['pair'][1]} to FL{int(random.choice([240, 320, 340]))}" for c in conflicts[:3]]
        })
    
    if strategy_type == "lateral_vectoring" or strategy_type == "combined":
        strategies.append({
            "name": "Lateral Vectoring",
            "description": "Issue heading changes to create horizontal separation",
            "actions": [f"Vector {c['pair'][0]} left 20°, maintain {c['pair'][1]} on course" for c in conflicts[:3]]
        })
    
    if strategy_type == "speed_control" or strategy_type == "combined":
        strategies.append({
            "name": "Speed Control",
            "description": "Adjust aircraft speeds to manage separation timing",
            "actions": [f"Reduce {c['pair'][0]} to 380 kts, maintain {c['pair'][1]} at 450 kts" for c in conflicts[:3]]
        })
    
    if strategy_type == "combined":
        strategies.append({
            "name": "Combined Approach",
            "description": "Use multiple techniques for optimal conflict resolution",
            "actions": [
                f"Assign {conflicts[0]['pair'][0]} to higher altitude and reduce speed",
                f"Vector {conflicts[0]['pair'][1]} for lateral separation",
                "Coordinate with adjacent sectors for handoff timing"
            ] if conflicts else []
        })
    
    return strategies

# ---------------------------
# Proposal engine (human-gated)
# ---------------------------

def maybe_generate_proposals(latest_row):
    clarity = latest_row["clarity"]
    pred_conf = latest_row["pred_future_conflicts"]
    workload_idx = latest_row["workload_index"]
    flights = latest_row["flight_count"]

    triggers = []
    if clarity < 80:
        triggers.append("low_clarity")
    if pred_conf > 0:
        triggers.append("predicted_conflict")
    if workload_idx > 0.7:
        triggers.append("high_workload")
    if flights > 16:
        triggers.append("high_traffic")

    open_proposals = [p for p in st.session_state.proposals if p["status"] == "PENDING"]
    if not triggers or len(open_proposals) > 5:
        return

    pid = len(st.session_state.proposals) + 1

    if "predicted_conflict" in triggers:
        title = "Slow departures & vector early to break conflicts"
    elif "high_workload" in triggers:
        title = "Temporarily cap arrivals into sector"
    else:
        title = "Stabilize sector configuration"

    rationale = []
    if "low_clarity" in triggers:
        rationale.append(f"clarity {clarity:.1f} below comfort band")
    if "predicted_conflict" in triggers:
        rationale.append(f"{pred_conf} likely conflicts within {st.session_state.prediction_horizon_min} min")
    if "high_workload" in triggers:
        rationale.append(f"workload index {workload_idx:.2f} trending high")
    if "high_traffic" in triggers:
        rationale.append(f"{flights} active tracks in sector")

    proposal = {
        "id": pid,
        "created": datetime.utcnow().strftime("%H:%M:%S"),
        "title": title,
        "status": "PENDING",
        "rationale": "; ".join(rationale),
        "bounds": {
            "max_arrival_rate_change": "≤ 25%",
            "max_departure_delay": "≤ 5 minutes",
            "no_automatic_clearances": True,
        },
        "snapshot": latest_row,
    }
    st.session_state.proposals.append(proposal)
    log_event("PROPOSAL", f"Proposal #{pid} queued: {title}", {"clarity": clarity})

# ---------------------------
# Sidebar: Configuration
# ---------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("🗺️ Sector Geometry", expanded=False):
        if st.session_state.selected_sector != st.session_state.last_selected_sector:
            current_sector = st.session_state.sectors.get(st.session_state.selected_sector, {})
            if current_sector:
                st.session_state.sector_center_lat = current_sector.get("center_lat", st.session_state.sector_center_lat)
                st.session_state.sector_center_lon = current_sector.get("center_lon", st.session_state.sector_center_lon)
                st.session_state.sector_lat_span = current_sector.get("lat_span", st.session_state.sector_lat_span)
                st.session_state.sector_lon_span = current_sector.get("lon_span", st.session_state.sector_lon_span)
            st.session_state.last_selected_sector = st.session_state.selected_sector
        
        st.write(f"**Editing: {st.session_state.selected_sector}**")
        
        st.session_state.sector_center_lat = st.number_input(
            "Center Latitude",
            value=st.session_state.sector_center_lat,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            help="Latitude of sector center in degrees"
        )
        st.session_state.sector_center_lon = st.number_input(
            "Center Longitude",
            value=st.session_state.sector_center_lon,
            min_value=-180.0,
            max_value=180.0,
            step=0.1,
            help="Longitude of sector center in degrees"
        )
        st.session_state.sector_lat_span = st.number_input(
            "Latitude Span",
            value=st.session_state.sector_lat_span,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            help="North-South extent in degrees"
        )
        st.session_state.sector_lon_span = st.number_input(
            "Longitude Span",
            value=st.session_state.sector_lon_span,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            help="East-West extent in degrees"
        )
        if st.button("Apply Sector Changes"):
            selected = st.session_state.selected_sector
            st.session_state.sectors[selected]["center_lat"] = st.session_state.sector_center_lat
            st.session_state.sectors[selected]["center_lon"] = st.session_state.sector_center_lon
            st.session_state.sectors[selected]["lat_span"] = st.session_state.sector_lat_span
            st.session_state.sectors[selected]["lon_span"] = st.session_state.sector_lon_span
            st.session_state.flights = []
            log_event("INFO", f"Sector geometry updated for {selected}")
            st.rerun()
    
    with st.expander("💾 Scenarios", expanded=False):
        scenario_name = st.text_input("Scenario Name", placeholder="My Scenario")
        if st.button("💾 Save Current Scenario"):
            if scenario_name and st.session_state.history:
                scenario = {
                    "name": scenario_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "history": st.session_state.history.copy(),
                    "events": st.session_state.events.copy(),
                    "sector_id": st.session_state.selected_sector,
                    "sector_config": get_active_sector_bounds(),
                }
                st.session_state.saved_scenarios.append(scenario)
                log_event("INFO", f"Saved scenario: {scenario_name} for sector {st.session_state.selected_sector}")
                st.success(f"Saved scenario: {scenario_name}")
            else:
                st.warning("Please enter a name and run the simulation first.")
        
        if st.session_state.saved_scenarios:
            st.markdown("**Saved Scenarios:**")
            for idx, scenario in enumerate(st.session_state.saved_scenarios):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{idx + 1}. {scenario['name']}")
                with col2:
                    if st.button("▶️", key=f"play_{idx}"):
                        st.session_state.playback_mode = True
                        st.session_state.playback_data = scenario["history"]
                        st.session_state.playback_events = scenario.get("events", [])
                        st.session_state.playback_index = 0
                        st.session_state.running = False
                        st.session_state.flights = []
                        
                        scenario_sector = scenario.get("sector_id", st.session_state.selected_sector)
                        
                        if "sector_config" in scenario:
                            config = scenario["sector_config"]
                            st.session_state.sector_center_lat = config["center_lat"]
                            st.session_state.sector_center_lon = config["center_lon"]
                            st.session_state.sector_lat_span = config["lat_span"]
                            st.session_state.sector_lon_span = config["lon_span"]
                            
                            if scenario_sector in st.session_state.sectors:
                                st.session_state.sectors[scenario_sector]["center_lat"] = config["center_lat"]
                                st.session_state.sectors[scenario_sector]["center_lon"] = config["center_lon"]
                                st.session_state.sectors[scenario_sector]["lat_span"] = config["lat_span"]
                                st.session_state.sectors[scenario_sector]["lon_span"] = config["lon_span"]
                                st.session_state.sectors[scenario_sector]["active"] = True
                                st.session_state.selected_sector = scenario_sector
                                st.session_state.sector_id = scenario_sector
                        
                        log_event("INFO", f"Loading playback: {scenario['name']} for sector {scenario_sector}")
                        st.rerun()

    with st.expander("🌐 Multi-Sector View", expanded=False):
        active_sectors = [sid for sid, data in st.session_state.sectors.items() if data["active"]]
        
        if active_sectors:
            prev_sector = st.session_state.selected_sector
            st.session_state.selected_sector = st.selectbox(
                "Focus Sector",
                options=active_sectors,
                index=active_sectors.index(st.session_state.selected_sector) if st.session_state.selected_sector in active_sectors else 0,
                help="Select which sector to view and simulate"
            )
            
            if prev_sector != st.session_state.selected_sector:
                st.session_state.flights = []
                st.session_state.running = False
                st.session_state.sector_id = st.session_state.selected_sector
                log_event("INFO", f"Switched to sector {st.session_state.selected_sector}")
                st.rerun()
        
        st.markdown("**Sector Toggles:**")
        for sector_id, sector_data in st.session_state.sectors.items():
            is_active = st.checkbox(
                f"{sector_id} ({sector_data['center_lat']:.1f}°, {sector_data['center_lon']:.1f}°)",
                value=sector_data["active"],
                key=f"sector_{sector_id}"
            )
            st.session_state.sectors[sector_id]["active"] = is_active
        
        active_count = sum(1 for s in st.session_state.sectors.values() if s["active"])
        st.info(f"{active_count} sector(s) active")
    
    with st.expander("🎯 Conflict Resolution Strategy", expanded=False):
        st.session_state.conflict_resolution_strategy = st.selectbox(
            "Resolution Strategy",
            options=["altitude_separation", "lateral_vectoring", "speed_control", "combined"],
            index=["altitude_separation", "lateral_vectoring", "speed_control", "combined"].index(
                st.session_state.conflict_resolution_strategy
            ),
            format_func=lambda x: {
                "altitude_separation": "Altitude Separation",
                "lateral_vectoring": "Lateral Vectoring",
                "speed_control": "Speed Control",
                "combined": "Combined Approach"
            }[x]
        )
        
        st.markdown("**Strategy Details:**")
        if st.session_state.conflict_resolution_strategy == "altitude_separation":
            st.write("- Assigns different flight levels to conflicting aircraft")
            st.write("- Most reliable for vertical separation")
        elif st.session_state.conflict_resolution_strategy == "lateral_vectoring":
            st.write("- Issues heading changes for horizontal separation")
            st.write("- Effective for converging traffic")
        elif st.session_state.conflict_resolution_strategy == "speed_control":
            st.write("- Adjusts aircraft speeds to manage timing")
            st.write("- Useful for same-direction conflicts")
        else:
            st.write("- Uses multiple techniques simultaneously")
            st.write("- Optimal for complex situations")

    with st.expander("⏮️ Playback Controls", expanded=st.session_state.playback_mode):
        if st.session_state.playback_mode:
            st.write(f"**Playing back saved scenario**")
            st.write(f"Frame {st.session_state.playback_index + 1} of {len(st.session_state.playback_data)}")
            
            pb_col1, pb_col2, pb_col3 = st.columns(3)
            with pb_col1:
                if st.button("⏮️ Previous"):
                    if st.session_state.playback_index > 0:
                        st.session_state.playback_index -= 1
                        st.rerun()
            with pb_col2:
                if st.button("⏭️ Next"):
                    if st.session_state.playback_index < len(st.session_state.playback_data) - 1:
                        st.session_state.playback_index += 1
                        st.rerun()
            with pb_col3:
                if st.button("⏹️ Stop"):
                    st.session_state.playback_mode = False
                    st.session_state.playback_data = []
                    st.session_state.playback_index = 0
                    st.session_state.playback_events = []
                    st.rerun()
            
            playback_slider = st.slider(
                "Timeline",
                min_value=0,
                max_value=len(st.session_state.playback_data) - 1,
                value=st.session_state.playback_index,
                key="playback_slider_widget"
            )
            if playback_slider != st.session_state.playback_index:
                st.session_state.playback_index = playback_slider
                st.rerun()
        else:
            st.info("Load a saved scenario to use playback controls")

# ---------------------------
# UI Header & Controls
# ---------------------------

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown(
        "<h1 style='margin-bottom:0;'>ATC Clarity Console — Synthetic Demo</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Synthetic sector: **{st.session_state.sector_id}** — Human-gated decision support. "
        "All data simulated. Not for operational use."
    )

with head_right:
    st.write("")
    st.write("")
    st.toggle("Human Gate Open", key="gate_open", help="Gate must be open to approve any proposals.")

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
with ctrl1:
    if st.button("▶ Start Simulation", type="primary"):
        st.session_state.running = True
        if not st.session_state.flights:
            st.session_state.flights = init_flights()
        log_event("INFO", "Simulation started.")
        st.rerun()
with ctrl2:
    if st.button("⏸ Pause"):
        st.session_state.running = False
        log_event("INFO", "Simulation paused.")
with ctrl3:
    if st.button("⟲ Reset"):
        st.session_state.running = False
        st.session_state.tick = 0
        st.session_state.flights = []
        st.session_state.history = []
        st.session_state.events = []
        st.session_state.proposals = []
        log_event("INFO", "System reset.")
with ctrl4:
    st.number_input(
        "Prediction Horizon (min)",
        min_value=2,
        max_value=20,
        key="prediction_horizon_min",
        help="How far ahead the system looks for likely conflicts.",
    )

# ---------------------------
# Simulation step
# ---------------------------

now = time.time()
if st.session_state.running and now - st.session_state.last_update >= 1.0:
    if not st.session_state.flights:
        st.session_state.flights = init_flights()

    step_flights(st.session_state.flights, dt_sec=10.0)
    st.session_state.tick += 1
    st.session_state.last_update = now

    # Metrics
    conflicts = detect_current_conflicts(st.session_state.flights)
    workload = compute_workload(st.session_state.flights)
    comms = estimate_comms_load(st.session_state.flights)
    pred = simulate_future(
        st.session_state.flights,
        horizon_min=st.session_state.prediction_horizon_min,
        step_sec=30,
    )

    clarity = compute_clarity(
        conflicts,
        workload,
        comms,
        predicted_conflicts_count=pred["future_conflicts"],
    )
    state = classify_state(clarity, len(conflicts))
    
    # Compute Bayesian confidence
    priors = {
        "STABLE": 0.45,
        "ELEVATED": 0.30,
        "HIGH_LOAD": 0.15,
        "CRITICAL": 0.10,
    }
    evidence = compute_evidence(
        clarity,
        len(conflicts),
        pred["future_conflicts"],
        workload["index"],
        comms["fraction"],
    )
    bayesian_posterior = bayesian_confidence(priors, evidence)
    st.session_state.bayesian_state = bayesian_posterior

    if state in ("HIGH_LOAD", "CRITICAL"):
        log_event(
            "ALERT",
            f"{state} at tick {st.session_state.tick}: clarity {clarity:.1f}, "
            f"conflicts {len(conflicts)}, predicted {pred['future_conflicts']}",
        )

    row = {
        "tick": st.session_state.tick,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "clarity": clarity,
        "state": state,
        "flight_count": len(st.session_state.flights),
        "conflicts_now": len(conflicts),
        "pred_future_conflicts": pred["future_conflicts"],
        "pred_worst_min_sep_nm": pred["worst_min_sep_nm"],
        "workload_index": workload["index"],
        "comms_fraction": comms["fraction"],
    }
    st.session_state.history.append(row)
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]

    maybe_generate_proposals(row)

# ---------------------------
# Auto-rerun while running (must be before st.stop())
# ---------------------------

if st.session_state.running:
    time.sleep(0.2)
    st.rerun()

# ---------------------------
# Handle playback mode
# ---------------------------

if st.session_state.playback_mode and st.session_state.playback_data:
    df = pd.DataFrame(st.session_state.playback_data)
    latest = df.iloc[st.session_state.playback_index]
elif not st.session_state.history:
    st.info("Press **Start Simulation** to bring the synthetic sector online.")
    st.stop()
else:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]

# ---------------------------
# Top metrics row
# ---------------------------

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Tick", int(latest["tick"]))
    st.metric("State", latest["state"])
with m2:
    st.metric("Clarity", f"{latest['clarity']:.1f}")
    st.metric("Active Flights", int(latest["flight_count"]))
with m3:
    st.metric("Conflicts (Now)", int(latest["conflicts_now"]))
    st.metric("Predicted Conflicts", int(latest["pred_future_conflicts"]))
with m4:
    st.metric("Workload Index", f"{latest['workload_index']:.2f}")
    st.metric("Comms Fraction", f"{latest['comms_fraction']:.2f}")
with m5:
    st.metric("Worst Future Min Sep (nm)", 
              f"{latest['pred_worst_min_sep_nm']:.1f}" if latest["pred_worst_min_sep_nm"] else "—")
    st.metric("Prediction Horizon (min)", st.session_state.prediction_horizon_min)

st.markdown("---")

# ---------------------------
# Middle layout: charts + map
# ---------------------------

top_left, top_mid, top_right = st.columns([1.4, 1.2, 1.2])

with top_left:
    st.subheader("Clarity & Conflicts Over Time")
    tail = df.tail(150).set_index("tick")
    st.line_chart(tail[["clarity"]])
    st.area_chart(tail[["conflicts_now", "pred_future_conflicts"]])

    st.subheader("Clarity Distribution")
    st.bar_chart(df["clarity"].tail(120))

with top_mid:
    st.subheader("Synthetic Sector Map")

    flights_df = pd.DataFrame(st.session_state.flights)
    if not flights_df.empty:
        flights_df["color"] = [
            [255, 0, 0, 220] if latest["conflicts_now"] > 0 else [0, 200, 255, 220]
            for _ in range(len(flights_df))
        ]

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=flights_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=15000,
            pickable=True,
            get_elevation="alt_ft",
        )

        sector_layers = []
        active_sectors = [sid for sid, data in st.session_state.sectors.items() if data["active"]]
        
        for sector_id in active_sectors:
            bounds = get_sector_bounds(sector_id)
            is_selected = (sector_id == st.session_state.selected_sector)
            
            sector_poly = [
                [bounds["center_lon"] - bounds["lon_span"]/2, bounds["center_lat"] - bounds["lat_span"]/2],
                [bounds["center_lon"] + bounds["lon_span"]/2, bounds["center_lat"] - bounds["lat_span"]/2],
                [bounds["center_lon"] + bounds["lon_span"]/2, bounds["center_lat"] + bounds["lat_span"]/2],
                [bounds["center_lon"] - bounds["lon_span"]/2, bounds["center_lat"] + bounds["lat_span"]/2],
            ]
            
            boundary = pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": sector_poly}],
                get_polygon="polygon",
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[255, 255, 0, 255] if is_selected else [100, 100, 100, 150],
                line_width_min_pixels=2 if is_selected else 1,
            )
            sector_layers.append(boundary)
        
        focus_bounds = get_active_sector_bounds()
        view_state = pdk.ViewState(
            latitude=focus_bounds["center_lat"],
            longitude=focus_bounds["center_lon"],
            zoom=6,
            pitch=30,
        )

        deck = pdk.Deck(
            layers=sector_layers + [scatter],
            initial_view_state=view_state,
            tooltip={"text": "{callsign}\nALT {alt_ft} ft\nHDG {heading_deg}°"},
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        st.pydeck_chart(deck)
    else:
        st.info("No flights yet.")

    st.caption("Synthetic air picture. All positions and tracks are simulated.")

with top_right:
    st.subheader("Predictive Layer — Summary")
    st.write(
        f"- Horizon: **{st.session_state.prediction_horizon_min} min**\n"
        f"- Predicted conflicts in window: **{int(latest['pred_future_conflicts'])}**\n"
        f"- Worst predicted min separation: "
        f"**{latest['pred_worst_min_sep_nm']:.1f} nm**" if latest["pred_worst_min_sep_nm"] else
        "- Worst predicted min separation: **None inside sector**"
    )

    st.subheader("Conflict Table (Current)")
    conflicts_now = detect_current_conflicts(st.session_state.flights)
    if conflicts_now:
        cdf = pd.DataFrame(conflicts_now)
        cdf["pair"] = cdf["pair"].apply(lambda p: f"{p[0]} / {p[1]}")
        st.dataframe(cdf, use_container_width=True, height=140)
        
        st.markdown("**Resolution Strategies:**")
        strategies = generate_resolution_strategies(conflicts_now, st.session_state.flights)
        if strategies:
            for strategy in strategies[:2]:
                with st.expander(f"📋 {strategy['name']}", expanded=False):
                    st.write(strategy['description'])
                    if strategy['actions']:
                        st.markdown("**Recommended Actions:**")
                        for action in strategy['actions'][:3]:
                            st.write(f"• {action}")
    else:
        st.info("No separation infringements in current snapshot.")

    st.subheader("Clarity Interpretation")
    st.write(
        "- **90–100**: stable\n"
        "- **80–89**: elevated but comfortable\n"
        "- **65–79**: high load, watch closely\n"
        "- **< 65**: critical — slow down the system"
    )

st.markdown("---")

# ---------------------------
# Bottom: Proposals, Events, Telemetry
# ---------------------------

bottom_left, bottom_mid, bottom_right = st.columns([1.3, 1.0, 1.1])

with bottom_left:
    st.subheader("Human-Gated Proposals")

    if not st.session_state.proposals:
        st.info("No proposals yet. System is monitoring only.")
    else:
        labels = [
            f"#{p['id']} [{p['status']}] {p['title']}"
            for p in st.session_state.proposals
        ]
        selected_label = st.selectbox("Select proposal", options=labels)
        selected_id = int(selected_label.split(" ")[0].replace("#", ""))
        selected = next(p for p in st.session_state.proposals if p["id"] == selected_id)

        st.markdown(f"**Title:** {selected['title']}")
        st.markdown(f"**Status:** `{selected['status']}`")
        st.markdown("**Rationale:**")
        st.write(selected["rationale"])

        st.markdown("**Bounds & Guardrails:**")
        bounds = selected["bounds"]
        st.table(
            pd.DataFrame(
                [
                    ["Max arrival rate change", bounds["max_arrival_rate_change"]],
                    ["Max departure delay", bounds["max_departure_delay"]],
                    ["Automatic clearances", "DISABLED" if bounds["no_automatic_clearances"] else "ALLOWED"],
                ],
                columns=["Parameter", "Constraint"],
            )
        )

        st.markdown("**Snapshot at Proposal Time:**")
        st.json(selected["snapshot"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            approve_click = st.button("✅ Approve", disabled=(selected["status"] != "PENDING"))
        with col_b:
            reject_click = st.button("⛔ Reject", disabled=(selected["status"] != "PENDING"))
        with col_c:
            defer_click = st.button("🕒 Defer", disabled=(selected["status"] != "PENDING"))

        if approve_click and selected["status"] == "PENDING":
            if not st.session_state.gate_open:
                st.warning("Human gate is CLOSED. Open the gate to approve.")
                log_event("WARN", f"Attempted approval of proposal #{selected['id']} with gate closed.")
            else:
                selected["status"] = "APPROVED"
                log_event("APPROVED", f"Proposal #{selected['id']} approved by operator.")
        if reject_click and selected["status"] == "PENDING":
            selected["status"] = "REJECTED"
            log_event("REJECTED", f"Proposal #{selected['id']} rejected by operator.")
        if defer_click and selected["status"] == "PENDING":
            selected["status"] = "DEFERRED"
            log_event("INFO", f"Proposal #{selected['id']} deferred by operator.")

with bottom_mid:
    st.subheader("Event Feed")
    
    if st.session_state.playback_mode:
        st.caption("🔄 Playback Mode - Showing saved events")

    colf1, colf2 = st.columns(2)
    with colf1:
        level_filter = st.selectbox(
            "Level",
            ["ALL", "INFO", "WARN", "ALERT", "PROPOSAL", "APPROVED", "REJECTED"],
        )
    with colf2:
        search_text = st.text_input("Search")

    if st.session_state.playback_mode and st.session_state.playback_events:
        events = st.session_state.playback_events
    else:
        events = st.session_state.events
    
    if level_filter != "ALL":
        events = [e for e in events if e["level"] == level_filter]
    if search_text:
        low = search_text.lower()
        events = [e for e in events if low in e["msg"].lower()]

    if events:
        edf = pd.DataFrame(events[-80:])
        edf = edf.iloc[::-1]
        st.dataframe(edf, use_container_width=True, height=300)
    else:
        st.info("No events match current filters.")

with bottom_right:
    st.subheader("Telemetry (Last 50 Ticks)")
    cols = [
        "tick",
        "clarity",
        "state",
        "flight_count",
        "conflicts_now",
        "pred_future_conflicts",
        "workload_index",
        "comms_fraction",
    ]
    st.dataframe(
        df[cols].tail(50).set_index("tick"),
        use_container_width=True,
        height=260,
    )

    st.caption("All data synthetic. For research & demo only.")

st.markdown("---")

# ---------------------------
# Bayesian Confidence & Human-Gated Actions
# ---------------------------

bayes_col, action_col = st.columns([1, 1])

with bayes_col:
    st.subheader("🎯 Bayesian State Confidence")
    if st.session_state.bayesian_state:
        bayes_df = pd.DataFrame(
            {
                "State": list(st.session_state.bayesian_state.keys()),
                "Confidence (%)": [
                    round(v * 100, 1) 
                    for v in st.session_state.bayesian_state.values()
                ],
            }
        )
        st.bar_chart(bayes_df.set_index("State"), height=220)
        
        best_state = max(
            st.session_state.bayesian_state.items(),
            key=lambda x: x[1],
        )[0]
        best_conf = st.session_state.bayesian_state[best_state] * 100
        st.info(
            f"**Most likely condition:** {best_state} "
            f"({best_conf:.1f}% confidence)"
        )
    else:
        st.info("Bayesian confidence will appear once simulation starts.")

with action_col:
    st.subheader("🚦 Human-Gated Interventions")
    st.write("System will **never** act alone. Operator approval required.")
    
    action = st.radio(
        "Action:",
        [
            "Hold all departures",
            "Issue spacing instructions",
            "Request altitude separation",
            "Do nothing (monitor only)",
        ],
        key="selected_action",
    )
    
    confirm_action = st.button(
        "Confirm Action",
        type="primary",
        key="confirm_action",
    )
    
    if confirm_action:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        action_entry = {
            "timestamp": timestamp,
            "action": action,
            "sector": st.session_state.sector_id,
            "tick": st.session_state.tick if st.session_state.history else 0,
        }
        st.session_state.action_log.append(action_entry)
        log_event("ACTION", f"Human intervention: {action}")
        st.success(f"Action logged: {action} at {timestamp}")
        st.rerun()
    
    if st.session_state.action_log:
        with st.expander(
            f"📋 Action Log ({len(st.session_state.action_log)} entries)",
            expanded=False,
        ):
            recent_actions = st.session_state.action_log[-10:]
            for entry in reversed(recent_actions):
                st.text(
                    f"[{entry['timestamp']}] Tick {entry['tick']}: {entry['action']}"
                )

st.markdown("---")

# ---------------------------
# Data Export Section
# ---------------------------

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    st.subheader("📊 Export Data")
    
with export_col2:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download History (CSV)",
            data=csv_history,
            file_name=f"atc_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        json_history = history_df.to_json(orient='records', indent=2)
        st.download_button(
            label="⬇️ Download History (JSON)",
            data=json_history,
            file_name=f"atc_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    else:
        st.info("No history data to export yet.")

with export_col3:
    if st.session_state.events:
        events_df = pd.DataFrame(st.session_state.events)
        
        csv_events = events_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Events (CSV)",
            data=csv_events,
            file_name=f"atc_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        json_events = events_df.to_json(orient='records', indent=2)
        st.download_button(
            label="⬇️ Download Events (JSON)",
            data=json_events,
            file_name=f"atc_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    else:
        st.info("No event data to export yet.")

