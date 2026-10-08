# ============================================================
# POLY-II IV PREDICTION — STREAMLIT DEPLOYMENT APP
# V9.4 FROZEN CANDIDATE MODEL
# ============================================================
#
# USER WORKFLOW
# --------------
# 1. Keep the frozen V9.4 model package + Nova_Part-A.xlsx
#    beside this application.
# 2. Upload the same raw 15-minute POLY-II DCS Excel file
#    used by the Colour Prediction application.
# 3. Select date.
# 4. Select DCS time.
# 5. Click PREDICT IV.
#
# DEPLOYMENT PIPELINE
# -------------------
# RAW DCS
#   -> Nova inventory conversion
#   -> residence time
#   -> RT-valid production check
#   -> backward causal trajectory
#   -> V9 causal feature reconstruction
#   -> V9.3 transition-aware features
#   -> exact frozen V9.4 feature order
#   -> frozen HistGradientBoosting model
#   -> IV prediction
#
# IMPORTANT
# ----------
# * No actual laboratory IV is used at prediction time.
# * No previous laboratory IV is used.
# * September is not special-cased or used for prediction.
# * The model is NEVER refitted in this application.
# * Prediction is refused if required frozen features cannot be reproduced.
# * NaN/inf values are converted to 0.0 only immediately before
#   the frozen model receives the matrix.
#
# EXPECTED APPLICATION FOLDER
# ---------------------------
# app_iv.py
# Nova_Part-A.xlsx
#
# V94_FROZEN_PRODUCTION_MODEL/
#   v94_candidate_model.joblib
#   v94_model_feature_order.json
#   v94_selected_base_features.json
#   v94_engineered_features.json
#   v94_reference_parameters.json
#   v94_ood_reference_statistics.csv
#   v94_feature_schema.csv
#   v94_frozen_manifest.json
#
# Optional:
#   v94_reload_verification.csv
#   build_transition_features_source.py
#
# ============================================================

from pathlib import Path
from io import BytesIO
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="POLY-II IV Prediction",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# APPLICATION PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NOVA_FILE = BASE_DIR / "Nova_Part-A.xlsx"

DEFAULT_MODEL_DIR = (
    BASE_DIR / "V94_FROZEN_PRODUCTION_MODEL"
)

# Allow the user to point to another folder without changing code.
with st.sidebar:
    st.header("IV Model Configuration")

    model_dir_text = st.text_input(
        "Frozen model folder",
        value=str(DEFAULT_MODEL_DIR),
        help=(
            "Folder containing the frozen V9.4 model artifacts. "
            "Use the default folder if it is beside app_iv.py."
        ),
    )

MODEL_DIR = Path(model_dir_text).expanduser()

MODEL_FILE = MODEL_DIR / "v94_candidate_model.joblib"
FEATURE_ORDER_FILE = MODEL_DIR / "v94_model_feature_order.json"
BASE_FEATURE_FILE = MODEL_DIR / "v94_selected_base_features.json"
ENGINEERED_FEATURE_FILE = MODEL_DIR / "v94_engineered_features.json"
REFERENCE_PARAM_FILE = MODEL_DIR / "v94_reference_parameters.json"
OOD_STATS_FILE = MODEL_DIR / "v94_ood_reference_statistics.csv"
SCHEMA_FILE = MODEL_DIR / "v94_feature_schema.csv"
MANIFEST_FILE = MODEL_DIR / "v94_frozen_manifest.json"


# ============================================================
# FROZEN V9.4 CONSTANTS
# ============================================================

IV_ACCEPTANCE_THRESHOLD = 0.008
PREDICTION_JUMP_LIMIT = 0.008
OOD_CAUTION_Z = 2.5
OOD_OUT_Z = 3.5

DYNAMIC_LOOKBACK_HOURS = 4.0
MAX_RT_HISTORY_GAP_HOURS = 1.0

# Same transition tags used by V9.3.
TRANSITION_TAGS = [
    "IC-17013",
    "PIC-15024",
    "TIC-13012",
    "TPD",
    "TI-17020",
    "PIC-12013",
]

# Same production DCS tag family used by the existing POLY-II
# Colour application, with IV-specific feature reconstruction.
DCS_SENSOR_COLS = [
    "YK-11001",
    "TPD",
    "FIC-12001",
    "LIC-12019",
    "TIC-12012",
    "PIC-12013",
    "LIC-13017",
    "TIC-13012",
    "LIC-14003",
    "TIC-14004",
    "PIC-14005",
    "LIC-15006",
    "TIC-15005",
    "TIC-15008",
    "PIC-15024",
    "SIK-17015",
    "SIK-17028",
    "IC-17013",
    "PIC-17024",
    "PV-17024",
    "LI-17017",
    "LI-17023",
    "LIC-17016",
    "TI-17019",
    "TI-17020",
    "TI-17022",
    "VIC-18020",
    "TI-17113",
    "TI-17114",
    "PIC-17183",
    "LIC-17175",
    "TI-50002",
    "TI-50003",
    "FIC-21009",
    "YK-21010",
    "FIC-41007",
    "YK-41008",
    "FIC-43007",
    "YK-43008",
    "FIC-42007",
    "YK-42008",
]


# ============================================================
# SECTION / NOVA CONFIGURATION
# ============================================================

EQUIPMENT_MAP = {
    "EST1": {
        "DCS_TAG": "LIC-12019",
        "EQUIPMENT": "12-R01-EST-I",
    },
    "EST2": {
        "DCS_TAG": "LIC-13017",
        "EQUIPMENT": "13-R01-EST-II",
    },
    "PP1": {
        "DCS_TAG": "LIC-14003",
        "EQUIPMENT": "14-R01-PP-I",
    },
    "PP2": {
        "DCS_TAG": "LIC-15006",
        "EQUIPMENT": "15-R01-PP-II",
    },
    "DRR": {
        "DCS_TAG": "LIC-17016",
        "EQUIPMENT": "17-R01- (DRR)",
    },
}

SECTION_ORDER = [
    ("DRR", "DRR_RT_HR"),
    ("PP2", "PP2_RT_HR"),
    ("PP1", "PP1_RT_HR"),
    ("EST2", "EST2_RT_HR"),
    ("EST1", "EST1_RT_HR"),
]

SECTION_TAGS = {
    "EST1": [
        "FIC-12001",
        "LIC-12019",
        "TIC-12012",
        "PIC-12013",
    ],
    "EST2": [
        "LIC-13017",
        "TIC-13012",
    ],
    "PP1": [
        "LIC-14003",
        "TIC-14004",
        "PIC-14005",
    ],
    "PP2": [
        "LIC-15006",
        "TIC-15005",
        "TIC-15008",
        "PIC-15024",
    ],
    "DRR": [
        "SIK-17015",
        "SIK-17028",
        "IC-17013",
        "PIC-17024",
        "PV-17024",
        "LI-17017",
        "LI-17023",
        "LIC-17016",
        "TI-17019",
        "TI-17020",
        "TI-17022",
        "VIC-18020",
    ],
}

AREA_TAGS = {
    "PASTE": ["YK-11001"],
    "THROUGHPUT": ["TPD"],
    "JET": [
        "TI-17113",
        "TI-17114",
        "PIC-17183",
        "LIC-17175",
    ],
    "HTM": [
        "TI-50002",
        "TI-50003",
    ],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_center_scale(series):
    x = pd.to_numeric(series, errors="coerce")
    med = float(x.median())
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-12:
        iqr = 1.0
    return med, iqr


def robust_z(series, med, iqr):
    x = pd.to_numeric(series, errors="coerce")
    z = (x - med) / iqr
    return z.clip(-8, 8)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FROZEN MODEL LOADING
# ============================================================

@st.cache_resource
def load_frozen_model(model_dir_string):
    model_dir = Path(model_dir_string)

    required = {
        "model": model_dir / "v94_candidate_model.joblib",
        "feature_order": model_dir / "v94_model_feature_order.json",
        "base_features": model_dir / "v94_selected_base_features.json",
        "engineered_features": model_dir / "v94_engineered_features.json",
        "reference_parameters": model_dir / "v94_reference_parameters.json",
        "ood_stats": model_dir / "v94_ood_reference_statistics.csv",
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Frozen V9.4 model package is incomplete.\n\n"
            + "\n".join(missing)
        )

    model = joblib.load(required["model"])
    feature_order = load_json(required["feature_order"])
    base_features = load_json(required["base_features"])
    engineered_features = load_json(required["engineered_features"])
    reference_parameters = load_json(required["reference_parameters"])
    ood_stats = pd.read_csv(required["ood_stats"])

    if isinstance(feature_order, dict):
        # Defensive support for {"features": [...]}
        if "features" in feature_order:
            feature_order = feature_order["features"]
        elif "MODEL_FEATURES" in feature_order:
            feature_order = feature_order["MODEL_FEATURES"]
        else:
            raise ValueError(
                "v94_model_feature_order.json has an unsupported structure."
            )

    feature_order = list(feature_order)

    return (
        model,
        feature_order,
        list(base_features),
        list(engineered_features),
        reference_parameters,
        ood_stats,
    )


# ============================================================
# RAW DCS LOADING
# ============================================================

@st.cache_data
def load_raw_dcs(uploaded_bytes):
    if not uploaded_bytes:
        raise ValueError("Please upload the raw DCS Excel file.")

    raw = pd.read_excel(
        BytesIO(uploaded_bytes),
        header=None,
    )

    if len(raw) < 6:
        raise ValueError(
            "Unexpected DCS Excel structure. "
            "Expected timestamp/tag rows followed by DCS data."
        )

    # Existing POLY-II Colour app uses row 2 for tag names.
    tag_row = raw.iloc[2]

    columns = []

    for i, value in enumerate(tag_row):
        if i == 0:
            columns.append("TIMESTAMP")
        elif pd.isna(value):
            columns.append(f"UNNAMED_{i}")
        else:
            columns.append(str(value).strip())

    dcs = raw.iloc[5:].copy()
    dcs.columns = columns

    dcs = (
        dcs
        .dropna(how="all")
        .reset_index(drop=True)
    )

    dcs["TIMESTAMP"] = pd.to_datetime(
        dcs["TIMESTAMP"],
        errors="coerce",
    )

    dcs = (
        dcs
        .dropna(subset=["TIMESTAMP"])
        .sort_values("TIMESTAMP")
        .drop_duplicates("TIMESTAMP")
        .reset_index(drop=True)
    )

    missing = [
        c for c in DCS_SENSOR_COLS
        if c not in dcs.columns
    ]

    if missing:
        raise ValueError(
            "Required DCS tags are missing:\n\n"
            + "\n".join(missing)
        )

    for col in DCS_SENSOR_COLS:
        dcs[col] = pd.to_numeric(
            dcs[col],
            errors="coerce",
        )

    # Existing application protects against the FLOAT32 invalid sentinel.
    if "YK-43008" in dcs.columns:
        sentinel = dcs["YK-43008"].abs() > 1e30
        dcs.loc[sentinel, "YK-43008"] = np.nan

    dcs["INTERVAL_MIN"] = (
        dcs["TIMESTAMP"].diff()
        .dt.total_seconds()
        / 60.0
    )

    return dcs


# ============================================================
# NOVA INVENTORY
# ============================================================

@st.cache_data
def load_nova_curves():
    if not NOVA_FILE.exists():
        raise FileNotFoundError(
            f"Nova file not found:\n{NOVA_FILE}"
        )

    nova = pd.read_excel(
        NOVA_FILE,
        sheet_name="Poly-2",
        header=1,
    )

    curves = {}

    for section, info in EQUIPMENT_MAP.items():

        required = [
            "silo_no",
            "Level Percent",
            "Level Weight Kgs.",
        ]
        missing = [c for c in required if c not in nova.columns]

        if missing:
            raise ValueError(
                "Nova_Part-A.xlsx is missing columns: "
                + ", ".join(missing)
            )

        mask = (
            nova["silo_no"]
            .astype(str)
            .str.strip()
            .eq(info["EQUIPMENT"])
        )

        curve = nova.loc[
            mask,
            [
                "Level Percent",
                "Level Weight Kgs.",
            ],
        ].copy()

        curve["Level Percent"] = pd.to_numeric(
            curve["Level Percent"],
            errors="coerce",
        )

        curve["Level Weight Kgs."] = pd.to_numeric(
            curve["Level Weight Kgs."],
            errors="coerce",
        )

        curve = (
            curve
            .dropna()
            .drop_duplicates("Level Percent")
            .sort_values("Level Percent")
            .reset_index(drop=True)
        )

        if len(curve) < 2:
            raise ValueError(
                f"Insufficient Nova curve for {section}."
            )

        curves[section] = curve

    return curves


def add_inventory(dcs, curves):
    out = dcs.copy()

    for section, info in EQUIPMENT_MAP.items():

        level = pd.to_numeric(
            out[info["DCS_TAG"]],
            errors="coerce",
        )

        curve = curves[section]

        x = curve["Level Percent"].to_numpy(float)
        y = curve["Level Weight Kgs."].to_numpy(float)

        inventory = np.full(
            len(out),
            np.nan,
        )

        valid = (
            level.notna()
            & level.ge(x.min())
            & level.le(x.max())
        )

        inventory[valid] = np.interp(
            level.loc[valid].to_numpy(float),
            x,
            y,
        )

        out[f"{section}_LEVEL_PCT"] = level
        out[f"{section}_INVENTORY_KG"] = inventory

    return out


# ============================================================
# RESIDENCE TIME
# ============================================================

def calculate_rt(dcs):
    out = dcs.copy()

    out["TPD"] = pd.to_numeric(
        out["TPD"],
        errors="coerce",
    )

    out["TPH"] = out["TPD"] / 24.0

    for section in EQUIPMENT_MAP:
        out[f"{section}_RT_HR"] = (
            out[f"{section}_INVENTORY_KG"]
            / (out["TPH"] * 1000.0)
        )

        out.loc[
            out["TPH"] <= 0,
            f"{section}_RT_HR",
        ] = np.nan

    rt_cols = [
        "EST1_RT_HR",
        "EST2_RT_HR",
        "PP1_RT_HR",
        "PP2_RT_HR",
        "DRR_RT_HR",
    ]

    out["TOTAL_RT_HR"] = out[rt_cols].sum(
        axis=1,
        min_count=5,
    )

    # Same practical production validity rule used by the
    # existing POLY-II deployment architecture.
    out["RT_VALID"] = (
        out["TPD"].ge(375)
        & out[rt_cols].notna().all(axis=1)
    )

    out["RT_STATUS"] = np.where(
        out["RT_VALID"],
        "VALID",
        np.where(
            out["TPD"].le(0),
            "SHUTDOWN_TPD_LE_0",
            "NOT_RT_VALID",
        ),
    )

    return out


def build_runs(dcs):
    gap = (
        dcs["TIMESTAMP"].diff()
        .dt.total_seconds()
        / 3600.0
    )

    continuous = gap.eq(0.25)

    run_break = (
        dcs["RT_VALID"].ne(
            dcs["RT_VALID"].shift()
        )
        | ~continuous
    )

    out = dcs.copy()
    out["VALID_RUN_ID"] = run_break.cumsum()

    runs = (
        out.loc[out["RT_VALID"]]
        .groupby("VALID_RUN_ID")
        .agg(
            START_TS=("TIMESTAMP", "min"),
            END_TS=("TIMESTAMP", "max"),
            N_ROWS=("TIMESTAMP", "size"),
        )
        .reset_index()
    )

    runs["DURATION_HR"] = (
        runs["END_TS"] - runs["START_TS"]
    ).dt.total_seconds() / 3600.0

    return out, runs


# ============================================================
# CAUSAL TRAJECTORY
# ============================================================

def find_run(runs, timestamp):
    match = runs[
        (runs["START_TS"] <= timestamp)
        & (runs["END_TS"] >= timestamp)
    ]

    if match.empty:
        return None

    return match.iloc[0]


def get_rt_at_time(dcs, timestamp, rt_column):
    timestamp = pd.Timestamp(timestamp)

    if (
        timestamp < dcs["TIMESTAMP"].min()
        or timestamp > dcs["TIMESTAMP"].max()
    ):
        return np.nan, "OUTSIDE_RANGE"

    before = dcs[
        (dcs["TIMESTAMP"] <= timestamp)
        & dcs["RT_VALID"]
    ].tail(1)

    after = dcs[
        (dcs["TIMESTAMP"] >= timestamp)
        & dcs["RT_VALID"]
    ].head(1)

    if before.empty or after.empty:
        return np.nan, "NO_BRACKET"

    t0 = before.iloc[0]["TIMESTAMP"]
    t1 = after.iloc[0]["TIMESTAMP"]

    gap_hr = (
        t1 - t0
    ).total_seconds() / 3600.0

    if gap_hr > MAX_RT_HISTORY_GAP_HOURS:
        return np.nan, "RT_HISTORY_GAP"

    r0 = before.iloc[0][rt_column]
    r1 = after.iloc[0][rt_column]

    if pd.isna(r0) or pd.isna(r1):
        return np.nan, "RT_MISSING"

    if t0 == t1:
        return float(r0), "DIRECT"

    fraction = (
        timestamp - t0
    ).total_seconds() / (
        t1 - t0
    ).total_seconds()

    return (
        float(r0 + fraction * (r1 - r0)),
        "INTERPOLATED",
    )


def calculate_trajectory(dcs, runs, timestamp):
    timestamp = pd.Timestamp(timestamp)

    run = find_run(runs, timestamp)

    if run is None:
        return None

    current = timestamp

    result = {
        "TIMESTAMP": timestamp,
        "TRAJECTORY_VALID": True,
        "VALID_RUN_ID": int(run["VALID_RUN_ID"]),
        "RUN_START": run["START_TS"],
        "RUN_END": run["END_TS"],
    }

    for section, rt_col in SECTION_ORDER:

        rt, status = get_rt_at_time(
            dcs,
            current,
            rt_col,
        )

        if pd.isna(rt):
            return None

        result[f"{section}_RT_HR"] = rt
        result[f"{section}_RT_STATUS"] = status

        boundary = (
            current
            - pd.Timedelta(hours=float(rt))
        )

        result[f"{section}_BOUNDARY_TS"] = boundary

        current = boundary

    result["EST1_UPSTREAM_TS"] = current

    boundaries = [
        result["DRR_BOUNDARY_TS"],
        result["PP2_BOUNDARY_TS"],
        result["PP1_BOUNDARY_TS"],
        result["EST2_BOUNDARY_TS"],
        result["EST1_BOUNDARY_TS"],
    ]

    if not all(
        run["START_TS"] <= x <= run["END_TS"]
        for x in boundaries
    ):
        return None

    return result


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

WINDOW_STATS = [
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
]


def extract_window_features(
    dcs,
    start_ts,
    end_ts,
    tags,
    prefix,
):
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    if (
        pd.isna(start_ts)
        or pd.isna(end_ts)
        or end_ts <= start_ts
    ):
        return {}

    window = dcs[
        (dcs["TIMESTAMP"] >= start_ts)
        & (dcs["TIMESTAMP"] <= end_ts)
    ].sort_values("TIMESTAMP").copy()

    if window.empty:
        return {}

    duration_hr = (
        end_ts - start_ts
    ).total_seconds() / 3600.0

    result = {}

    for tag in tags:

        if tag not in window.columns:
            continue

        values = pd.to_numeric(
            window[tag],
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() == 0:
            continue

        v = values.loc[valid].astype(float)

        first = float(v.iloc[0])
        last = float(v.iloc[-1])

        result[f"{prefix}{tag}_MEAN"] = float(v.mean())
        result[f"{prefix}{tag}_STD"] = float(v.std(ddof=0))
        result[f"{prefix}{tag}_MIN"] = float(v.min())
        result[f"{prefix}{tag}_MAX"] = float(v.max())
        result[f"{prefix}{tag}_RANGE"] = float(v.max() - v.min())
        result[f"{prefix}{tag}_FIRST"] = first
        result[f"{prefix}{tag}_LAST"] = last
        result[f"{prefix}{tag}_DELTA"] = last - first
        result[f"{prefix}{tag}_N"] = int(valid.sum())

        for q in WINDOW_STATS:
            result[
                f"{prefix}{tag}_P{int(q * 100)}"
            ] = float(v.quantile(q))

        if len(v) >= 2 and duration_hr > 0:

            times = (
                window.loc[valid, "TIMESTAMP"]
                - window.loc[valid, "TIMESTAMP"].iloc[0]
            ).dt.total_seconds().to_numpy() / 3600.0

            y = v.to_numpy(float)

            if len(np.unique(times)) >= 2:
                slope = np.polyfit(
                    times,
                    y,
                    1,
                )[0]
            else:
                slope = 0.0
        else:
            slope = 0.0

        result[f"{prefix}{tag}_SLOPE"] = float(slope)

    return result


# ============================================================
# BASE V9 CAUSAL FEATURES
# ============================================================

def build_v9_base_features(
    trajectory,
    dcs,
):
    """
    Reconstructs the deployment-side V9 feature family.

    RTWIN_*:
        Full causal residence-time trajectory:
        EST-I upstream boundary -> selected prediction timestamp.

    END30_*:
        Most recent 30 minutes ending at the selected timestamp.

    Section windows:
        EST1, EST2, PP1, PP2, DRR.

    The frozen feature-order artifact remains the final authority.
    Any feature that the frozen model requires but this function
    cannot reproduce causes a hard prediction stop.
    """

    ts = pd.Timestamp(
        trajectory["TIMESTAMP"]
    )

    est1 = pd.Timestamp(
        trajectory["EST1_BOUNDARY_TS"]
    )
    est2 = pd.Timestamp(
        trajectory["EST2_BOUNDARY_TS"]
    )
    pp1 = pd.Timestamp(
        trajectory["PP1_BOUNDARY_TS"]
    )
    pp2 = pd.Timestamp(
        trajectory["PP2_BOUNDARY_TS"]
    )
    drr = pd.Timestamp(
        trajectory["DRR_BOUNDARY_TS"]
    )

    row = {
        "LAB_TIMESTAMP": ts,

        "EST1_RT_HR_TRAJ": trajectory["EST1_RT_HR"],
        "EST2_RT_HR_TRAJ": trajectory["EST2_RT_HR"],
        "PP1_RT_HR_TRAJ": trajectory["PP1_RT_HR"],
        "PP2_RT_HR_TRAJ": trajectory["PP2_RT_HR"],
        "DRR_RT_HR_TRAJ": trajectory["DRR_RT_HR"],

        "TOTAL_RT_HR_TRAJ": sum(
            trajectory[x]
            for x in [
                "EST1_RT_HR",
                "EST2_RT_HR",
                "PP1_RT_HR",
                "PP2_RT_HR",
                "DRR_RT_HR",
            ]
        ),

        "TPD": np.nan,
    }

    # --------------------------------------------------------
    # Selected-time process values
    # --------------------------------------------------------

    selected = dcs[
        dcs["TIMESTAMP"] <= ts
    ].tail(1)

    if not selected.empty:
        for tag in DCS_SENSOR_COLS:
            if tag in selected.columns:
                row[tag] = selected.iloc[0][tag]

    # --------------------------------------------------------
    # Section windows
    # --------------------------------------------------------

    section_windows = {
        "EST1": (est1, est2),
        "EST2": (est2, pp1),
        "PP1": (pp1, pp2),
        "PP2": (pp2, drr),
        "DRR": (drr, ts),
    }

    for section, (start, end) in section_windows.items():

        row.update(
            extract_window_features(
                dcs,
                start,
                end,
                SECTION_TAGS[section],
                f"{section}_",
            )
        )

        row[f"{section}_WINDOW_START"] = start
        row[f"{section}_WINDOW_END"] = end
        row[f"{section}_WINDOW_HR"] = (
            end - start
        ).total_seconds() / 3600.0

        row[f"{section}_DCS_N"] = len(
            dcs[
                (dcs["TIMESTAMP"] >= start)
                & (dcs["TIMESTAMP"] <= end)
            ]
        )

    # --------------------------------------------------------
    # Full causal RT window
    # --------------------------------------------------------
    # IMPORTANT:
    # V9.0 did NOT represent the two DRR relationships as ordinary
    # tag statistics. They were explicitly engineered from:
    #
    #   DRR level difference   = LI-17017 - LI-17023
    #   DRR agitator difference = SIK-17015 - SIK-17028
    #
    # These eight features are part of the frozen 40-feature V9/V9.3
    # base feature definition and MUST be reconstructed exactly.

    global_start = est1
    global_end = ts

    all_process_tags = list(
        dict.fromkeys(
            DCS_SENSOR_COLS
        )
    )

    full_window = dcs[
        (dcs["TIMESTAMP"] >= global_start)
        & (dcs["TIMESTAMP"] <= global_end)
    ].copy()

    recent_start = ts - pd.Timedelta(minutes=30)

    recent_window = dcs[
        (dcs["TIMESTAMP"] >= recent_start)
        & (dcs["TIMESTAMP"] <= ts)
    ].copy()

    row.update(
        extract_window_features(
            dcs,
            global_start,
            global_end,
            all_process_tags,
            "RTWIN_",
        )
    )

    # Exact frozen V9 DRR level relationship.
    if (
        "LI-17017" in full_window.columns
        and "LI-17023" in full_window.columns
    ):
        full_diff = (
            pd.to_numeric(
                full_window["LI-17017"],
                errors="coerce",
            )
            - pd.to_numeric(
                full_window["LI-17023"],
                errors="coerce",
            )
        )

        recent_diff = (
            pd.to_numeric(
                recent_window["LI-17017"],
                errors="coerce",
            )
            - pd.to_numeric(
                recent_window["LI-17023"],
                errors="coerce",
            )
        )

        full_valid = full_diff.dropna()
        recent_valid = recent_diff.dropna()

        row["RTWIN_DRR_LEVEL_DIFFERENCE_MEAN"] = (
            float(full_diff.mean())
        )
        row["RTWIN_DRR_LEVEL_DIFFERENCE_LAST"] = (
            float(full_valid.iloc[-1])
            if len(full_valid)
            else np.nan
        )
        row["END30_DRR_LEVEL_DIFFERENCE_MEAN"] = (
            float(recent_diff.mean())
        )
        row["END30_DRR_LEVEL_DIFFERENCE_LAST"] = (
            float(recent_valid.iloc[-1])
            if len(recent_valid)
            else np.nan
        )

    # Exact frozen V9 DRR agitator relationship.
    if (
        "SIK-17015" in full_window.columns
        and "SIK-17028" in full_window.columns
    ):
        full_diff = (
            pd.to_numeric(
                full_window["SIK-17015"],
                errors="coerce",
            )
            - pd.to_numeric(
                full_window["SIK-17028"],
                errors="coerce",
            )
        )

        recent_diff = (
            pd.to_numeric(
                recent_window["SIK-17015"],
                errors="coerce",
            )
            - pd.to_numeric(
                recent_window["SIK-17028"],
                errors="coerce",
            )
        )

        full_valid = full_diff.dropna()
        recent_valid = recent_diff.dropna()

        row["RTWIN_DRR_AGITATOR_DIFF_MEAN"] = (
            float(full_diff.mean())
        )
        row["RTWIN_DRR_AGITATOR_DIFF_LAST"] = (
            float(full_valid.iloc[-1])
            if len(full_valid)
            else np.nan
        )
        row["END30_DRR_AGITATOR_DIFF_MEAN"] = (
            float(recent_diff.mean())
        )
        row["END30_DRR_AGITATOR_DIFF_LAST"] = (
            float(recent_valid.iloc[-1])
            if len(recent_valid)
            else np.nan
        )

    # --------------------------------------------------------
    # Recent 30-minute window
    # --------------------------------------------------------

    end30_start = ts - pd.Timedelta(minutes=30)

    row.update(
        extract_window_features(
            dcs,
            end30_start,
            ts,
            all_process_tags,
            "END30_",
        )
    )

    # --------------------------------------------------------
    # Additional short causal windows commonly used in V9
    # --------------------------------------------------------

    for minutes in [60, 120, 240]:
        row.update(
            extract_window_features(
                dcs,
                ts - pd.Timedelta(minutes=minutes),
                ts,
                all_process_tags,
                f"END{minutes}_",
            )
        )

    # --------------------------------------------------------
    # Process-level aliases
    # --------------------------------------------------------

    process_tags = [
        "TPD",
        "TIC-12012",
        "PIC-12013",
        "TIC-13012",
        "TIC-14004",
        "PIC-14005",
        "TIC-15005",
        "TIC-15008",
        "PIC-15024",
        "TI-17019",
        "TI-17020",
        "TI-17022",
        "PIC-17024",
    ]

    for tag in process_tags:
        clean = tag.replace("-", "_")

        for prefix in [
            "RTWIN_",
            "END30_",
        ]:
            source = f"{prefix}{tag}"

            mean_col = f"{source}_MEAN"
            last_col = f"{source}_LAST"
            first_col = f"{source}_FIRST"
            delta_col = f"{source}_DELTA"

            if mean_col in row:
                row[f"{prefix}{clean}_MEAN"] = row[mean_col]
            if last_col in row:
                row[f"{prefix}{clean}_LAST"] = row[last_col]
            if first_col in row:
                row[f"{prefix}{clean}_FIRST"] = row[first_col]
            if delta_col in row:
                row[f"{prefix}{clean}_DELTA"] = row[delta_col]

    return row


# ============================================================
# V9.3 TRANSITION FEATURES
# ============================================================

def build_transition_features(
    df,
    fit_reference,
):
    out = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # A. Individual process movement
    # --------------------------------------------------------

    for tag in TRANSITION_TAGS:

        source_pairs = [
            f"RTWIN_{tag}_DELTA",
            f"RTWIN_{tag}_SLOPE",
            f"END30_{tag}_DELTA",
            f"END30_{tag}_SLOPE",
        ]

        for source in source_pairs:
            if source in df.columns:
                out[
                    f"V93_ABS_{source}"
                ] = pd.to_numeric(
                    df[source],
                    errors="coerce",
                ).abs()

    # --------------------------------------------------------
    # B. Recent-vs-RT state displacement
    # --------------------------------------------------------

    shift_cols = []

    for tag in TRANSITION_TAGS:

        rt = f"RTWIN_{tag}_MEAN"
        recent = f"END30_{tag}_MEAN"

        if (
            rt in df.columns
            and recent in df.columns
        ):
            name = f"V93_STATE_SHIFT_{tag}"

            out[name] = (
                pd.to_numeric(
                    df[recent],
                    errors="coerce",
                )
                - pd.to_numeric(
                    df[rt],
                    errors="coerce",
                )
            )

            shift_cols.append(name)

    # --------------------------------------------------------
    # C. Robust movement scaling
    # --------------------------------------------------------

    z_cols = []

    for c in list(out.columns):

        if not c.startswith("V93_ABS_"):
            continue

        source = c.replace(
            "V93_ABS_",
            "",
            1,
        )

        if source not in fit_reference.columns:
            continue

        ref_abs = pd.to_numeric(
            fit_reference[source],
            errors="coerce",
        ).abs()

        med, iqr = robust_center_scale(
            ref_abs
        )

        z = robust_z(
            out[c],
            med,
            iqr,
        )

        zname = c.replace(
            "V93_ABS_",
            "V93_ROBUST_MOVE_",
            1,
        )

        out[zname] = z
        z_cols.append(zname)

    for c in shift_cols:

        tag = c.replace(
            "V93_STATE_SHIFT_",
            "",
            1,
        )

        rt = f"RTWIN_{tag}_MEAN"
        recent = f"END30_{tag}_MEAN"

        if (
            rt not in fit_reference.columns
            or recent not in fit_reference.columns
        ):
            continue

        ref_shift = (
            pd.to_numeric(
                fit_reference[recent],
                errors="coerce",
            )
            - pd.to_numeric(
                fit_reference[rt],
                errors="coerce",
            )
        )

        med, iqr = robust_center_scale(
            ref_shift
        )

        zname = (
            f"V93_ROBUST_STATE_SHIFT_{tag}"
        )

        out[zname] = robust_z(
            out[c],
            med,
            iqr,
        )

        z_cols.append(zname)

    # --------------------------------------------------------
    # D. Composite transition intensity
    # --------------------------------------------------------

    if z_cols:

        zmat = out[z_cols].abs()

        out[
            "V93_PROCESS_TRANSITION_SCORE"
        ] = zmat.mean(
            axis=1,
            skipna=True,
        )

        out[
            "V93_MAX_TRANSITION_Z"
        ] = zmat.max(
            axis=1,
            skipna=True,
        )

        high_count = pd.Series(
            0.0,
            index=df.index,
        )

        valid_count = pd.Series(
            0.0,
            index=df.index,
        )

        for c in z_cols:

            valid = out[c].notna()

            high_count.loc[valid] += (
                out.loc[
                    valid,
                    c,
                ].abs() >= 1.0
            ).astype(float)

            valid_count.loc[valid] += 1.0

        out[
            "V93_N_HIGH_TRANSITIONS"
        ] = high_count

        out[
            "V93_FRACTION_HIGH_TRANSITIONS"
        ] = (
            high_count
            / valid_count.replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # E. Residence-time anomaly
    # --------------------------------------------------------

    rt_cols = [
        c for c in [
            "TOTAL_RT_HR_TRAJ",
            "DRR_RT_HR_TRAJ",
            "PP2_RT_HR_TRAJ",
            "PP1_RT_HR_TRAJ",
            "EST2_RT_HR_TRAJ",
            "EST1_RT_HR_TRAJ",
        ]
        if (
            c in df.columns
            and c in fit_reference.columns
        )
    ]

    rt_z_cols = []

    for c in rt_cols:

        ref = pd.to_numeric(
            fit_reference[c],
            errors="coerce",
        )

        med, iqr = robust_center_scale(
            ref
        )

        zname = (
            "V93_RT_ABS_Z_"
            + c.replace(
                "_HR_TRAJ",
                "",
            )
        )

        out[zname] = robust_z(
            pd.to_numeric(
                df[c],
                errors="coerce",
            ),
            med,
            iqr,
        ).abs()

        rt_z_cols.append(zname)

    if rt_z_cols:

        out[
            "V93_RT_ANOMALY_SCORE"
        ] = out[
            rt_z_cols
        ].mean(
            axis=1,
            skipna=True,
        )

        out[
            "V93_MAX_RT_ANOMALY"
        ] = out[
            rt_z_cols
        ].max(
            axis=1,
            skipna=True,
        )

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================
# FEATURE REFERENCE
# ============================================================

def build_reference_from_frozen_stats(
    feature_names,
    ood_stats,
):
    """
    Converts the frozen V9.4 development statistics into a
    one-row reference dataframe used only for OOD detection.
    """

    ref = {}

    # The OOD file normally contains FEATURE, MEDIAN, Q1, Q3, IQR.
    if "FEATURE" in ood_stats.columns:
        feature_col = "FEATURE"
    elif "feature" in ood_stats.columns:
        feature_col = "feature"
    else:
        feature_col = ood_stats.columns[0]

    lookup = ood_stats.set_index(
        feature_col,
        drop=False,
    )

    for feature in feature_names:

        if feature not in lookup.index:
            continue

        row = lookup.loc[feature]

        # Store median only; IQR is handled separately by
        # the direct OOD function below.
        ref[feature] = row.get(
            "MEDIAN",
            row.get(
                "median",
                np.nan,
            ),
        )

    return pd.DataFrame([ref])


def calculate_ood_status(
    X,
    ood_stats,
    feature_names,
):
    """
    Development-frozen robust median/IQR support check.

    This is diagnostic. It is NOT an automatic prediction
    suppression rule, matching V9.4's industrial validation.
    """

    if ood_stats is None or ood_stats.empty:
        return (
            "UNAVAILABLE",
            np.nan,
            np.nan,
        )

    feature_col = None

    for candidate in [
        "FEATURE",
        "feature",
        "COLUMN",
        "column",
    ]:
        if candidate in ood_stats.columns:
            feature_col = candidate
            break

    if feature_col is None:
        feature_col = ood_stats.columns[0]

    stats = ood_stats.copy()
    stats[feature_col] = stats[
        feature_col
    ].astype(str)

    stats = stats.set_index(
        feature_col
    )

    max_z = 0.0
    z_values = []

    for feature in feature_names:

        if feature not in X.columns:
            continue

        if feature not in stats.index:
            continue

        value = pd.to_numeric(
            X.iloc[0][feature],
            errors="coerce",
        )

        if pd.isna(value):
            continue

        row = stats.loc[feature]

        med = row.get(
            "MEDIAN",
            row.get(
                "median",
                np.nan,
            ),
        )

        iqr = row.get(
            "IQR",
            row.get(
                "iqr",
                np.nan,
            ),
        )

        if pd.isna(iqr):

            q1 = row.get(
                "Q1",
                row.get(
                    "q1",
                    np.nan,
                ),
            )

            q3 = row.get(
                "Q3",
                row.get(
                    "q3",
                    np.nan,
                ),
            )

            if (
                pd.notna(q1)
                and pd.notna(q3)
            ):
                iqr = q3 - q1

        if (
            pd.isna(med)
            or pd.isna(iqr)
        ):
            continue

        iqr = float(iqr)

        if not np.isfinite(iqr) or iqr < 1e-12:
            iqr = 1.0

        z = abs(
            (float(value) - float(med))
            / iqr
        )

        z_values.append(z)

    if not z_values:
        return (
            "UNAVAILABLE",
            np.nan,
            np.nan,
        )

    max_z = float(np.max(z_values))
    frac_caution = float(
        np.mean(
            np.asarray(z_values)
            >= OOD_CAUTION_Z
        )
    )

    if max_z >= OOD_OUT_Z:
        status = "OUT_OF_SUPPORT"
    elif max_z >= OOD_CAUTION_Z:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return (
        status,
        max_z,
        frac_caution,
    )


# ============================================================
# FEATURE MATRIX PREPARATION
# ============================================================

def prepare_model_matrix(
    feature_row,
    feature_order,
):
    missing = [
        c for c in feature_order
        if c not in feature_row.columns
    ]

    if missing:
        return None, missing

    X = (
        feature_row[
            feature_order
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    return X, []


# ============================================================
# PREVIOUS DCS STATE
# ============================================================

def find_previous_timestamp(
    dcs,
    selected_ts,
):
    lower = (
        selected_ts
        - pd.Timedelta(
            hours=DYNAMIC_LOOKBACK_HOURS
        )
    )

    candidates = dcs[
        (dcs["TIMESTAMP"] < selected_ts)
        & (dcs["TIMESTAMP"] >= lower)
        & dcs["RT_VALID"]
    ]

    if candidates.empty:
        return None

    return candidates.iloc[-1]["TIMESTAMP"]


# ============================================================
# BUILD CURRENT FEATURE ROW
# ============================================================

def build_prediction_row(
    dcs,
    runs,
    selected_ts,
    ood_stats,
    feature_order,
    base_feature_names,
    engineered_feature_names,
):
    trajectory = calculate_trajectory(
        dcs,
        runs,
        selected_ts,
    )

    if trajectory is None:
        raise ValueError(
            "The selected timestamp does not have a valid "
            "continuous causal RT trajectory. "
            "Select a timestamp inside a continuous RT-valid "
            "production run with enough upstream history."
        )

    current_dict = build_v9_base_features(
        trajectory,
        dcs,
    )

    current = pd.DataFrame(
        [current_dict]
    )

    # The transition scaler must be frozen from development.
    # We therefore reconstruct scaling from the frozen OOD/reference
    # statistics where possible. For exact V9.4 deployment, the
    # reference-statistics file is the authoritative artifact.
    #
    # If the saved transition-feature source contains a richer
    # training-reference table, it should be used instead. The app
    # intentionally refuses to silently invent such statistics.
    #
    # For the normal V9.4 freeze package, the selected engineered
    # feature definitions and reference parameters are included.

    # Load reference parameters for transition thresholds/statistics.
    reference_parameters = {}

    if REFERENCE_PARAM_FILE.exists():
        reference_parameters = load_json(
            REFERENCE_PARAM_FILE
        )

    # If reference_parameters contains explicit transition statistics,
    # use them. Otherwise use the frozen OOD statistics for the same
    # source features where available.
    #
    # To avoid target leakage, no current-row statistics are used.

    # Build an engineering reference from available frozen stats.
    #
    # IMPORTANT:
    # The V9.3 transition features require development-only median/IQR
    # for each source movement feature. The frozen model package should
    # contain these values in v94_reference_parameters.json.
    transition_reference = {}

    if isinstance(
        reference_parameters,
        dict,
    ):
        transition_reference = (
            reference_parameters.get(
                "transition_reference",
                reference_parameters.get(
                    "TRANSITION_REFERENCE",
                    {},
                ),
            )
        )

    if transition_reference:
        eng = build_transition_features_from_frozen_stats(
            current,
            transition_reference,
        )

    else:
        # V9.3's frozen engineered features are based on six
        # END30_MEAN - RTWIN_MEAN state shifts. Their development
        # median/IQR are present in the frozen OOD statistics because
        # the raw V93_STATE_SHIFT_* features are part of the model.
        #
        # Use ONLY those development-frozen statistics. Never estimate
        # them from the current prediction row.
        required_shift_stats = [
            f"V93_STATE_SHIFT_{tag}"
            for tag in TRANSITION_TAGS
        ]

        stats_feature_col = None
        for candidate in [
            "FEATURE",
            "feature",
            "COLUMN",
            "column",
        ]:
            if candidate in ood_stats.columns:
                stats_feature_col = candidate
                break

        if stats_feature_col is None:
            stats_feature_col = ood_stats.columns[0]

        available_stats = set(
            ood_stats[stats_feature_col].astype(str)
        )

        missing_shift_stats = [
            c for c in required_shift_stats
            if c not in available_stats
        ]

        if missing_shift_stats:
            raise ValueError(
                "Frozen V9.3 transition reference statistics are "
                "not available in the V9.4 package. Missing:\n"
                + "\n".join(missing_shift_stats)
            )

        eng = build_transition_features_from_ood_stats(
            current,
            ood_stats,
        )

    feature_row = pd.concat(
        [current, eng],
        axis=1,
    )

    feature_row = feature_row.loc[
        :,
        ~feature_row.columns.duplicated(),
    ]

    # Add a useful deployment timestamp alias if required by schema.
    if (
        "LAB_TIMESTAMP"
        not in feature_row.columns
    ):
        feature_row[
            "LAB_TIMESTAMP"
        ] = selected_ts

    X, missing = prepare_model_matrix(
        feature_row,
        feature_order,
    )

    return (
        X,
        feature_row,
        trajectory,
        missing,
        reference_parameters,
    )


# ============================================================
# FROZEN-STAT TRANSITION ENGINE
# ============================================================

def _get_stat(
    stats,
    feature,
    key_names,
    default=np.nan,
):
    if feature not in stats:
        return default

    value = stats[feature]

    if not isinstance(
        value,
        dict,
    ):
        return default

    for key in key_names:
        if key in value:
            return value[key]

    return default


def _frozen_robust_z(
    value,
    med,
    iqr,
):
    if pd.isna(value):
        return np.nan

    if (
        pd.isna(med)
        or pd.isna(iqr)
    ):
        return np.nan

    iqr = float(iqr)

    if (
        not np.isfinite(iqr)
        or iqr < 1e-12
    ):
        iqr = 1.0

    z = (
        float(value)
        - float(med)
    ) / iqr

    return float(
        np.clip(
            z,
            -8,
            8,
        )
    )


def build_transition_features_from_frozen_stats(
    df,
    stats,
):
    """
    Exact deployment equivalent of V9.3's transition feature logic,
    using frozen development statistics.

    Expected stats structure:
    {
      "RTWIN_IC-17013_DELTA": {
          "MEDIAN": ...,
          "IQR": ...
      },
      ...
    }

    No data from the current prediction row is used to estimate
    these parameters.
    """

    out = pd.DataFrame(
        index=df.index
    )

    z_cols = []

    for tag in TRANSITION_TAGS:

        source_cols = [
            f"RTWIN_{tag}_DELTA",
            f"RTWIN_{tag}_SLOPE",
            f"END30_{tag}_DELTA",
            f"END30_{tag}_SLOPE",
        ]

        for source in source_cols:

            if source not in df.columns:
                continue

            value = pd.to_numeric(
                df[source],
                errors="coerce",
            ).abs().iloc[0]

            name = (
                f"V93_ABS_{source}"
            )

            out.loc[
                df.index[0],
                name,
            ] = value

            med = _get_stat(
                stats,
                source,
                ["ABS_MEDIAN", "MEDIAN"],
            )

            iqr = _get_stat(
                stats,
                source,
                ["ABS_IQR", "IQR"],
            )

            zname = (
                f"V93_ROBUST_MOVE_{source}"
            )

            out.loc[
                df.index[0],
                zname,
            ] = _frozen_robust_z(
                value,
                med,
                iqr,
            )

            z_cols.append(zname)

    # State shifts.
    for tag in TRANSITION_TAGS:

        rt = f"RTWIN_{tag}_MEAN"
        recent = f"END30_{tag}_MEAN"

        if (
            rt not in df.columns
            or recent not in df.columns
        ):
            continue

        value = (
            pd.to_numeric(
                df[recent],
                errors="coerce",
            ).iloc[0]
            -
            pd.to_numeric(
                df[rt],
                errors="coerce",
            ).iloc[0]
        )

        name = (
            f"V93_STATE_SHIFT_{tag}"
        )

        out.loc[
            df.index[0],
            name,
        ] = value

        med = _get_stat(
            stats,
            name,
            ["MEDIAN"],
        )

        iqr = _get_stat(
            stats,
            name,
            ["IQR"],
        )

        zname = (
            f"V93_ROBUST_STATE_SHIFT_{tag}"
        )

        out.loc[
            df.index[0],
            zname,
        ] = _frozen_robust_z(
            value,
            med,
            iqr,
        )

        z_cols.append(zname)

    if z_cols:

        zmat = out[z_cols].abs()

        out[
            "V93_PROCESS_TRANSITION_SCORE"
        ] = zmat.mean(
            axis=1,
            skipna=True,
        )

        out[
            "V93_MAX_TRANSITION_Z"
        ] = zmat.max(
            axis=1,
            skipna=True,
        )

        high_count = 0.0
        valid_count = 0.0

        for c in z_cols:

            value = out.iloc[0][c]

            if pd.notna(value):
                valid_count += 1.0

                if abs(float(value)) >= 1.0:
                    high_count += 1.0

        out[
            "V93_N_HIGH_TRANSITIONS"
        ] = high_count

        out[
            "V93_FRACTION_HIGH_TRANSITIONS"
        ] = (
            high_count / valid_count
            if valid_count > 0
            else np.nan
        )

    # RT anomaly.
    rt_cols = [
        "TOTAL_RT_HR_TRAJ",
        "DRR_RT_HR_TRAJ",
        "PP2_RT_HR_TRAJ",
        "PP1_RT_HR_TRAJ",
        "EST2_RT_HR_TRAJ",
        "EST1_RT_HR_TRAJ",
    ]

    rt_z_cols = []

    for c in rt_cols:

        if c not in df.columns:
            continue

        value = pd.to_numeric(
            df[c],
            errors="coerce",
        ).iloc[0]

        med = _get_stat(
            stats,
            c,
            ["MEDIAN"],
        )

        iqr = _get_stat(
            stats,
            c,
            ["IQR"],
        )

        zname = (
            "V93_RT_ABS_Z_"
            + c.replace(
                "_HR_TRAJ",
                "",
            )
        )

        out.loc[
            df.index[0],
            zname,
        ] = abs(
            _frozen_robust_z(
                value,
                med,
                iqr,
            )
        )

        rt_z_cols.append(zname)

    if rt_z_cols:

        out[
            "V93_RT_ANOMALY_SCORE"
        ] = out[
            rt_z_cols
        ].mean(
            axis=1,
            skipna=True,
        )

        out[
            "V93_MAX_RT_ANOMALY"
        ] = out[
            rt_z_cols
        ].max(
            axis=1,
            skipna=True,
        )

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def build_transition_features_from_ood_stats(
    df,
    ood_stats,
):
    """
    Fallback using the frozen V9.4 development statistics file.

    This is deliberately conservative. If the required frozen
    statistics are not present, the application stops instead of
    fitting statistics from the prediction data.
    """

    if ood_stats is None or ood_stats.empty:
        raise ValueError(
            "Frozen transition reference statistics are unavailable."
        )

    feature_col = None

    for c in [
        "FEATURE",
        "feature",
        "COLUMN",
        "column",
    ]:
        if c in ood_stats.columns:
            feature_col = c
            break

    if feature_col is None:
        feature_col = ood_stats.columns[0]

    stats = {}

    for _, r in ood_stats.iterrows():

        feature = str(
            r[feature_col]
        )

        stats[feature] = {
            "MEDIAN": r.get(
                "MEDIAN",
                r.get(
                    "median",
                    np.nan,
                ),
            ),
            "IQR": r.get(
                "IQR",
                r.get(
                    "iqr",
                    np.nan,
                ),
            ),
            "Q1": r.get(
                "Q1",
                r.get(
                    "q1",
                    np.nan,
                ),
            ),
            "Q3": r.get(
                "Q3",
                r.get(
                    "q3",
                    np.nan,
                ),
            ),
        }

    return build_transition_features_from_frozen_stats(
        df,
        stats,
    )


# ============================================================
# APPLICATION UI
# ============================================================

if "app_reset_id" not in st.session_state:
    st.session_state.app_reset_id = 0


header1, header2 = st.columns([8, 1.6])

with header1:
    st.title("POLY-II IV Prediction")
    st.caption(
        "V9.4 frozen candidate model | "
        "Single timestamp prediction from raw 15-minute DCS"
    )

with header2:
    st.markdown(
        "<div style='height:15px'></div>",
        unsafe_allow_html=True,
    )

    reset_clicked = st.button(
        "↻  New Prediction",
        use_container_width=True,
    )

if reset_clicked:

    current_id = st.session_state.app_reset_id

    for key in list(
        st.session_state.keys()
    ):
        if key != "app_reset_id":
            del st.session_state[key]

    st.session_state.app_reset_id = (
        current_id + 1
    )

    st.cache_data.clear()

    st.rerun()


st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader("0. Frozen V9.4 Model")

try:

    (
        frozen_model,
        feature_order,
        base_feature_names,
        engineered_feature_names,
        reference_parameters,
        ood_stats,
    ) = load_frozen_model(
        str(MODEL_DIR)
    )

    model_hash = sha256_file(
        MODEL_FILE
    )

    st.success(
        "Frozen V9.4 model loaded successfully."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Algorithm",
        "HistGradientBoosting",
    )

    c2.metric(
        "Frozen Features",
        str(len(feature_order)),
    )

    c3.metric(
        "Base Features",
        str(len(base_feature_names)),
    )

    c4.metric(
        "Engineered Features",
        str(len(engineered_feature_names)),
    )

    if MANIFEST_FILE.exists():

        manifest = load_json(
            MANIFEST_FILE
        )

        st.caption(
            f"Model status: "
            f"{manifest.get('status', 'FROZEN CANDIDATE')}"
        )

    with st.expander(
        "Model artifact information"
    ):
        st.write(
            "Model file:",
            str(MODEL_FILE),
        )
        st.code(
            model_hash,
            language="text",
        )

except Exception as exc:

    st.error(
        "Frozen V9.4 model package could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# DCS UPLOAD
# ============================================================

st.subheader(
    "1. Upload DCS Data"
)

st.write(
    "Upload the same raw 15-minute POLY-II DCS Excel "
    "format used by the Colour Prediction application."
)

uploaded_dcs = st.file_uploader(
    "Raw DCS Excel File",
    type=["xlsx", "xls"],
    key=f"dcs_uploader_{st.session_state.app_reset_id}",
)

if uploaded_dcs is None:

    st.info(
        "Please upload the raw DCS Excel file."
    )

    st.caption(
        f"Nova file expected at: {NOVA_FILE}"
    )

    st.stop()


# ============================================================
# LOAD / PREPROCESS DCS
# ============================================================

try:

    dcs = load_raw_dcs(
        uploaded_dcs.getvalue()
    )

    curves = load_nova_curves()

    dcs = add_inventory(
        dcs,
        curves,
    )

    dcs = calculate_rt(
        dcs
    )

    dcs, runs = build_runs(
        dcs
    )

except Exception as exc:

    st.error(
        "DCS preprocessing failed."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================

st.subheader(
    "2. DCS Data Summary"
)

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "DCS Rows",
    f"{len(dcs):,}",
)

s2.metric(
    "DCS Start",
    dcs["TIMESTAMP"].min().strftime(
        "%d-%m-%Y %H:%M"
    ),
)

s3.metric(
    "DCS End",
    dcs["TIMESTAMP"].max().strftime(
        "%d-%m-%Y %H:%M"
    ),
)

s4.metric(
    "Valid RT Rows",
    f"{int(dcs['RT_VALID'].sum()):,}",
)


# ============================================================
# TIMESTAMP SELECTION
# ============================================================

st.subheader(
    "3. Select Prediction Date & Time"
)

valid_timestamps = (
    dcs.loc[
        dcs["RT_VALID"],
        "TIMESTAMP",
    ]
    .dropna()
    .sort_values()
    .tolist()
)

if not valid_timestamps:

    st.error(
        "No RT-valid DCS timestamps are available."
    )

    st.stop()


dates = sorted(
    {
        pd.Timestamp(x).date()
        for x in valid_timestamps
    }
)

date_col, time_col = st.columns(2)

with date_col:

    selected_date = st.selectbox(
        "Prediction Date",
        dates,
        format_func=lambda x: x.strftime(
            "%d-%m-%Y"
        ),
    )

date_timestamps = [
    pd.Timestamp(x)
    for x in valid_timestamps
    if pd.Timestamp(x).date()
    == selected_date
]

time_values = [
    x.time()
    for x in date_timestamps
]

with time_col:

    selected_time = st.selectbox(
        "DCS Time",
        time_values,
        format_func=lambda x: x.strftime(
            "%I:%M:%S %p"
        ),
    )


selected_ts = pd.Timestamp(
    selected_date
).normalize() + pd.Timedelta(
    hours=selected_time.hour,
    minutes=selected_time.minute,
    seconds=selected_time.second,
)


selected_row = (
    dcs[
        dcs["TIMESTAMP"]
        == selected_ts
    ]
)

if selected_row.empty:

    # Defensive nearest exact DCS timestamp match.
    nearest_idx = (
        (
            dcs["TIMESTAMP"]
            - selected_ts
        )
        .abs()
        .idxmin()
    )

    selected_ts = pd.Timestamp(
        dcs.loc[
            nearest_idx,
            "TIMESTAMP",
        ]
    )

    selected_row = dcs.loc[
        [nearest_idx]
    ]


# ============================================================
# PREDICTION
# ============================================================

st.subheader(
    "4. IV Prediction"
)

predict_clicked = st.button(
    "🔬  PREDICT IV",
    type="primary",
    use_container_width=True,
)

if predict_clicked:

    try:

        X, feature_row, trajectory, missing, ref_params = (
            build_prediction_row(
                dcs=dcs,
                runs=runs,
                selected_ts=selected_ts,
                ood_stats=ood_stats,
                feature_order=feature_order,
                base_feature_names=base_feature_names,
                engineered_feature_names=engineered_feature_names,
            )
        )

        # ----------------------------------------------------
        # HARD FEATURE SCHEMA CHECK
        # ----------------------------------------------------

        if missing:

            st.error(
                "Prediction stopped because the frozen V9.4 "
                "feature definition could not be reproduced. "
                "No prediction was generated."
            )

            st.write(
                f"Missing frozen features: {len(missing)}"
            )

            st.code(
                "\n".join(
                    missing
                ),
                language="text",
            )

            st.stop()

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        if X is None:

            st.error(
                "Prediction matrix could not be constructed."
            )

            st.stop()

        # Final exact feature-order assertion.
        if list(X.columns) != list(
            feature_order
        ):

            st.error(
                "Feature order mismatch. "
                "Prediction was refused."
            )

            st.stop()

        # ----------------------------------------------------
        # PREDICT — NO FIT / REFIT
        # ----------------------------------------------------

        prediction = float(
            frozen_model.predict(
                X
            )[0]
        )

        # ----------------------------------------------------
        # PROCESS / RT INFORMATION
        # ----------------------------------------------------

        tpd = pd.to_numeric(
            selected_row.iloc[0]["TPD"],
            errors="coerce",
        )

        total_rt = float(
            trajectory[
                "EST1_RT_HR"
            ]
            + trajectory[
                "EST2_RT_HR"
            ]
            + trajectory[
                "PP1_RT_HR"
            ]
            + trajectory[
                "PP2_RT_HR"
            ]
            + trajectory[
                "DRR_RT_HR"
            ]
        )

        # ----------------------------------------------------
        # OOD STATUS
        # ----------------------------------------------------

        ood_status, max_z, frac_caution = (
            calculate_ood_status(
                X,
                ood_stats,
                feature_order,
            )
        )

        # ----------------------------------------------------
        # TRANSITION STATUS
        # ----------------------------------------------------

        transition_score = np.nan

        if (
            "V93_PROCESS_TRANSITION_SCORE"
            in feature_row.columns
        ):
            transition_score = pd.to_numeric(
                feature_row[
                    "V93_PROCESS_TRANSITION_SCORE"
                ].iloc[0],
                errors="coerce",
            )

        transition_high_threshold = np.nan

        if isinstance(
            ref_params,
            dict,
        ):
            transition_high_threshold = (
                ref_params.get(
                    "transition_high_threshold",
                    ref_params.get(
                        "TRANSITION_HIGH_THRESHOLD",
                        np.nan,
                    ),
                )
            )

        if pd.notna(
            transition_score
        ) and pd.notna(
            transition_high_threshold
        ):

            if (
                float(transition_score)
                >= float(
                    transition_high_threshold
                )
            ):
                transition_status = (
                    "HIGH TRANSITION"
                )
            else:
                transition_status = "STABLE"

        elif pd.notna(
            transition_score
        ):

            transition_status = (
                "CALCULATED"
            )

        else:

            transition_status = (
                "UNAVAILABLE"
            )

        # ----------------------------------------------------
        # INPUT QUALITY
        # ----------------------------------------------------

        numeric_missing = int(
            X.isna().sum().sum()
        )

        # X has NaNs filled before this point.
        input_status = (
            "VALID"
            if numeric_missing == 0
            else "CHECK"
        )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success(
            "IV prediction generated successfully."
        )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:

            st.metric(
                "Predicted IV",
                f"{prediction:.5f}",
            )

        with result_col2:

            st.metric(
                "Acceptance Tolerance",
                "±0.008",
            )

        with result_col3:

            st.metric(
                "TPD",
                (
                    f"{tpd:.2f}"
                    if pd.notna(tpd)
                    else "N/A"
                ),
            )

        # ----------------------------------------------------
        # STATUS CARDS
        # ----------------------------------------------------

        st.markdown(
            "### Prediction Status"
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "RT Status",
            "VALID",
        )

        q2.metric(
            "Input Status",
            input_status,
        )

        q3.metric(
            "Transition",
            transition_status,
        )

        q4.metric(
            "OOD / Support",
            ood_status,
        )

        # ----------------------------------------------------
        # CAUSAL TRAJECTORY
        # ----------------------------------------------------

        st.markdown(
            "### Causal Residence-Time Trajectory"
        )

        rt_table = pd.DataFrame(
            [
                {
                    "Section": "EST-I",
                    "RT (hr)": trajectory[
                        "EST1_RT_HR"
                    ],
                    "Upstream Boundary": trajectory[
                        "EST1_BOUNDARY_TS"
                    ],
                },
                {
                    "Section": "EST-II",
                    "RT (hr)": trajectory[
                        "EST2_RT_HR"
                    ],
                    "Upstream Boundary": trajectory[
                        "EST2_BOUNDARY_TS"
                    ],
                },
                {
                    "Section": "PP-I",
                    "RT (hr)": trajectory[
                        "PP1_RT_HR"
                    ],
                    "Upstream Boundary": trajectory[
                        "PP1_BOUNDARY_TS"
                    ],
                },
                {
                    "Section": "PP-II",
                    "RT (hr)": trajectory[
                        "PP2_RT_HR"
                    ],
                    "Upstream Boundary": trajectory[
                        "PP2_BOUNDARY_TS"
                    ],
                },
                {
                    "Section": "DRR",
                    "RT (hr)": trajectory[
                        "DRR_RT_HR"
                    ],
                    "Upstream Boundary": trajectory[
                        "DRR_BOUNDARY_TS"
                    ],
                },
            ]
        )

        st.dataframe(
            rt_table,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # MODEL DIAGNOSTICS
        # ----------------------------------------------------

        st.markdown(
            "### Model Diagnostics"
        )

        diag = pd.DataFrame(
            [
                {
                    "Item": "Prediction timestamp",
                    "Value": selected_ts.strftime(
                        "%d-%m-%Y %I:%M:%S %p"
                    ),
                },
                {
                    "Item": "Predicted IV",
                    "Value": f"{prediction:.6f}",
                },
                {
                    "Item": "TPD",
                    "Value": (
                        f"{tpd:.3f}"
                        if pd.notna(tpd)
                        else "N/A"
                    ),
                },
                {
                    "Item": "Total RT (hr)",
                    "Value": f"{total_rt:.3f}",
                },
                {
                    "Item": "Transition score",
                    "Value": (
                        f"{float(transition_score):.4f}"
                        if pd.notna(
                            transition_score
                        )
                        else "N/A"
                    ),
                },
                {
                    "Item": "OOD max robust Z",
                    "Value": (
                        f"{max_z:.3f}"
                        if pd.notna(max_z)
                        else "N/A"
                    ),
                },
                {
                    "Item": "OOD caution fraction",
                    "Value": (
                        f"{100 * frac_caution:.2f}%"
                        if pd.notna(frac_caution)
                        else "N/A"
                    ),
                },
                {
                    "Item": "Frozen model features",
                    "Value": str(
                        len(feature_order)
                    ),
                },
                {
                    "Item": "Model refit during prediction",
                    "Value": "NO",
                },
            ]
        )

        st.dataframe(
            diag,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # FEATURE AUDIT
        # ----------------------------------------------------

        with st.expander(
            "Feature reproduction audit"
        ):

            st.write(
                f"Frozen model feature count: "
                f"{len(feature_order)}"
            )

            st.write(
                f"Reproduced feature count: "
                f"{len(feature_row.columns)}"
            )

            st.write(
                f"Missing frozen features: "
                f"{len(missing)}"
            )

            if missing:
                st.code(
                    "\n".join(missing)
                )
            else:
                st.success(
                    "All frozen model features were reproduced."
                )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        export = pd.DataFrame(
            [
                {
                    "TIMESTAMP": selected_ts,
                    "PRED_IV": prediction,
                    "TPD": tpd,
                    "TOTAL_RT_HR": total_rt,
                    "RT_STATUS": "VALID",
                    "TRANSITION_STATUS": transition_status,
                    "TRANSITION_SCORE": transition_score,
                    "OOD_STATUS": ood_status,
                    "OOD_MAX_ROBUST_Z": max_z,
                    "OOD_CAUTION_FRACTION": frac_caution,
                    "MODEL": "V9.4_FROZEN_V9.3_TRANSITION_AWARE",
                    "IV_ACCEPTANCE_THRESHOLD": IV_ACCEPTANCE_THRESHOLD,
                    "FEATURE_COUNT": len(feature_order),
                }
            ]
        )

        st.download_button(
            "Download IV Prediction CSV",
            export.to_csv(
                index=False
            ),
            file_name=(
                "POLY2_IV_prediction_"
                f"{selected_ts.strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
        )

    except Exception as exc:

        st.error(
            "IV prediction failed."
        )

        st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "POLY-II IV Prediction | V9.4 frozen candidate | "
    "Raw 15-minute DCS | Causal residence-time inference | "
    "No previous laboratory IV used | No model refit at deployment"
)
