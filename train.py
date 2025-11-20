import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ============================================================
# 1. CONFIG
# ============================================================

DATA_PATH = "../laptimes_v5.csv"   # adjust if needed
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

# Just used in the example block at the bottom
driver = "TOEP"
track = 0


# ============================================================
# 2. CLEANING HELPERS
# ============================================================

def parse_time_value(x):
    """
    Parse a lap time value into float seconds.
    Supports formats like:
    - 34.123
    - "34.123"
    - "1:00.339" (mm:ss.xxx)
    Returns None if it cannot be parsed.
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Handle "M:SS.xxx" or "MM:SS.xxx"
    if ":" in s:
        try:
            mins, secs = s.split(":")
            return float(mins) * 60.0 + float(secs)
        except Exception:
            return None

    # Normal float-like
    try:
        return float(s)
    except Exception:
        return None


def clean_lap_times(df: pd.DataFrame,
                    time_column: str = "LapTime",
                    min_valid: float = 20.0,
                    max_valid: float = 36.999) -> pd.DataFrame:
    """
    Clean lap times:
    - Convert LapTime to float seconds.
    - Drop rows with invalid LapTime.
    - Drop rows outside [min_valid, max_valid].
    """
    df = df.copy()

    # Parse to seconds
    df[time_column] = df[time_column].apply(parse_time_value)

    # Drop invalid / unparsable
    before = len(df)
    df = df.dropna(subset=[time_column])
    after_parse = len(df)

    # Filter by range
    df = df[
        (df[time_column] >= min_valid) &
        (df[time_column] <= max_valid)
    ]
    after_range = len(df)

    print(
        f"[Cleaning] {before} rows -> {after_parse} after parsing -> "
        f"{after_range} after filtering [{min_valid}, {max_valid}]"
    )

    return df


# ============================================================
# 3. LOAD RAW LAP DATA + CLEANING
# ============================================================

def load_lap_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load flat laptime data from CSV and clean LapTime outliers."""
    df = pd.read_csv(path, sep=";")

    # Ensure expected columns exist
    expected_cols = {
        "Driver", "Track", "Session", "SessionStart",
        "Kart", "Lap", "LapTime", "Date"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Normalize dtypes for categorical columns
    df["Driver"] = df["Driver"].astype(str)
    df["Session"] = df["Session"].astype(str)
    df["Kart"] = df["Kart"].astype(str)

    # Clean lap times (parse + drop outliers)
    df = clean_lap_times(df, time_column="LapTime",
                         min_valid=20.0, max_valid=35.999)

    # Build DateTime column (after cleaning so length matches)
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["SessionStart"],
        dayfirst=True,
        errors="coerce"
    )

    return df


# ============================================================
# 4. BUILD SUMMARY DATA PER DRIVER–KART–SESSION
# ============================================================

def build_summary(df_track: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise laps per (Driver, Kart, Session) for a single track.
    Returns a DataFrame with targets and features like hour/weekday.
    """
    summary = (
        df_track
        .groupby(["Driver", "Kart", "Session"], as_index=False)
        .agg(
            avg_lap_time=("LapTime", "mean"),
            best_lap_time=("LapTime", "min"),
            std_lap_time=("LapTime", "std"),
            laps=("Lap", "max"),
            session_start=("SessionStart", "first"),
            date=("Date", "first"),
            datetime=("DateTime", "first"),
        )
    )

    # Time-based features
    summary["hour"] = summary["datetime"].dt.hour
    summary["weekday"] = summary["datetime"].dt.weekday

    # Ensure categorical as string
    summary["Driver"] = summary["Driver"].astype(str)
    summary["Kart"] = summary["Kart"].astype(str)
    summary["Session"] = summary["Session"].astype(str)

    return summary


# ============================================================
# 5. TRAIN MODEL FOR A SINGLE TRACK
# ============================================================

def train_model_for_track(df: pd.DataFrame, track_id: int) -> None:
    """
    Train a RandomForestRegressor for a single track and save it to disk.
    Target: best_lap_time (can be changed to avg_lap_time if desired).
    """
    df_track = df[df["Track"] == track_id].copy()
    if df_track.empty:
        print(f"[Track {track_id}] No data, skipping.")
        return

    summary = build_summary(df_track)

    cat_features = ["Driver", "Kart", "Session"]
    num_features = ["hour", "weekday"]

    X = summary[cat_features + num_features].copy()
    y = summary["best_lap_time"]  # or "avg_lap_time"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    model_path = MODELS_DIR / f"model_track_{track_id}.joblib"
    joblib.dump(pipe, model_path)

    print(f"[Track {track_id}] Trained. MAE = {mae:.3f} s. Saved to {model_path}")


# ============================================================
# 6. LOAD MODEL AND BUILD FEATURE ROWS FOR PREDICTION
# ============================================================

def load_model(track_id: int):
    """Load a previously trained model for a given track."""
    model_path = MODELS_DIR / f"model_track_{track_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for track {track_id} at {model_path}")
    return joblib.load(model_path)


def build_feature_row(df_track: pd.DataFrame,
                      driver: str,
                      kart: str | None = None,
                      session: str | None = None) -> pd.DataFrame:
    """
    Build a single-feature row for prediction, filling hour/weekday from
    typical values for the given driver (and optionally session).
    """
    driver = str(driver)
    if kart is not None:
        kart = str(kart)
    if session is not None:
        session = str(session)

    # Filter to driver (and session if provided) to estimate typical time-of-day
    df_sel = df_track[df_track["Driver"] == driver]
    if session is not None:
        df_sel = df_sel[df_sel["Session"].astype(str) == session]

    if df_sel.empty:
        df_sel = df_track

    dt = pd.to_datetime(
        df_sel["Date"] + " " + df_sel["SessionStart"],
        dayfirst=True,
        errors="coerce"
    ).dropna()

    if dt.empty:
        hour = 20.0
        weekday = 3.0
    else:
        hour = float(dt.dt.hour.median())
        weekday = float(dt.dt.weekday.median())

    # Fill missing kart/session with most common
    if kart is None:
        kart = str(df_track["Kart"].mode()[0])
    if session is None:
        session = str(df_track["Session"].mode()[0])

    row = pd.DataFrame([{
        "Driver": driver,
        "Kart": kart,
        "Session": session,
        "hour": hour,
        "weekday": weekday,
    }])
    return row


# ============================================================
# 7. PREDICTION FUNCTIONS (YOUR 3 USE CASES)
# ============================================================

def predict_per_kart(model,
                     df_track: pd.DataFrame,
                     driver: str) -> pd.DataFrame:
    """
    Function 1:
    Given a driver, predict best lap for all karts on that track.
    Returns a DataFrame with Kart and predicted best lap.
    """
    results = []
    for kart in sorted(df_track["Kart"].unique(), key=str):
        X_pred = build_feature_row(df_track, driver=driver, kart=kart, session=None)
        pred = float(model.predict(X_pred)[0])
        results.append({"Kart": kart, "pred_best_lap": pred})
    return pd.DataFrame(results).sort_values("pred_best_lap")


def predict_driver_kart(model,
                        df_track: pd.DataFrame,
                        driver: str,
                        kart: str) -> float:
    """
    Function 2:
    Given driver + kart, predict best lap (session not specified).
    """
    X_pred = build_feature_row(df_track, driver=driver, kart=kart, session=None)
    pred = float(model.predict(X_pred)[0])
    return pred


def predict_driver_kart_session(model,
                                df_track: pd.DataFrame,
                                driver: str,
                                kart: str,
                                session: str) -> float:
    """
    Function 3:
    Given driver + kart + session, predict best lap.
    """
    X_pred = build_feature_row(df_track, driver=driver, kart=kart, session=session)
    pred = float(model.predict(X_pred)[0])
    return pred


# ============================================================
# 8. MAIN: TRAIN MODELS FOR ALL TRACKS (RUN ONCE)
# ============================================================

if __name__ == "__main__":
    df_all = load_lap_data()

    # Train a model for each track present in the data
    track_ids = sorted(df_all["Track"].unique())
    for track_id in track_ids:
        train_model_for_track(df_all, track_id)

    # Only run examples if we have at least one track
    if track_ids:
        model_example = load_model(track)
        df_track_example = df_all[df_all["Track"] == track].copy()

        # Example: function 1 – predictions for all karts for a driver
        driver = df_track_example["Driver"].iloc[0]
        print(
            f"\nPredicted best lap per kart for driver '{driver}' "
            f"on track {track}:"
        )
        print(predict_per_kart(model_example, df_track_example, driver))

        # Example: function 2
        kart = df_track_example["Kart"].iloc[0]
        pred2 = predict_driver_kart(
            model_example, df_track_example, driver, kart
        )
        print(
            f"\nPredicted best lap for driver '{driver}' in kart "
            f"{kart}: {pred2:.3f} s"
        )

        # Example: function 3
        session = df_track_example["Session"].iloc[0]
        pred3 = predict_driver_kart_session(
            model_example,
            df_track_example,
            driver,
            kart,
            session,
        )
        print(
            f"\nPredicted best lap for driver '{driver}' in kart "
            f"{kart} session {session}: {pred3:.3f} s"
        )
