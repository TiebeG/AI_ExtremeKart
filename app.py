import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "laptimes_v5.csv"   # same CSV you used for training
MODELS_DIR = Path("models")


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_data
def load_lap_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load flat lap data from CSV."""
    df = pd.read_csv(path, sep=";")

    # Normalize dtypes
    df["Driver"] = df["Driver"].astype(str)
    df["Session"] = df["Session"].astype(str)
    df["Kart"] = df["Kart"].astype(str)

    # Build DateTime column for hour/weekday features
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["SessionStart"],
        dayfirst=True,
        errors="coerce",
    )

    return df


@st.cache_resource
def load_model(track_id: int):
    """Load a previously trained model for a given track."""
    model_path = MODELS_DIR / f"model_track_{track_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found for track {track_id} at {model_path}. "
            f"Run train.py first for this track."
        )
    return joblib.load(model_path)


# ============================================================
# FEATURE BUILDING FOR PREDICTION
# ============================================================

def build_feature_row(
    df_track: pd.DataFrame,
    driver: str,
    kart: str,
    session: str | None,
) -> pd.DataFrame:
    """
    Build a single feature row for prediction, filling hour/weekday
    from typical values for the given driver (and optionally session).
    """
    driver = str(driver)
    kart = str(kart)
    if session is not None:
        session = str(session)

    # Filter to driver (and session if provided) to estimate typical time-of-day
    df_sel = df_track[df_track["Driver"] == driver].copy()
    if session is not None:
        df_sel = df_sel[df_sel["Session"].astype(str) == session]

    if df_sel.empty:
        # Fallback to all laps from this track
        df_sel = df_track

    dt = pd.to_datetime(
        df_sel["Date"] + " " + df_sel["SessionStart"],
        dayfirst=True,
        errors="coerce",
    ).dropna()

    if dt.empty:
        hour = 20.0
        weekday = 3.0
    else:
        hour = float(dt.dt.hour.median())
        weekday = float(dt.dt.weekday.median())

    # If session still None, use most common session for this track
    if session is None:
        session = str(df_track["Session"].mode()[0])

    row = pd.DataFrame(
        [{
            "Driver": driver,
            "Kart": kart,
            "Session": session,
            "hour": hour,
            "weekday": weekday,
        }]
    )
    return row


# ============================================================
# PREDICTION FUNCTIONS (3 USE CASES)
# ============================================================

def predict_per_kart(
    model,
    df_track: pd.DataFrame,
    driver: str,
    session: str | None = None,
) -> pd.DataFrame:
    """
    Function 1:
    Given a driver (and optionally session), predict best lap for all karts.
    Returns a DataFrame with Kart and predicted best lap.
    """
    if driver not in set(df_track["Driver"].unique()):
        raise ValueError(f"Driver '{driver}' not found in data for this track.")

    results = []
    for kart in sorted(df_track["Kart"].unique(), key=str):
        X_pred = build_feature_row(df_track, driver=driver, kart=kart, session=session)
        pred = float(model.predict(X_pred)[0])
        results.append({"Kart": kart, "Predicted avg laps (s)": pred})

    return pd.DataFrame(results).sort_values("Predicted avg laps (s)")


def predict_driver_kart(
    model,
    df_track: pd.DataFrame,
    driver: str,
    kart: str,
    session: str | None = None,
) -> float:
    """
    Function 2:
    Given driver + kart, predict best lap (optionally using a session hint).
    If session is None, a typical session is used via build_feature_row.
    """
    if driver not in set(df_track["Driver"].unique()):
        raise ValueError(f"Driver '{driver}' not found in data for this track.")
    if kart not in set(df_track["Kart"].unique()):
        raise ValueError(f"Kart '{kart}' not found in data for this track.")

    X_pred = build_feature_row(df_track, driver=driver, kart=kart, session=session)
    pred = float(model.predict(X_pred)[0])
    return pred


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(page_title="Extreme Kart AI Predictor", page_icon="🏁", layout="centered")
    st.title("🏁 Extreme Kart Lap Time Predictor")
    st.markdown(
        """
        This app uses historical indoor karting data from **Extreme Kart** to predict **best lap times**.

        You can:
        1. Select a **driver** → see predicted best lap per **kart**.  
        2. Select **driver + kart** → predicted best lap.  
        3. Select **driver + kart + session** → more specific prediction.
        """
    )

    # Load data
    try:
        df_all = load_lap_data()
    except Exception as e:
        st.error(f"Error loading data from `{DATA_PATH}`:\n{e}")
        st.stop()

    if df_all.empty:
        st.error("No data loaded from CSV. Check your `laptimes_v5.csv` file.")
        st.stop()

    track_ids = sorted(df_all["Track"].unique())
    if not track_ids:
        st.error("No tracks found in the data.")
        st.stop()

    # --- Select track (model per track) ---
    track_id = st.selectbox(
        "Track",
        options=track_ids,
        format_func=lambda x: f"Track {x}",
    )

    # === NEW: Show track image below the selector ===
    # Assumes files like: track_0.png, track_1.png, ...
    # If you put them in a folder (e.g. "images"), change to Path("images") / f"track_{track_id}.png"
    image_path = Path("images") / f"track_{track_id}.jpg"

    if image_path.exists():
        st.image(
            str(image_path),
            caption=f"Track {track_id}",
            use_container_width=False, width=300
        )
    else:   
        st.info(f"Track image not found for track {track_id} (expected `{image_path.name}`).")

    df_track = df_all[df_all["Track"] == track_id].copy()
    if df_track.empty:
        st.error(f"No data for track {track_id}.")
        st.stop()

    # Load model for this track
    try:
        model = load_model(track_id)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run your training script (train.py) for this track first.")
        st.stop()

    # --- Select driver / kart / session ---
    st.subheader("Input")

    known_drivers = sorted(df_track["Driver"].unique())
    known_karts = sorted(df_track["Kart"].unique(), key=str)
    known_sessions = sorted(df_track["Session"].unique(), key=str)

    if not known_drivers:
        st.error("No drivers found for this track.")
        st.stop()

    driver = st.selectbox("Driver", options=known_drivers)

    kart_option = st.selectbox(
        "Kart (optional)",
        options=["(Any)"] + known_karts,
    )
    kart_selected = None if kart_option == "(Any)" else kart_option

    session_option = st.selectbox(
        "Session (optional)",
        options=["(Any)"] + known_sessions,
    )
    session_selected = None if session_option == "(Any)" else session_option

    st.markdown("---")

    # --- Logic for the three use cases ---

    # CASE 1: only driver (kart = Any) -> table for all karts
    if kart_selected is None:
        st.subheader("Prediction per kart")

        if st.button("Predict for all karts"):
            try:
                df_pred = predict_per_kart(
                    model=model,
                    df_track=df_track,
                    driver=driver,
                    session=session_selected,
                )
                df_pred["Predicted avg laps (s)"] = df_pred["Predicted avg laps (s)"].round(3)
                st.dataframe(df_pred, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # CASE 2 & 3: driver + kart (with or without session)
    else:
        if session_selected is None:
            st.subheader("Prediction for driver + kart")
        else:
            st.subheader("Prediction for driver + kart + session")

        if st.button("Predict"):
            try:
                pred = predict_driver_kart(
                    model=model,
                    df_track=df_track,
                    driver=driver,
                    kart=kart_selected,
                    session=session_selected,
                )
                st.metric(
                    label="Predicted avg laps (s)",
                    value=f"{pred:.3f}",
                )

                st.caption(
                    "Note: This is a statistical expectation based on historical data, "
                    "not a guarantee. Fuel, track evolution, and your form still matter 😉"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
