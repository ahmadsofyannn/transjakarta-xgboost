import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Dashboard Transjakarta — Kepadatan & Insight",
    layout="wide",
)

DAYS_ID = [
    "Senin",
    "Selasa",
    "Rabu",
    "Kamis",
    "Jumat",
    "Sabtu",
    "Minggu",
]

SAMPLE_STOPS = [
    "Pal Putih",
    "Tegalan",
    "Gg. Kunir II",
    "Simpang Kunir Kemukus",
    "Penjaringan",
    "BKN",
    "BNN LRT",
    "Cibubur Junction",
    "Duren Tiga",
    "Penggilingan",
]


@st.cache_data
def load_summary_data(path: str = "data/density_summary.csv") -> pd.DataFrame:
  """Load data agregat untuk grafik & konteks historis; fallback dummy jika belum ada."""
  try:
    df = pd.read_csv(path)
    return df
  except Exception:
    rng = np.random.default_rng(42)
    rows = []
    for stop in SAMPLE_STOPS:
      for dow in range(7):
        is_wknd = dow >= 5
        for hour in range(5, 22):
          rush = (
              1.8
              if (not is_wknd and hour in [7, 8, 17, 18])
              else (0.7 if is_wknd else 1.0)
          )
          count = int(max(1, rng.poisson(lam=18 * rush)))
          rows.append({
              "tapInStopsName": stop,
              "tapInDayOfWeek": dow,
              "tapInHour": hour,
              "passenger_count": count,
          })
    return pd.DataFrame(rows)


@st.cache_resource
def load_density_model(path: str = "density_model.pkl"):
  """Load model_density.pkl dengan aman; return None jika belum tersedia."""
  try:
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
      return artifact["model"], artifact.get("features")
    return artifact, None
  except Exception:
    return None, None


def build_model_input(
    stop_name: str, day_idx: int, hour: int, features: list | None = None
) -> pd.DataFrame:
  """Susun DataFrame input sesuai kebutuhan model_density.pkl (mudah diubah nanti)."""
  row = {
      "tapInStopsName": stop_name,
      "tapInDayOfWeek": day_idx,
      "tapInHour": hour,
  }
  df_in = pd.DataFrame([row])
  if features:
    for col in features:
      if col not in df_in.columns:
        df_in[col] = 0
    df_in = df_in[features]
  return df_in


def predict_passenger_count(
    model, features, stop_name: str, day_idx: int, hour: int
) -> tuple[float, str]:
  """Jalankan prediksi; jika model belum ada atau error kolom, pakai estimasi fallback."""
  if model is not None:
    try:
      X_input = build_model_input(stop_name, day_idx, hour, features)
      pred = float(model.predict(X_input)[0])
      return max(0.0, pred), "model"
    except Exception as e:
      st.caption(
          f"Catatan: memakai estimasi sementara karena input model belum"
          f" disesuaikan ({e})."
      )

  # Fallback sementara sebelum model_density.pkl terhubung penuh
  is_weekend = day_idx >= 5
  rush_boost = (
      1.75
      if (not is_weekend and hour in [7, 8, 17, 18])
      else (0.75 if is_weekend else 1.0)
  )
  base = 22.0 * rush_boost
  return base, "dummy"


# --- Sidebar Navigasi ---
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Ikhtisar Data & EDA",
        "Business Insights",
        "Prediksi Kepadatan Halte",
    ],
)

df_summary = load_summary_data()
model, model_features = load_density_model()

# --- Halaman 1: Ikhtisar Data & EDA ---
if menu == "Ikhtisar Data & EDA":
  st.title("Ikhtisar Data & EDA Transjakarta")
  st.caption(
      "Kerangka awal dashboard — hubungkan metrik & grafik ke data final"
      " agregasi."
  )

  c1, c2, c3 = st.columns(3)
  c1.metric("Total Baris Agregat", f"{len(df_summary):,}")
  c2.metric("Jumlah Halte", f"{df_summary['tapInStopsName'].nunique():,}")
  c3.metric(
      "Total Penumpang (Agregat)", f"{df_summary['passenger_count'].sum():,}"
  )

  hourly = df_summary.groupby("tapInHour", as_index=False)[
      "passenger_count"
  ].sum()
  fig_hr = px.line(
      hourly,
      x="tapInHour",
      y="passenger_count",
      markers=True,
      title="Tren Penumpang Tap-In Berdasarkan Jam",
      labels={"tapInHour": "Jam", "passenger_count": "Jumlah Penumpang"},
  )
  st.plotly_chart(fig_hr, use_container_width=True)

  top_stops = (
      df_summary.groupby("tapInStopsName", as_index=False)["passenger_count"]
      .sum()
      .sort_values("passenger_count", ascending=False)
      .head(10)
  )
  fig_top = px.bar(
      top_stops,
      x="passenger_count",
      y="tapInStopsName",
      orientation="h",
      title="Top 10 Halte Tap-In Tersibuk",
      labels={
          "passenger_count": "Jumlah Penumpang",
          "tapInStopsName": "Halte",
      },
  )
  fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
  st.plotly_chart(fig_top, use_container_width=True)

# --- Halaman 2: Business Insights ---
elif menu == "Business Insights":
  st.title("Business Insights")
  tab1, tab2 = st.tabs(
      ["Hari Kerja vs Akhir Pekan", "Pola Jam per Hari (Heatmap)"]
  )

  with tab1:
    tmp = df_summary.copy()
    tmp["tipe_hari"] = np.where(
        tmp["tapInDayOfWeek"] >= 5, "Akhir Pekan", "Hari Kerja"
    )
    wk_hr = tmp.groupby(["tapInHour", "tipe_hari"], as_index=False)[
        "passenger_count"
    ].sum()
    fig_wk = px.line(
        wk_hr,
        x="tapInHour",
        y="passenger_count",
        color="tipe_hari",
        markers=True,
        title="Perbandingan Penumpang per Jam: Hari Kerja vs Akhir Pekan",
        labels={
            "tapInHour": "Jam",
            "passenger_count": "Jumlah Penumpang",
            "tipe_hari": "Tipe Hari",
        },
    )
    st.plotly_chart(fig_wk, use_container_width=True)

  with tab2:
    pivot_df = (
        df_summary.pivot_table(
            index="tapInDayOfWeek",
            columns="tapInHour",
            values="passenger_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=range(7))
        .rename(index=lambda i: DAYS_ID[i])
    )
    fig_hm = px.imshow(
        pivot_df,
        aspect="auto",
        title="Heatmap Kepadatan Penumpang (Hari vs Jam)",
        labels={
            "x": "Jam",
            "y": "Hari",
            "color": "Jumlah Penumpang",
        },
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# --- Halaman 3: Prediksi Kepadatan Halte ---
elif menu == "Prediksi Kepadatan Halte":
  st.title("Prediksi Kepadatan Halte")

  stops_list = sorted(df_summary["tapInStopsName"].unique().tolist())
  col_a, col_b, col_c = st.columns(3)
  with col_a:
    selected_stop = st.selectbox("Pilih Halte", stops_list)
  with col_b:
    selected_day_name = st.selectbox("Pilih Hari", DAYS_ID)
    selected_day_idx = DAYS_ID.index(selected_day_name)
  with col_c:
    selected_hour = st.slider("Pilih Jam", min_value=0, max_value=23, value=8)

  if st.button("Prediksi Kepadatan", type="primary"):
    est_count, mode = predict_passenger_count(
        model, model_features, selected_stop, selected_day_idx, selected_hour
    )
    label_sumber = (
        "Model `density_model.pkl`"
        if mode == "model"
        else ""
    )
    st.info(
        f"Diperkirakan ada **{est_count:.0f}** penumpang Tap-In di Halte"
        f" **{selected_stop}** pada hari **{selected_day_name}** jam"
        f" **{selected_hour}:00**"
    )

  st.subheader("Konteks Historis")
  hist = df_summary[
      (df_summary["tapInStopsName"] == selected_stop)
      & (df_summary["tapInDayOfWeek"] == selected_day_idx)
  ].sort_values("tapInHour")

  if hist.empty:
    st.warning(
        "Data historis untuk kombinasi halte dan hari ini belum mencukupi."
    )
  else:
    fig_hist = px.bar(
        hist,
        x="tapInHour",
        y="passenger_count",
        title=(
            f"Kepadatan Historis {selected_stop} — Hari {selected_day_name}"
        ),
        labels={"tapInHour": "Jam", "passenger_count": "Jumlah Penumpang"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)