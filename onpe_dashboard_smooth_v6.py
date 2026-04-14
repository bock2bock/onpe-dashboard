from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


# =====================================
# CONFIGURACIÓN
# =====================================

CSV_FILE = "onpe_tracking_v2.csv"
ACTAS_CSV_FILE = "onpe_actas_tracking.csv"

REFRESH_SECONDS = 60
PAGE_REFRESH_MS = REFRESH_SECONDS * 1000


# =====================================
# CONFIG STREAMLIT
# =====================================

st.set_page_config(
    page_title="CONTEO EN VIVO",
    layout="wide"
)


# =====================================
# ESTILOS
# =====================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .sub-title {
        font-size: 1.05rem;
        opacity: 0.82;
        margin-bottom: 1.5rem;
        max-width: 1100px;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
    }

    .soft-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .mini-note {
        font-size: 0.92rem;
        opacity: 0.78;
    }

    .badge-line {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.88rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================
# UTILIDADES
# =====================================

def safe_int(value):
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def fmt_int(value):
    return f"{safe_int(value):,}"


# =====================================
# CARGA DE DATOS
# =====================================

@st.cache_data(ttl=5)
def load_data():
    if not Path(CSV_FILE).exists():
        return None

    df = pd.read_csv(CSV_FILE)

    required_cols = [
        "timestamp",
        "porky_votes",
        "nieto_votes",
        "sanchez_votes",
        "diff_porky_nieto",
        "leader",
    ]

    for col in required_cols:
        if col not in df.columns:
            return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["porky_votes"] = pd.to_numeric(df["porky_votes"], errors="coerce")
    df["nieto_votes"] = pd.to_numeric(df["nieto_votes"], errors="coerce")
    df["sanchez_votes"] = pd.to_numeric(df["sanchez_votes"], errors="coerce")
    df["diff_porky_nieto"] = pd.to_numeric(df["diff_porky_nieto"], errors="coerce")

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        return None

    df["delta_porky"] = df["porky_votes"].diff().fillna(0)
    df["delta_nieto"] = df["nieto_votes"].diff().fillna(0)
    df["delta_sanchez"] = df["sanchez_votes"].diff().fillna(0)

    df["delta_minutes"] = df["timestamp"].diff().dt.total_seconds().div(60)
    df["delta_minutes"] = df["delta_minutes"].fillna(0)

    df["porky_per_min"] = df.apply(
        lambda row: row["delta_porky"] / row["delta_minutes"] if row["delta_minutes"] and row["delta_minutes"] > 0 else 0,
        axis=1
    )
    df["nieto_per_min"] = df.apply(
        lambda row: row["delta_nieto"] / row["delta_minutes"] if row["delta_minutes"] and row["delta_minutes"] > 0 else 0,
        axis=1
    )
    df["sanchez_per_min"] = df.apply(
        lambda row: row["delta_sanchez"] / row["delta_minutes"] if row["delta_minutes"] and row["delta_minutes"] > 0 else 0,
        axis=1
    )

    df["porky_per_min_ma3"] = df["porky_per_min"].rolling(3, min_periods=1).mean()
    df["nieto_per_min_ma3"] = df["nieto_per_min"].rolling(3, min_periods=1).mean()
    df["sanchez_per_min_ma3"] = df["sanchez_per_min"].rolling(3, min_periods=1).mean()

    return df


@st.cache_data(ttl=5)
def load_actas_data():
    if not Path(ACTAS_CSV_FILE).exists():
        return None

    df = pd.read_csv(ACTAS_CSV_FILE)

    if df.empty:
        return None

    required_cols = ["timestamp", "actas_pct"]
    for col in required_cols:
        if col not in df.columns:
            return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["actas_pct"] = pd.to_numeric(df["actas_pct"], errors="coerce")

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        return None

    return df


# =====================================
# CÁLCULOS
# =====================================

def build_snapshot(df):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    return {"prev": prev, "last": last}


def build_insights(df):
    if len(df) < 2:
        return ["Todavía no hay suficientes puntos reales para leer la tendencia."]

    prev = df.iloc[-2]
    last = df.iloc[-1]

    delta_p = last["delta_porky"]
    delta_n = last["delta_nieto"]
    delta_s = last["delta_sanchez"]

    gains = {
        "🐷 Porky": delta_p,
        "🌞 Nieto": delta_n,
        "🤠 Sánchez": delta_s,
    }

    winner = max(gains, key=gains.get)

    insights = []
    insights.append(
        "En el último tramo real confirmado, el mayor crecimiento fue para {} (+{:,.0f} votos).".format(
            winner, gains[winner]
        )
    )

    gap_prev = prev["porky_votes"] - prev["nieto_votes"]
    gap_last = last["porky_votes"] - last["nieto_votes"]

    if gap_last > gap_prev:
        insights.append("La brecha entre 🐷 Porky y 🌞 Nieto se amplió en el último tramo real.")
    elif gap_last < gap_prev:
        insights.append("🌞 Nieto recortó distancia frente a 🐷 Porky en el último tramo real.")
    else:
        insights.append("La brecha entre 🐷 Porky y 🌞 Nieto se mantuvo estable en el último tramo real.")

    insights.append(
        "Los contadores animados reproducen el último tramo real para mantener continuidad visual. No representan una proyección del próximo corte."
    )

    return insights


# =====================================
# GRÁFICOS
# =====================================

def plot_curvas_votos(df):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.plot(df["timestamp"], df["porky_votes"], linewidth=3, label="🐷 Porky")
    ax.plot(df["timestamp"], df["nieto_votes"], linewidth=3, label="🌞 Nieto")
    ax.plot(df["timestamp"], df["sanchez_votes"], linewidth=3, label="🤠 Sánchez")

    ax.scatter(df["timestamp"].iloc[-1], df["porky_votes"].iloc[-1], s=60)
    ax.scatter(df["timestamp"].iloc[-1], df["nieto_votes"].iloc[-1], s=60)
    ax.scatter(df["timestamp"].iloc[-1], df["sanchez_votes"].iloc[-1], s=60)

    ax.annotate(
        f"Porky {safe_int(df['porky_votes'].iloc[-1]):,}",
        (df["timestamp"].iloc[-1], df["porky_votes"].iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )
    ax.annotate(
        f"Nieto {safe_int(df['nieto_votes'].iloc[-1]):,}",
        (df["timestamp"].iloc[-1], df["nieto_votes"].iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )
    ax.annotate(
        f"Sanchez {safe_int(df['sanchez_votes'].iloc[-1]):,}",
        (df["timestamp"].iloc[-1], df["sanchez_votes"].iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )

    ax.set_title("Votos confirmados", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Votos")
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_brecha(df):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.plot(df["timestamp"], df["diff_porky_nieto"], linewidth=3, label="📏 Brecha Porky-Nieto")
    ax.scatter(df["timestamp"].iloc[-1], df["diff_porky_nieto"].iloc[-1], s=60)

    ax.annotate(
        f"{safe_int(df['diff_porky_nieto'].iloc[-1]):,}",
        (df["timestamp"].iloc[-1], df["diff_porky_nieto"].iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )

    ax.set_title("Brecha entre Porky y Nieto", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Diferencia de votos")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =====================================
# TARJETAS ANIMADAS
# =====================================

def animated_candidate_card(nombre, emoji, prev_value, last_value, delta_value, per_min, duration_ms, max_value, color):
    progress_start = 0 if max_value <= 0 else prev_value / max_value
    progress_end = 0 if max_value <= 0 else last_value / max_value

    payload = {
        "nombre": nombre,
        "emoji": emoji,
        "prev_value": safe_int(prev_value),
        "last_value": safe_int(last_value),
        "delta_value": safe_int(delta_value),
        "per_min": round(float(per_min), 1),
        "duration_ms": int(duration_ms),
        "progress_start": float(progress_start),
        "progress_end": float(progress_end),
        "color": color,
    }

    html = f"""
    <html>
    <head>
      <style>
        body {{
          margin: 0;
          background: transparent;
          color: white;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .card {{
          padding: 20px;
          border-radius: 22px;
          background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
          border: 1px solid rgba(255,255,255,0.08);
          min-height: 390px;
          box-sizing: border-box;
          box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .title {{
          font-size: 2rem;
          font-weight: 800;
          margin-bottom: 12px;
        }}
        .label {{
          font-size: 0.95rem;
          opacity: 0.8;
          margin-top: 10px;
          margin-bottom: 6px;
        }}
        .official {{
          font-size: 2.2rem;
          font-weight: 900;
          margin-bottom: 12px;
        }}
        .animated {{
          font-size: 2rem;
          font-weight: 700;
          margin-bottom: 14px;
        }}
        .bar-wrap {{
          width: 100%;
          height: 18px;
          background: rgba(255,255,255,0.08);
          border-radius: 999px;
          overflow: hidden;
          margin-top: 8px;
          margin-bottom: 18px;
        }}
        .bar {{
          height: 100%;
          width: 0%;
          background: {payload["color"]};
          border-radius: 999px;
          transition: width 0.08s linear;
        }}
        .metrics {{
          display: flex;
          gap: 14px;
          margin-top: 10px;
        }}
        .metric-box {{
          flex: 1;
          padding: 12px 12px;
          background: rgba(255,255,255,0.04);
          border-radius: 14px;
          min-height: 82px;
          box-sizing: border-box;
        }}
        .metric-title {{
          font-size: 0.82rem;
          opacity: 0.78;
          line-height: 1.2;
        }}
        .metric-value {{
          font-size: 1.15rem;
          font-weight: 700;
          margin-top: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="title">{payload["emoji"]} {payload["nombre"]}</div>

        <div class="label">Votos confirmados</div>
        <div class="official">{payload["last_value"]:,}</div>

        <div class="label">Contador en movimiento</div>
        <div id="animatedValue" class="animated">{payload["prev_value"]:,}</div>

        <div class="label">Avance del último tramo real</div>
        <div class="bar-wrap">
          <div id="bar" class="bar"></div>
        </div>

        <div class="metrics">
          <div class="metric-box">
            <div class="metric-title">Ganó en el último tramo</div>
            <div class="metric-value">+{payload["delta_value"]:,}</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Ritmo reciente</div>
            <div class="metric-value">{payload["per_min"]:.1f} votos/min</div>
          </div>
        </div>
      </div>

      <script>
        const startValue = {payload["prev_value"]};
        const endValue = {payload["last_value"]};
        const duration = {payload["duration_ms"]};
        const progressStart = {payload["progress_start"]};
        const progressEnd = {payload["progress_end"]};

        const animatedValue = document.getElementById("animatedValue");
        const bar = document.getElementById("bar");

        function formatNumber(n) {{
          return Math.round(n).toLocaleString("es-ES");
        }}

        function animate() {{
          const startTime = performance.now();

          function step(now) {{
            const elapsed = now - startTime;
            const t = Math.min(elapsed / duration, 1);

            const currentValue = startValue + (endValue - startValue) * t;
            const currentProgress = progressStart + (progressEnd - progressStart) * t;

            animatedValue.textContent = formatNumber(currentValue);
            bar.style.width = (currentProgress * 100).toFixed(2) + "%";

            if (t < 1) {{
              requestAnimationFrame(step);
            }}
          }}

          requestAnimationFrame(step);
        }}

        animate();
      </script>
    </body>
    </html>
    """

    components.html(html, height=410)


# =====================================
# REFRESCO AUTOMÁTICO
# =====================================

st_autorefresh(interval=PAGE_REFRESH_MS, key="conteo-en-vivo-v6-refresh")


# =====================================
# INTERFAZ
# =====================================

df = load_data()
actas_df = load_actas_data()

if df is None or df.empty:
    st.warning("No se encontró información en onpe_tracking_v2.csv. Primero deja corriendo onpe_watch_v2.py.")
    st.stop()

snapshot = build_snapshot(df)
prev_row = snapshot["prev"]
last_row = snapshot["last"]

official_gap = last_row["porky_votes"] - last_row["nieto_votes"]
leader = "🐷 Porky" if official_gap > 0 else "🌞 Nieto" if official_gap < 0 else "Empate"

st.markdown('<div class="main-title">CONTEO EN VIVO 📊🔥 ¿QUIEN PASA A SEGUNDA VUELTA CONTRA LA K? </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Seguimiento visual en tiempo real con datos confirmados y una reproducción animada del último tramo real para mantener continuidad. Los contadores en movimiento no representan una proyección del próximo corte.</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Votos confirmados — 🐷 Porky", fmt_int(last_row["porky_votes"]))
k2.metric("Votos confirmados — 🌞 Nieto", fmt_int(last_row["nieto_votes"]))
k3.metric("Votos confirmados — 🤠 Sánchez", fmt_int(last_row["sanchez_votes"]))
k4.metric("Brecha confirmada — 🐷 Porky vs 🌞 Nieto", fmt_int(abs(official_gap)), delta=leader)

st.markdown('<div class="section-title">Avance del conteo nacional</div>', unsafe_allow_html=True)

if actas_df is not None and not actas_df.empty:
    last_actas = actas_df.iloc[-1]
    actas_pct = float(last_actas["actas_pct"])

    a1, a2 = st.columns([1, 3])

    with a1:
        st.metric("Actas contabilizadas", f"{actas_pct:.3f}%")

    with a2:
        st.markdown(
            f"""
            <div style="
                width:100%;
                height:24px;
                background: rgba(255,255,255,0.08);
                border-radius:999px;
                overflow:hidden;
                margin-top: 24px;
            ">
                <div style="
                    width:{actas_pct:.3f}%;
                    height:100%;
                    background: linear-gradient(90deg, #4facfe, #00f2fe);
                    border-radius:999px;
                "></div>
            </div>
            <div style="margin-top:8px; opacity:0.82; font-size:0.95rem;">
                Progreso aproximado del conteo oficial
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Todavía no hay datos de actas contabilizadas.")

st.markdown('<div class="section-title">Lectura gráfica</div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    fig1 = plot_curvas_votos(df)
    st.pyplot(fig1, clear_figure=True)

with g2:
    fig2 = plot_brecha(df)
    st.pyplot(fig2, clear_figure=True)

st.markdown('<div class="section-title">Panel de candidatos</div>', unsafe_allow_html=True)

max_value = max(
    safe_int(last_row["porky_votes"]),
    safe_int(last_row["nieto_votes"]),
    safe_int(last_row["sanchez_votes"]),
    1
)

c1, c2, c3 = st.columns(3)

with c1:
    animated_candidate_card(
        "Porky",
        "🐷",
        prev_row["porky_votes"],
        last_row["porky_votes"],
        last_row["delta_porky"],
        last_row["porky_per_min_ma3"],
        PAGE_REFRESH_MS,
        max_value,
        "linear-gradient(90deg, #ff8c42, #ffb26b)"
    )

with c2:
    animated_candidate_card(
        "Nieto",
        "🌞",
        prev_row["nieto_votes"],
        last_row["nieto_votes"],
        last_row["delta_nieto"],
        last_row["nieto_per_min_ma3"],
        PAGE_REFRESH_MS,
        max_value,
        "linear-gradient(90deg, #f6d365, #fda085)"
    )

with c3:
    animated_candidate_card(
        "Sánchez",
        "🤠",
        prev_row["sanchez_votes"],
        last_row["sanchez_votes"],
        last_row["delta_sanchez"],
        last_row["sanchez_per_min_ma3"],
        PAGE_REFRESH_MS,
        max_value,
        "linear-gradient(90deg, #84fab0, #8fd3f4)"
    )

st.markdown('<div class="section-title">Cómo leer esta pantalla</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="soft-card"><strong>Qué es oficial</strong><br><br>Los números superiores muestran el último dato confirmado.<br>La brecha principal también usa el último dato confirmado.<br><br><span class="badge-line">Último dato confirmado: {}</span></div>'.format(
            last_row["timestamp"]
        ),
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        '<div class="soft-card"><strong>Qué está animado</strong><br><br>El contador en movimiento reproduce la subida del último tramo real.<br>La barra también recorre ese mismo último tramo.<br><br><span class="mini-note">No es una proyección del próximo corte.</span></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="section-title">Ritmo reciente</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
r1.metric("🐷 Porky — votos/min", f"{last_row['porky_per_min_ma3']:.1f}")
r2.metric("🌞 Nieto — votos/min", f"{last_row['nieto_per_min_ma3']:.1f}")
r3.metric("🤠 Sánchez — votos/min", f"{last_row['sanchez_per_min_ma3']:.1f}")

st.markdown('<div class="section-title">Lectura del último tramo real</div>', unsafe_allow_html=True)
for insight in build_insights(df):
    st.write("- " + insight)

st.markdown('<div class="section-title">Últimos registros oficiales</div>', unsafe_allow_html=True)

display_df = df.tail(15).copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

display_df = display_df.rename(columns={
    "timestamp": "Hora",
    "porky_votes": "🐷 Porky",
    "nieto_votes": "🌞 Nieto",
    "sanchez_votes": "🤠 Sánchez",
    "diff_porky_nieto": "📏 Brecha Porky-Nieto",
    "leader": "Líder",
    "delta_porky": "Δ 🐷",
    "delta_nieto": "Δ 🌞",
    "delta_sanchez": "Δ 🤠",
    "porky_per_min": "🐷 votos/min",
    "nieto_per_min": "🌞 votos/min",
    "sanchez_per_min": "🤠 votos/min",
})

st.dataframe(display_df, width="stretch")