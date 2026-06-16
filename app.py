"""
EduPro Analytics & Machine Learning Dashboard
=============================================
A production-quality Streamlit dashboard for educational data analytics and ML insights.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — PROFESSIONAL DARK-ACCENT THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Root Variables ---------- */
:root {
    --bg-primary:   #0f1117;
    --bg-card:      #1a1d27;
    --bg-card2:     #20243a;
    --accent:       #6c63ff;
    --accent2:      #00d4aa;
    --accent3:      #ff6b6b;
    --accent4:      #ffd166;
    --text-primary: #f0f2f8;
    --text-muted:   #8b90a7;
    --border:       #2a2f45;
    --radius:       14px;
}

/* ---------- Global Resets ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-primary) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13162a 0%, #0f1117 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] label { color: var(--text-muted) !important; }

/* ---------- Multiselect / Selectbox ---------- */
[data-testid="stMultiSelect"] > div,
[data-testid="stSelectbox"] > div { 
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ---------- KPI Cards ---------- */
.kpi-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(108,99,255,0.15);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-purple::before  { background: linear-gradient(90deg, #6c63ff, #a78bfa); }
.kpi-teal::before    { background: linear-gradient(90deg, #00d4aa, #06b6d4); }
.kpi-coral::before   { background: linear-gradient(90deg, #ff6b6b, #f97316); }
.kpi-gold::before    { background: linear-gradient(90deg, #ffd166, #f59e0b); }

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-icon {
    font-size: 28px;
    float: right;
    margin-top: -4px;
    opacity: 0.7;
}
.kpi-delta {
    font-size: 12px;
    font-weight: 500;
    color: var(--accent2);
}

/* ---------- Section Headers ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 36px 0 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.section-header .icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.section-header h2 {
    font-size: 20px; font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-header .subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin: 0;
}

/* ---------- ML Metric Cards ---------- */
.ml-metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    text-align: center;
}
.ml-metric-value {
    font-size: 26px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.ml-metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ---------- Cluster Badge ---------- */
.cluster-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px;
}

/* ---------- Info Banner ---------- */
.info-banner {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(0,212,170,0.08));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 20px;
}
.info-banner p { margin: 0; color: var(--text-muted); font-size: 14px; }
.info-banner strong { color: var(--text-primary); }

/* ---------- Plotly Charts BG ---------- */
.js-plotly-plot { border-radius: var(--radius) !important; }

/* ---------- DataFrame styling ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ---------- Divider ---------- */
hr { border-color: var(--border) !important; margin: 28px 0 !important; }

/* ---------- Header Brand ---------- */
.brand-header {
    background: linear-gradient(135deg, #13162a 0%, #1a1d27 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.brand-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.brand-header h1 {
    font-size: 30px; font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
}
.brand-header p {
    color: var(--text-muted);
    font-size: 14px; margin: 0;
}
.brand-badge {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.4);
    color: #a78bfa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS & PLOTLY THEME
# ─────────────────────────────────────────────
CHART_BG      = "#1a1d27"
PAPER_BG      = "#1a1d27"
FONT_COLOR    = "#f0f2f8"
GRID_COLOR    = "#2a2f45"
PALETTE       = ["#6c63ff", "#00d4aa", "#ff6b6b", "#ffd166", "#a78bfa", "#06b6d4", "#f97316", "#ec4899"]

def chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=FONT_COLOR, family="Inter"), x=0.01),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Inter"),
        height=height,
        margin=dict(t=50, b=40, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(size=11), title_font=dict(size=12)),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(size=11), title_font=dict(size=12)),
    )
    return fig

def section(icon_emoji, title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
      <div class="icon">{icon_emoji}</div>
      <div>
        <h2>{title}</h2>
        {"<p class='subtitle'>" + subtitle + "</p>" if subtitle else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

def kpi(label, value, icon, color_class, delta=""):
    st.markdown(f"""
    <div class="kpi-card {color_class}">
      <span class="kpi-icon">{icon}</span>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {f'<div class="kpi-delta">▲ {delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def ml_metric(value, label, color):
    st.markdown(f"""
    <div class="ml-metric-card">
      <div class="ml-metric-value" style="color:{color}">{value}</div>
      <div class="ml-metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("edupro_master_v2.csv")
    except FileNotFoundError:
        # ── Synthetic fallback ──────────────────────────────
        np.random.seed(42)
        n = 1200
        categories = ["Programming", "Data Science", "Design", "Business", "Marketing", "Cloud", "Cybersecurity"]
        levels     = ["Beginner", "Intermediate", "Advanced"]
        teachers   = [f"Instructor_{chr(65+i)}" for i in range(20)]
        expertise_map = {t: np.random.choice(categories) for t in teachers}
        df = pd.DataFrame({
            "UserID":           [f"U{1000+i}" for i in range(n)],
            "CourseID":         [f"C{np.random.randint(100,200)}" for _ in range(n)],
            "TeacherID":        [f"T{np.random.randint(10,30)}"  for _ in range(n)],
            "Amount":           np.random.randint(500, 15000, n),
            "CourseCategory":   np.random.choice(categories, n),
            "CourseLevel":      np.random.choice(levels, n, p=[0.4,0.38,0.22]),
            "CoursePrice":      np.random.randint(999, 9999, n),
            "CourseDuration":   np.random.randint(5, 120, n),
            "TeacherName":      np.random.choice(teachers, n),
            "TeacherRating":    np.round(np.random.uniform(3.0, 5.0, n), 1),
            "YearsOfExperience":np.random.randint(1, 20, n),
            "Expertise":        [expertise_map[t] for t in np.random.choice(teachers, n)],
            "Cluster":          np.random.randint(0, 4, n),
        })
        st.sidebar.warning("⚠️ Using synthetic demo data — place `edupro_master_v2.csv` in the same directory for real data.")
    return df

df_raw = load_data()


# ─────────────────────────────────────────────
# SIDEBAR — BRAND + FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
      <div style="font-size:40px; margin-bottom:4px;">🎓</div>
      <div style="font-size:17px; font-weight:800; background:linear-gradient(135deg,#6c63ff,#00d4aa);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text;">EduPro Analytics</div>
      <div style="font-size:11px; color:#8b90a7; margin-top:2px;">ML Dashboard v2.0</div>
    </div>
    <hr style="margin:14px 0 20px; border-color:#2a2f45;">
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Filters")

    all_categories = sorted(df_raw["CourseCategory"].dropna().unique().tolist())
    sel_categories = st.multiselect("📚 Course Category", all_categories, default=all_categories,
                                     help="Filter by course category")

    all_levels = sorted(df_raw["CourseLevel"].dropna().unique().tolist())
    sel_levels = st.multiselect("📊 Course Level", all_levels, default=all_levels,
                                 help="Filter by difficulty level")

    all_expertise = sorted(df_raw["Expertise"].dropna().unique().tolist())
    sel_expertise = st.multiselect("🏅 Teacher Expertise", all_expertise, default=all_expertise,
                                    help="Filter by teacher expertise area")

    st.markdown("<hr style='border-color:#2a2f45; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 ML Settings")
    n_clusters  = st.slider("K-Means Clusters", 2, 8, 4)
    test_size   = st.slider("Train/Test Split", 0.1, 0.4, 0.2, 0.05,
                             help="Proportion held out for testing")
    show_raw    = st.checkbox("Show Raw Dataset", value=True)

    st.markdown("""
    <hr style='border-color:#2a2f45; margin:20px 0;'>
    <div style="font-size:11px; color:#8b90a7; text-align:center;">
      Built with Streamlit + Plotly<br>
      <span style="color:#6c63ff; font-weight:600;">EduPro Analytics © 2025</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()
if sel_categories: df = df[df["CourseCategory"].isin(sel_categories)]
if sel_levels:     df = df[df["CourseLevel"].isin(sel_levels)]
if sel_expertise:  df = df[df["Expertise"].isin(sel_expertise)]

if df.empty:
    st.warning("⚠️ No data matches the current filters. Please broaden your selection.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER BRAND BLOCK
# ─────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
  <div class="brand-badge">🎓 Education Intelligence Platform</div>
  <h1>EduPro Analytics & Machine Learning Dashboard</h1>
  <p>Real-time insights into courses, instructors, revenue, and predictive ML models — all in one place.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
section("📌", "Key Performance Indicators", "High-level metrics from filtered data")

k1, k2, k3, k4 = st.columns(4)
total_users    = df["UserID"].nunique()
total_teachers = df["TeacherID"].nunique()
total_courses  = df["CourseID"].nunique()
total_revenue  = df["Amount"].sum()

with k1: kpi("Total Users",    f"{total_users:,}",    "👥", "kpi-purple",  f"{total_users} enrolled")
with k2: kpi("Total Teachers", f"{total_teachers:,}", "👨‍🏫", "kpi-teal",   f"{total_teachers} active")
with k3: kpi("Total Courses",  f"{total_courses:,}",  "📚", "kpi-coral",   f"{total_courses} available")
with k4: kpi("Total Revenue",  f"₹{total_revenue:,.0f}", "💰", "kpi-gold", "cumulative")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — COURSE ANALYTICS
# ─────────────────────────────────────────────
section("📊", "Course Analytics", "Distribution and breakdown of course inventory")

col1, col2 = st.columns(2)

with col1:
    cat_counts = df["CourseCategory"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.bar(cat_counts, x="Count", y="Category", orientation="h",
                 color="Count", color_continuous_scale=["#6c63ff","#00d4aa"],
                 text="Count")
    fig.update_traces(textfont_size=11, textposition="outside",
                      marker_line_width=0, hovertemplate="%{y}: %{x} courses<extra></extra>")
    fig.update_coloraxes(showscale=False)
    fig = chart_layout(fig, "Course Category Distribution")
    fig.update_layout(yaxis_categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    level_counts = df["CourseLevel"].value_counts().reset_index()
    level_counts.columns = ["Level", "Count"]
    colors_level = {"Beginner":"#00d4aa","Intermediate":"#6c63ff","Advanced":"#ff6b6b"}
    fig2 = px.pie(level_counts, values="Count", names="Level",
                  color="Level", color_discrete_map=colors_level,
                  hole=0.52)
    fig2.update_traces(textinfo="label+percent", pull=[0.05]*len(level_counts),
                       marker_line=dict(color=CHART_BG, width=2),
                       hovertemplate="%{label}: %{value} (%{percent})<extra></extra>")
    fig2 = chart_layout(fig2, "Course Level Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# Course Duration by Category (box plot)
fig3 = px.box(df, x="CourseCategory", y="CourseDuration",
              color="CourseCategory", color_discrete_sequence=PALETTE,
              points="outliers")
fig3.update_traces(hovertemplate="%{x}<br>Duration: %{y} hrs<extra></extra>")
fig3 = chart_layout(fig3, "Course Duration Distribution by Category", height=340)
fig3.update_layout(showlegend=False, xaxis_title="", yaxis_title="Duration (hours)")
st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 2 — INSTRUCTOR INSIGHTS
# ─────────────────────────────────────────────
section("👨‍🏫", "Instructor Insights", "Teacher ratings, experience, and top performers")

col1, col2 = st.columns(2)

with col1:
    fig4 = px.histogram(df, x="TeacherRating", nbins=20,
                        color_discrete_sequence=["#6c63ff"])
    fig4.update_traces(marker_line_width=0, opacity=0.85,
                       hovertemplate="Rating: %{x}<br>Count: %{y}<extra></extra>")
    fig4.add_vline(x=df["TeacherRating"].mean(), line_dash="dash",
                   line_color="#00d4aa", annotation_text=f"Mean: {df['TeacherRating'].mean():.2f}",
                   annotation_font_color="#00d4aa", annotation_position="top right")
    fig4 = chart_layout(fig4, "Teacher Rating Distribution")
    fig4.update_layout(xaxis_title="Rating", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    fig5 = px.histogram(df, x="YearsOfExperience", nbins=18,
                        color_discrete_sequence=["#00d4aa"])
    fig5.update_traces(marker_line_width=0, opacity=0.85)
    fig5.add_vline(x=df["YearsOfExperience"].mean(), line_dash="dash",
                   line_color="#ffd166", annotation_text=f"Mean: {df['YearsOfExperience'].mean():.1f} yrs",
                   annotation_font_color="#ffd166", annotation_position="top right")
    fig5 = chart_layout(fig5, "Years of Experience Distribution")
    fig5.update_layout(xaxis_title="Years", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

# Top 10 Instructors by Rating
top10 = (df.groupby("TeacherName")
           .agg(Avg_Rating=("TeacherRating","mean"),
                Total_Students=("UserID","count"),
                Avg_Experience=("YearsOfExperience","mean"))
           .reset_index()
           .sort_values("Avg_Rating", ascending=False)
           .head(10))
top10["Avg_Rating"] = top10["Avg_Rating"].round(2)

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    y=top10["TeacherName"], x=top10["Avg_Rating"],
    orientation="h", name="Avg Rating",
    marker=dict(color=top10["Avg_Rating"],
                colorscale=[[0,"#6c63ff"],[0.5,"#a78bfa"],[1,"#00d4aa"]],
                line_width=0),
    text=[f"⭐ {r}" for r in top10["Avg_Rating"]],
    textposition="outside",
    hovertemplate="%{y}<br>Rating: %{x:.2f}<extra></extra>",
))
fig6 = chart_layout(fig6, "🏆 Top 10 Instructors by Average Rating", height=380)
fig6.update_layout(yaxis_categoryorder="total ascending", showlegend=False,
                   xaxis_title="Average Rating", yaxis_title="")
st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3 — REVENUE & PRICING
# ─────────────────────────────────────────────
section("💰", "Revenue & Pricing Analysis", "Revenue trends by category, level and price distribution")

col1, col2 = st.columns(2)

with col1:
    rev_cat = df.groupby("CourseCategory")["Amount"].sum().reset_index()
    rev_cat.columns = ["Category","Revenue"]
    rev_cat = rev_cat.sort_values("Revenue", ascending=False)
    fig7 = px.bar(rev_cat, x="Category", y="Revenue",
                  color="Revenue", color_continuous_scale=["#6c63ff","#ffd166"],
                  text=rev_cat["Revenue"].apply(lambda x: f"₹{x/1000:.1f}K"))
    fig7.update_traces(marker_line_width=0, textposition="outside",
                       hovertemplate="%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>")
    fig7.update_coloraxes(showscale=False)
    fig7 = chart_layout(fig7, "Revenue by Course Category")
    fig7.update_layout(xaxis_title="", yaxis_title="Total Revenue (₹)", showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    rev_lvl = df.groupby("CourseLevel")["Amount"].sum().reset_index()
    rev_lvl.columns = ["Level","Revenue"]
    fig8 = px.funnel(rev_lvl, x="Revenue", y="Level",
                     color="Level", color_discrete_sequence=PALETTE)
    fig8.update_traces(hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>")
    fig8 = chart_layout(fig8, "Revenue Funnel by Course Level")
    st.plotly_chart(fig8, use_container_width=True)

# Price distribution violin
fig9 = px.violin(df, x="CourseLevel", y="CoursePrice",
                 color="CourseLevel", color_discrete_sequence=PALETTE,
                 box=True, points="outliers")
fig9.update_traces(hovertemplate="%{x}<br>Price: ₹%{y:,.0f}<extra></extra>")
fig9 = chart_layout(fig9, "Course Price Distribution by Level", height=360)
fig9.update_layout(xaxis_title="", yaxis_title="Course Price (₹)", showlegend=False)
st.plotly_chart(fig9, use_container_width=True)

# Revenue scatter: Rating vs Amount
fig10 = px.scatter(df, x="TeacherRating", y="Amount",
                   color="CourseCategory", size="CoursePrice",
                   color_discrete_sequence=PALETTE,
                   opacity=0.65,
                   hover_data=["TeacherName","CourseLevel"])
fig10 = chart_layout(fig10, "Revenue vs Teacher Rating (bubble = course price)", height=400)
fig10.update_layout(xaxis_title="Teacher Rating", yaxis_title="Amount (₹)")
st.plotly_chart(fig10, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4 — CLUSTERING INSIGHTS
# ─────────────────────────────────────────────
section("🔵", "Clustering Insights", "Segment analysis using the Cluster column from the dataset")

cluster_colors = {0:"#6c63ff", 1:"#00d4aa", 2:"#ff6b6b", 3:"#ffd166",
                  4:"#a78bfa", 5:"#06b6d4", 6:"#f97316", 7:"#ec4899"}

col1, col2 = st.columns([1.2, 1])

with col1:
    cluster_summary = df.groupby("Cluster").agg(
        Count=("UserID","count"),
        Avg_Revenue=("Amount","mean"),
        Avg_Rating=("TeacherRating","mean"),
        Avg_Price=("CoursePrice","mean"),
        Avg_Experience=("YearsOfExperience","mean"),
    ).reset_index().round(2)

    fig_cl = px.scatter(cluster_summary, x="Avg_Rating", y="Avg_Revenue",
                        size="Count", color="Cluster",
                        color_discrete_sequence=PALETTE,
                        text="Cluster",
                        hover_data=["Count","Avg_Price","Avg_Experience"],
                        size_max=55)
    fig_cl.update_traces(textfont=dict(color="white",size=11), textposition="middle center",
                          hovertemplate="Cluster %{text}<br>Rating: %{x:.2f}<br>Revenue: ₹%{y:,.0f}<extra></extra>")
    fig_cl = chart_layout(fig_cl, "Cluster Map: Rating vs Avg Revenue", height=380)
    fig_cl.update_layout(showlegend=False, xaxis_title="Avg Teacher Rating",
                          yaxis_title="Avg Revenue (₹)")
    st.plotly_chart(fig_cl, use_container_width=True)

with col2:
    clust_dist = df["Cluster"].value_counts().reset_index()
    clust_dist.columns = ["Cluster","Count"]
    clust_dist["Cluster"] = clust_dist["Cluster"].astype(str)
    fig_cd = px.pie(clust_dist, values="Count", names="Cluster",
                    color_discrete_sequence=PALETTE, hole=0.5)
    fig_cd.update_traces(textinfo="label+percent",
                          marker_line=dict(color=CHART_BG, width=2),
                          hovertemplate="Cluster %{label}: %{value}<extra></extra>")
    fig_cd = chart_layout(fig_cd, "Cluster Size Distribution", height=380)
    st.plotly_chart(fig_cd, use_container_width=True)

# Cluster heatmap
pivot = df.groupby(["Cluster","CourseCategory"])["Amount"].mean().unstack(fill_value=0)
fig_heat = go.Figure(data=go.Heatmap(
    z=pivot.values, x=pivot.columns.tolist(), y=[f"Cluster {c}" for c in pivot.index],
    colorscale=[[0,"#1a1d27"],[0.3,"#6c63ff"],[0.7,"#a78bfa"],[1,"#00d4aa"]],
    hovertemplate="Cluster: %{y}<br>Category: %{x}<br>Avg Revenue: ₹%{z:,.0f}<extra></extra>",
    text=np.round(pivot.values/1000,1),
    texttemplate="%{text}K", textfont=dict(size=10),
))
fig_heat = chart_layout(fig_heat, "Cluster × Category — Average Revenue Heatmap", height=340)
fig_heat.update_layout(xaxis_title="", yaxis_title="")
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("📋 Cluster Summary Table"):
    st.dataframe(cluster_summary.style
                 .format({"Avg_Revenue":"₹{:,.0f}","Avg_Rating":"{:.2f}","Avg_Price":"₹{:,.0f}","Avg_Experience":"{:.1f} yrs"})
                 .background_gradient(cmap="Purples", subset=["Avg_Revenue"]),
                 use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 5 — MACHINE LEARNING
# ─────────────────────────────────────────────
section("🤖", "Machine Learning Models", "Regression, ensemble, and clustering models trained on filtered data")

# ── Feature Engineering ──────────────────────
@st.cache_data
def prepare_ml(data, ts):
    d = data.copy()
    le = LabelEncoder()
    for col in ["CourseCategory","CourseLevel","Expertise"]:
        if col in d.columns:
            d[col+"_enc"] = le.fit_transform(d[col].astype(str))
    features = [c for c in ["CoursePrice","CourseDuration","TeacherRating",
                              "YearsOfExperience","CourseCategory_enc",
                              "CourseLevel_enc","Expertise_enc"] if c in d.columns]
    target = "Amount"
    d = d.dropna(subset=features+[target])
    X, y = d[features], d[target]
    return train_test_split(X, y, test_size=ts, random_state=42), features

(X_train, X_test, y_train, y_test), feat_names = prepare_ml(df, test_size)

# ── Linear Regression ────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred   = lr.predict(X_test)
lr_r2     = r2_score(y_test, lr_pred)
lr_rmse   = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae    = mean_absolute_error(y_test, lr_pred)

# ── Random Forest ────────────────────────────
rf = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred   = rf.predict(X_test)
rf_r2     = r2_score(y_test, rf_pred)
rf_rmse   = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae    = mean_absolute_error(y_test, rf_pred)

# ── K-Means ──────────────────────────────────
scaler = StandardScaler()
cluster_features = [c for c in ["CoursePrice","TeacherRating","YearsOfExperience","Amount"] if c in df.columns]
X_km = scaler.fit_transform(df[cluster_features].dropna())
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
km_labels = km.fit_predict(X_km)
km_inertia = km.inertia_


# ── Performance Metric Cards ────────────────
st.markdown("### 📐 Model Performance Metrics")
mA, mB, mC, mD, mE, mF = st.columns(6)
with mA: ml_metric(f"{lr_r2:.3f}",  "LR — R²",   "#6c63ff")
with mB: ml_metric(f"₹{lr_rmse:,.0f}", "LR — RMSE", "#a78bfa")
with mC: ml_metric(f"₹{lr_mae:,.0f}",  "LR — MAE",  "#c4b5fd")
with mD: ml_metric(f"{rf_r2:.3f}",  "RF — R²",   "#00d4aa")
with mE: ml_metric(f"₹{rf_rmse:,.0f}", "RF — RMSE", "#06b6d4")
with mF: ml_metric(f"₹{rf_mae:,.0f}",  "RF — MAE",  "#67e8f9")

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Comparison Scatter ────────────────
col1, col2 = st.columns(2)

with col1:
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(
        x=y_test.values, y=lr_pred,
        mode="markers", name="Predictions",
        marker=dict(color="#6c63ff", size=5, opacity=0.6, line=dict(width=0)),
        hovertemplate="Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>"
    ))
    mn, mx = float(y_test.min()), float(y_test.max())
    fig_lr.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                line=dict(color="#ffd166", dash="dash", width=1.5),
                                name="Perfect Fit"))
    fig_lr = chart_layout(fig_lr, f"Linear Regression — Actual vs Predicted (R²={lr_r2:.3f})")
    fig_lr.update_layout(xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)")
    st.plotly_chart(fig_lr, use_container_width=True)

with col2:
    fig_rf = go.Figure()
    fig_rf.add_trace(go.Scatter(
        x=y_test.values, y=rf_pred,
        mode="markers", name="Predictions",
        marker=dict(color="#00d4aa", size=5, opacity=0.6, line=dict(width=0)),
        hovertemplate="Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>"
    ))
    fig_rf.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                line=dict(color="#ffd166", dash="dash", width=1.5),
                                name="Perfect Fit"))
    fig_rf = chart_layout(fig_rf, f"Random Forest — Actual vs Predicted (R²={rf_r2:.3f})")
    fig_rf.update_layout(xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)")
    st.plotly_chart(fig_rf, use_container_width=True)

# ── Model Comparison Bar ─────────────────────
col1, col2 = st.columns(2)

with col1:
    metrics_df = pd.DataFrame({
        "Model":  ["Linear Regression", "Random Forest"],
        "R²":     [lr_r2, rf_r2],
        "RMSE":   [lr_rmse, rf_rmse],
        "MAE":    [lr_mae, rf_mae],
    })
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="R²",  x=metrics_df["Model"], y=metrics_df["R²"],
                              marker_color="#6c63ff", yaxis="y", text=metrics_df["R²"].round(3),
                              textposition="outside"))
    fig_cmp.update_layout(barmode="group")
    fig_cmp = chart_layout(fig_cmp, "Model R² Comparison")
    fig_cmp.update_layout(showlegend=False, yaxis_title="R² Score",
                           yaxis_range=[0, max(lr_r2, rf_r2)*1.2])
    st.plotly_chart(fig_cmp, use_container_width=True)

with col2:
    # Feature Importance (RF)
    importance_df = pd.DataFrame({
        "Feature":   feat_names,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#6c63ff","#00d4aa"],
                     text=importance_df["Importance"].round(3))
    fig_imp.update_traces(textposition="outside", marker_line_width=0)
    fig_imp.update_coloraxes(showscale=False)
    fig_imp = chart_layout(fig_imp, "Random Forest — Feature Importance")
    fig_imp.update_layout(showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# ── K-Means Visualization ───────────────────
st.markdown("### 🔵 K-Means Clustering")
km_df = df[cluster_features].dropna().copy()
km_df["KM_Cluster"] = km_labels.astype(str)

col1, col2 = st.columns(2)
with col1:
    fig_km = px.scatter(km_df, x=cluster_features[0], y=cluster_features[1],
                         color="KM_Cluster", color_discrete_sequence=PALETTE,
                         opacity=0.65, title="")
    fig_km.update_traces(marker_size=4, marker_line_width=0)
    fig_km = chart_layout(fig_km, f"K-Means (k={n_clusters}) — Price vs Rating", height=360)
    fig_km.update_layout(xaxis_title=cluster_features[0], yaxis_title=cluster_features[1])
    st.plotly_chart(fig_km, use_container_width=True)

with col2:
    # Elbow curve
    inertias = []
    ks = range(2, min(10, len(km_df)))
    for k in ks:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_km).inertia_)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(ks), y=inertias, mode="lines+markers",
                                    line=dict(color="#6c63ff", width=2),
                                    marker=dict(size=7, color="#00d4aa"),
                                    hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"))
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#ffd166",
                         annotation_text=f"k={n_clusters} (selected)",
                         annotation_font_color="#ffd166")
    fig_elbow = chart_layout(fig_elbow, "Elbow Curve — Optimal k", height=360)
    fig_elbow.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 6 — DATASET PREVIEW
# ─────────────────────────────────────────────
if show_raw:
    section("🗃️", "Dataset Preview", f"Showing {min(200, len(df))} of {len(df):,} filtered records")

    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Descriptive Stats", "🔍 Column Info"])

    with tab1:
        st.dataframe(df.head(200), use_container_width=True, height=360)

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(df[num_cols].describe().T
                     .style.background_gradient(cmap="Purples", subset=["mean","std"]),
                     use_container_width=True)

    with tab3:
        info_df = pd.DataFrame({
            "Column":    df.columns.tolist(),
            "Dtype":     [str(df[c].dtype) for c in df.columns],
            "Non-Null":  [df[c].notna().sum() for c in df.columns],
            "Null %":    [(df[c].isna().mean()*100).round(2) for c in df.columns],
            "Unique":    [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px; color:#8b90a7; font-size:13px;
            border-top:1px solid #2a2f45; margin-top:20px;">
  🎓 <strong style="color:#6c63ff;">EduPro Analytics & ML Dashboard</strong> — 
  Built with Streamlit · Plotly · scikit-learn &nbsp;|&nbsp; 
  <span style="color:#00d4aa;">v2.0 · 2025</span>
</div>
""", unsafe_allow_html=True)
