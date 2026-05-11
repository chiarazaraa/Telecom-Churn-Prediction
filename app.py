import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title='Customer Retention Intelligence',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
html { color-scheme: light; }
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #ffffff !important;
    color: #111827 !important;
}
[data-testid="stTabs"], [data-testid="stTabsContent"],
button[data-baseweb="tab"] {
    background-color: #ffffff !important;
    color: #374151 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    font-weight: 600 !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}
[data-testid="metric-container"] { background-color: #f8fafd !important; }
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricDelta"] { color: #111827 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #111827 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.5rem;
    margin: 1.8rem 0 1rem 0;
    letter-spacing: -0.01em;
}
.callout {
    background: #f0f7ff;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1.1rem;
    font-size: 0.875rem;
    color: #1e3a8a;
    margin: 0.75rem 0 1.25rem 0;
    line-height: 1.6;
}
.exec-bullet {
    background: #f8fafd;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.875rem;
    color: #111827;
    line-height: 1.65;
}
.section-sep { height: 1px; background: #e5e7eb; margin: 2rem 0; }
.tech-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.getenv('AIRFLOW_HOME', '/opt/airflow')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
EDA_DIR    = os.path.join(OUTPUT_DIR, 'eda')

# ---------------------------------------------------------------------------
# Data loading — unchanged
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = {}
    for name, path in [
        ('evaluation',   os.path.join(OUTPUT_DIR, 'final_evaluation.json')),
        ('selection',    os.path.join(OUTPUT_DIR, 'model_selection_report.json')),
        ('segmentation', os.path.join(OUTPUT_DIR, 'business_segmentation.json')),
        ('drivers',      os.path.join(OUTPUT_DIR, 'top_churn_drivers.json')),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                data[name] = json.load(f)
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    if os.path.exists(pred_path):
        data['predictions'] = pd.read_csv(pred_path)
    return data

# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def metric_card(label, value, subtitle=None, accent='#2563eb'):
    sub = (f"<p style='color:#6b7280;font-size:0.78rem;margin:0.3rem 0 0 0;"
           f"line-height:1.4;'>{subtitle}</p>") if subtitle else ""
    st.markdown(f"""
        <div style='background:#f8fafd;border:1px solid #e5e7eb;border-radius:10px;
                    padding:1.3rem 1.5rem;border-top:3px solid {accent};'>
            <p style='color:#6b7280;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;margin:0;'>{label}</p>
            <p style='color:#111827;font-size:1.9rem;font-weight:700;
                      margin:0.35rem 0 0 0;line-height:1;'>{value}</p>
            {sub}
        </div>
    """, unsafe_allow_html=True)


def insight_card(title, description, action, accent='#2563eb'):
    st.markdown(f"""
        <div style='background:#f8fafd;border:1px solid #e5e7eb;border-radius:8px;
                    padding:1.2rem 1.4rem;border-left:3px solid {accent};height:100%;'>
            <p style='color:#111827;font-weight:700;font-size:0.88rem;margin:0 0 0.5rem 0;'>{title}</p>
            <p style='color:#374151;font-size:0.83rem;margin:0 0 0.8rem 0;line-height:1.55;'>{description}</p>
            <p style='color:{accent};font-size:0.77rem;font-weight:700;margin:0;text-transform:uppercase;
                      letter-spacing:0.04em;'>Recommended action</p>
            <p style='color:#374151;font-size:0.83rem;margin:0.2rem 0 0 0;'>{action}</p>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Feature → business language
# ---------------------------------------------------------------------------
FEATURE_MAP = {
    'eqpdays':            ('Equipment Age',          '#dc2626',
                           'Customers with devices older than ~500 days show markedly higher churn intent — they are often waiting for a reason to switch operators.',
                           'Launch subsidised device upgrade campaigns for all accounts with equipment age > 18 months.'),
    'old_phone':          ('Outdated Device',         '#dc2626',
                           'Binary flag for equipment age > 500 days. A strong, non-linear churn predictor that captures the step-change effect of very old hardware.',
                           'Trigger device refresh offers for all flagged accounts before a competitor does.'),
    'months':             ('Customer Tenure',         '#d97706',
                           'Shorter-tenure customers churn at higher rates. The first 12 months represent the highest attrition risk window in the customer lifecycle.',
                           'Deploy structured early-lifecycle retention programmes targeting customers in their first year.'),
    'mou_Mean':           ('Average Call Volume',     '#d97706',
                           'Below-average minutes-of-use signals weakening product dependency — a consistent leading indicator of churn.',
                           'Re-engagement campaigns with usage incentives when monthly usage drops > 20%.'),
    'drop_rate':          ('Call Drop Rate',          '#dc2626',
                           'Elevated call drop rates directly correlate with customer dissatisfaction and accelerate churn decisions.',
                           'Priority network quality intervention for high-risk accounts with elevated drop rates.'),
    'recent_usage_delta': ('Declining Usage Trend',   '#d97706',
                           'Customers whose 3-month usage is below their 6-month baseline are showing early warning signals of disengagement.',
                           'Proactive outreach when usage trend turns negative for two consecutive months.'),
    'rev_Mean':           ('Revenue Volatility',      '#d97706',
                           'Revenue instability often indicates customers exploring competitor offers or reducing service consumption.',
                           'Offer loyalty pricing or plan renegotiation to stabilise spend and reinforce commitment.'),
    'change_mou':         ('Usage Change Rate',       '#d97706',
                           'A sustained downward trend in call volume is one of the strongest leading predictors of imminent churn.',
                           'Include in real-time churn alert triggers for call centre escalation queues.'),
    'totrev':             ('Total Billed Revenue',    '#059669',
                           'Lower total revenue customers exhibit higher churn sensitivity, often driven by a perceived value gap.',
                           'Apply segment-specific retention budgets — ROI is higher when focused on revenue-at-risk accounts.'),
    'avgmou':             ('Average Usage',           '#059669',
                           'Customers with consistently below-average usage have weaker product dependency and are easier for competitors to poach.',
                           'Bundle value-added services to increase stickiness for low-usage accounts.'),
}

# ---------------------------------------------------------------------------
# Load + derive shared values
# ---------------------------------------------------------------------------
data = load_data()

ev           = data.get('evaluation', {})
model_name   = ev.get('model', 'Best Model')
metrics_data = ev.get('metrics', {})

# In final_evaluation.json these are saved under "threshold_optimization"
thresholds   = ev.get('threshold_optimization', {})

# Prefer standalone segmentation file; fallback to embedded evaluation segmentation
seg          = data.get('segmentation', {}) or ev.get('business_segmentation', {})

auc     = metrics_data.get('AUC-ROC', {})
biz_thr = thresholds.get('business_threshold', None)
top20_capture = seg.get('top_20_percent', {}).get('churn_captured_pct', None)

# Business-threshold precision & recall are already saved in final_evaluation.json
biz_metrics = ev.get('business_threshold_metrics', {})
biz_prec = biz_metrics.get('precision_business', None)
biz_rec  = biz_metrics.get('recall_business', None)

# Aggregated SHAP from top_churn_drivers.json
feature_counts = {}
if 'drivers' in data:
    for _d in data['drivers']:
        for _dr in _d.get('top_drivers', []):
            _f = _dr['feature']
            feature_counts[_f] = feature_counts.get(_f, 0) + abs(_dr['shap_value'])
top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style='padding:1.4rem 0 0.6rem 0;border-bottom:1px solid #e5e7eb;margin-bottom:0.25rem;'>
    <h1 style='font-size:1.65rem;font-weight:700;color:#111827;margin:0;letter-spacing:-0.02em;'>
        Customer Retention Intelligence Platform
    </h1>
    <p style='color:#6b7280;margin:0.3rem 0 0 0;font-size:0.875rem;'>
        Prioritising retention interventions using calibrated churn risk scoring
    </p>
</div>
""", unsafe_allow_html=True)

if 'evaluation' not in data:
    st.warning('No pipeline results found. Please run the Airflow churn pipeline first.')
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    'Executive Overview',
    'Customer Segmentation',
    'Explainability & Drivers',
    'Technical Appendix',
])

# ============================================================
# TAB 1 — Executive Overview
# ============================================================
with tab1:

    # ── KPIs ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Performance at a Glance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        metric_card(
            'Model Discrimination',
            f"{auc.get('mean', 0):.3f}",
            f"AUC-ROC · 95% CI [{auc.get('ci_low', 0):.3f}–{auc.get('ci_high', 0):.3f}]",
        )
    with k2:
        metric_card(
            'Churners Identified — Top 20%',
            f"{top20_capture}%" if top20_capture else '—',
            'Of all churners captured by targeting the highest-risk 20%',
            accent='#059669',
        )
    with k3:
        metric_card(
            'Campaign Targeting Threshold',
            f"{biz_thr:.2f}" if biz_thr is not None else '—',
            'Calibrated to precision ≥ 60%, maximum recall',
            accent='#d97706',
        )
    with k4:
        metric_card(
            'Retention Campaign Accuracy',
            f"{biz_prec:.0%}" if biz_prec is not None else '—',
            'Of flagged customers are genuine churners',
            accent='#2563eb',
        )
    with k5:
        metric_card(
            'Churners Identified',
            f"{biz_rec:.0%}" if biz_rec is not None else '—',
            'Recall at the campaign targeting threshold',
            accent='#7c3aed',
        )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Executive Summary ────────────────────────────────────
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

    auc_val  = auc.get('mean', 0)
    auc_desc = 'strong' if auc_val >= 0.78 else 'good' if auc_val >= 0.72 else 'moderate'

    bullets = []

    bullets.append(
        f"<strong>Model performance:</strong> The retention model achieves a discrimination score of "
        f"<strong>{auc_val:.3f}</strong> (AUC-ROC), indicating {auc_desc} ability to separate likely "
        f"churners from stable customers — well above the random baseline of 0.500."
    )

    if top20_capture:
        lift20 = round(float(str(top20_capture).replace('%', '')) / 20, 1) if top20_capture else '—'
        bullets.append(
            f"<strong>Prioritisation efficiency:</strong> Targeting the top 20% of highest-risk customers "
            f"captures <strong>{top20_capture}%</strong> of all actual churners — a "
            f"<strong>{lift20}× lift</strong> over random outreach — enabling retention teams to "
            f"concentrate resources where they have the highest operational impact."
        )
    else:
        bullets.append(
            "<strong>Prioritisation efficiency:</strong> Risk-ranked customer lists enable retention "
            "teams to concentrate resources on the highest-impact accounts, avoiding wasted spend on "
            "low-risk customers unlikely to churn."
        )

    if biz_thr and biz_prec:
        bullets.append(
            f"<strong>Campaign viability:</strong> At the recommended threshold ({biz_thr:.2f}), "
            f"<strong>{biz_prec:.0%}</strong> of flagged customers are genuine churners. "
            f"This precision level makes targeted interventions economically viable and avoids "
            f"the customer experience cost of excessive false-positive contacts."
        )
    else:
        bullets.append(
            "<strong>Campaign viability:</strong> The campaign threshold is calibrated to maintain "
            "precision above 60%, ensuring retention spend is directed at customers who are "
            "genuinely at risk."
        )

    top_labels = [
        FEATURE_MAP.get(f, (f.replace('_', ' ').title(), '', '', ''))[0]
        for f, _ in top_features[:3]
    ]
    if top_labels:
        bullets.append(
            f"<strong>Primary churn drivers:</strong> {', '.join(top_labels)} are the leading "
            f"predictors of churn intent across the customer base. These signals are operationally "
            f"addressable through device upgrade campaigns, proactive service interventions, "
            f"and usage re-engagement programmes."
        )
    else:
        bullets.append(
            "<strong>Primary churn drivers:</strong> Equipment age, declining usage, and call "
            "quality are the leading churn signals — all directly addressable through targeted "
            "operational programmes."
        )

    for b in bullets:
        st.markdown(f'<div class="exec-bullet">{b}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Campaign Reach Simulator ─────────────────────────────
    st.markdown('<div class="section-title">Campaign Reach Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'Adjust the share of customers to contact — ranked by churn risk, highest first — '
        'to estimate campaign coverage and efficiency. Values are derived from model outputs; '
        'no retraining is required.'
        '</div>',
        unsafe_allow_html=True,
    )

    if 'predictions' in data:
        _sim = (data['predictions']
                .sort_values('churn_probability', ascending=False)
                .reset_index(drop=True))
        _tot        = len(_sim)
        _tot_churn  = int((_sim['true_label'] == 1).sum())

        contact_pct = st.slider(
            'Percentage of customers to contact (ranked by risk score)',
            min_value=1, max_value=50, value=20, step=1, format='%d%%',
            help='Customers are sorted highest risk first. Higher coverage captures more churners but reduces campaign precision.',
        )

        _n      = max(1, int(_tot * contact_pct / 100))
        _seg    = _sim.head(_n)
        _cap    = int((_seg['true_label'] == 1).sum())
        _caprt  = _cap / _tot_churn * 100 if _tot_churn > 0 else 0
        _cprec  = _cap / _n * 100
        _lift   = _caprt / contact_pct if contact_pct > 0 else 0

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            metric_card('Customers to Contact', f'{_n:,}', f'{contact_pct}% of {_tot:,} total')
        with sc2:
            metric_card('Churners Captured', f'{_cap:,}',
                        f'{_caprt:.0f}% of all {_tot_churn:,} churners', accent='#059669')
        with sc3:
            metric_card('Campaign Precision', f'{_cprec:.0f}%',
                        'Flagged customers who are genuine churners', accent='#2563eb')
        with sc4:
            metric_card('Lift vs. Random', f'{_lift:.1f}×',
                        'Efficiency gain over untargeted outreach', accent='#7c3aed')
    else:
        st.info('Predictions not yet available. Run the pipeline first.')

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Recommended Retention Actions ────────────────────────
    st.markdown('<div class="section-title">Recommended Retention Actions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'Suggested interventions mapped to the primary churn signals identified by the model. '
        'Match each action to the dominant risk driver visible in a customer\'s profile.'
        '</div>',
        unsafe_allow_html=True,
    )

    actions_df = pd.DataFrame([
        ('Equipment age > 18 months',    'eqpdays > 500 days',                     'Subsidised device upgrade offer',                      'High'),
        ('Declining call volume',         'recent_usage_delta < 0 for 2+ months',   'Proactive outreach — usage re-engagement programme',   'High'),
        ('High call drop rate',           'drop_rate above segment average',         'Service quality intervention + goodwill credit',       'High'),
        ('Top 10% risk score',            'Churn probability above threshold',       'Priority call centre contact — senior agent handling', 'High'),
        ('Revenue instability',           'rev_Mean declining over 3 months',        'Loyalty pricing or plan renegotiation offer',          'Medium'),
        ('Short tenure (< 12 months)',    'months < 12',                             'Early lifecycle retention programme',                  'Medium'),
        ('Low product engagement',        'avgmou below segment median',             'Bundle value-added services to increase stickiness',   'Low'),
    ], columns=['Risk Pattern', 'Trigger Signal', 'Suggested Action', 'Priority'])

    st.dataframe(
        actions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Risk Pattern':    st.column_config.TextColumn('Risk Pattern',    width='medium'),
            'Trigger Signal':  st.column_config.TextColumn('Trigger Signal',  width='medium'),
            'Suggested Action':st.column_config.TextColumn('Suggested Action',width='large'),
            'Priority':        st.column_config.TextColumn('Priority',        width='small'),
        },
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Gain Curve ───────────────────────────────────────────
    st.markdown('<div class="section-title">Cumulative Gain — Churn Capture Efficiency</div>',
                unsafe_allow_html=True)

    _gc_data = (seg.get('gain_curve_data') or
                seg.get('gain_curve') or
                seg.get('cumulative_data'))

    if _gc_data and isinstance(_gc_data, list) and len(_gc_data) > 1:
        try:
            gdf   = pd.DataFrame(_gc_data)
            x_col = next((c for c in gdf.columns
                          if any(k in c.lower() for k in ('pct','percent','customer'))), gdf.columns[0])
            y_col = next((c for c in gdf.columns
                          if any(k in c.lower() for k in ('capture','churn'))), gdf.columns[1])
            fig_gain = go.Figure()
            fig_gain.add_trace(go.Scatter(
                x=gdf[x_col], y=gdf[y_col], mode='lines', name='Model',
                line=dict(color='#2563eb', width=2.5),
                fill='tozeroy', fillcolor='rgba(37,99,235,0.06)',
            ))
            fig_gain.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100], mode='lines', name='Random baseline',
                line=dict(color='#d1d5db', width=1.5, dash='dash'),
            ))
            fig_gain.update_layout(
                plot_bgcolor='white', paper_bgcolor='white', font_color='#111827',
                xaxis_title='% Customers Contacted (highest risk first)',
                yaxis_title='% Churners Captured',
                height=370,
                margin=dict(l=60, r=30, t=20, b=50),
                legend=dict(yanchor='top', y=0.95, xanchor='left', x=0.05,
                            bgcolor='rgba(255,255,255,0.85)',
                            bordercolor='#e5e7eb', borderwidth=1),
            )
            fig_gain.update_xaxes(showgrid=True, gridcolor='#f3f4f6', zeroline=False)
            fig_gain.update_yaxes(showgrid=True, gridcolor='#f3f4f6', zeroline=False)
            st.plotly_chart(fig_gain, use_container_width=True)
        except Exception:
            _gp = os.path.join(OUTPUT_DIR, 'gain_curve.png')
            if os.path.exists(_gp):
                _, gc, _ = st.columns([1, 5, 1])
                with gc:
                    st.image(_gp, use_container_width=True)
    else:
        _gp = os.path.join(OUTPUT_DIR, 'gain_curve.png')
        if os.path.exists(_gp):
            _, gc, _ = st.columns([1, 5, 1])
            with gc:
                st.image(_gp, use_container_width=True)


# ============================================================
# TAB 2 — Customer Segmentation
# ============================================================
with tab2:

    # ── Risk Tier Summary ────────────────────────────────────
    st.markdown('<div class="section-title">Risk Tier Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'Customers are ranked by predicted churn probability and grouped into tiers. '
        'Higher tiers contain a disproportionate share of genuine churners — concentrating '
        'retention spend on the top tier maximises campaign ROI while minimising false-positive contact costs.'
        '</div>',
        unsafe_allow_html=True,
    )

    if 'segmentation' in data:
        rt1, rt2, rt3 = st.columns(3)
        for col, pct, accent in zip(
            [rt1, rt2, rt3], [10, 20, 30],
            ['#dc2626', '#d97706', '#2563eb'],
        ):
            s = seg.get(f'top_{pct}_percent', {})
            with col:
                metric_card(
                    f'Top {pct}% — Highest Risk',
                    f"{s.get('churn_captured_pct', '—')}% churners",
                    f"Segment churn rate: {s.get('churn_rate_pct', '—')}%",
                    accent=accent,
                )
    else:
        st.info('Segmentation data not available. Run the pipeline first.')

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Risk Distribution ────────────────────────────────────
    if 'predictions' in data:
        st.markdown('<div class="section-title">Predicted Risk Distribution</div>',
                    unsafe_allow_html=True)
        _dist = data['predictions']
        fig_hist = px.histogram(
            _dist, x='churn_probability', color='true_label',
            nbins=50, barmode='overlay', opacity=0.72,
            labels={'churn_probability': 'Predicted Churn Probability', 'true_label': 'Actual'},
            color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
        )
        fig_hist.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font_color='#111827',
            height=310, margin=dict(l=50, r=30, t=20, b=50),
            legend_title='Actual label',
            legend=dict(bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='#e5e7eb', borderwidth=1),
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor='#f3f4f6')
        fig_hist.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown(
            '<div class="callout">'
            'Separation between the blue (retained) and red (churned) distributions reflects '
            'model discrimination quality. Overlap in the 0.30–0.60 range is where threshold '
            'selection has the greatest operational impact on campaign cost and coverage.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Regional View ────────────────────────────────────────
    st.markdown('<div class="section-title">Churn Rate by Region</div>', unsafe_allow_html=True)
    try:
        _mp = os.path.join(BASE_DIR, 'models', 'dataset_lgbm.pkl')
        _a2s = {
            'NORTHWEST/ROCKY MOUNTAIN AREA': 'MT', 'MIDWEST AREA': 'IL',
            'NEW ENGLAND AREA': 'MA',  'GREAT LAKES AREA': 'MI', 'OHIO AREA': 'OH',
            'TENNESSEE AREA': 'TN',    'HOUSTON AREA': 'TX',     'DALLAS AREA': 'TX',
            'CENTRAL/SOUTH TEXAS AREA': 'TX', 'SOUTHWEST AREA': 'AZ',
            'CALIFORNIA NORTH AREA': 'CA', 'LOS ANGELES AREA': 'CA',
            'SOUTH FLORIDA AREA': 'FL', 'NORTH FLORIDA AREA': 'FL',
            'ATLANTIC SOUTH AREA': 'GA', 'PHILADELPHIA AREA': 'PA',
            'NEW YORK CITY AREA': 'NY', 'CHICAGO AREA': 'IL',
            'DC/MARYLAND/VIRGINIA AREA': 'VA',
        }
        if os.path.exists(_mp):
            with open(_mp, 'rb') as f:
                _lgbm = pickle.load(f)
            _X, _y = _lgbm['X'], _lgbm['y']
            if 'area' in _X.columns:
                _dm = pd.DataFrame({
                    'area':  _X['area'].astype(str).str.strip().str.upper(),
                    'churn': _y.values,
                })
                _ac = (_dm.groupby('area')['churn']
                         .agg(['mean', 'count'])
                         .reset_index()
                         .rename(columns={'mean': 'churn_rate', 'count': 'n'}))
                _ac['state']          = _ac['area'].map(_a2s)
                _ac                   = _ac.dropna(subset=['state'])
                _ac['churn_rate_pct'] = (_ac['churn_rate'] * 100).round(1)
                fig_map = px.choropleth(
                    _ac, locations='state', locationmode='USA-states',
                    color='churn_rate_pct', scope='usa',
                    color_continuous_scale=[[0, '#dbeafe'], [0.5, '#3b82f6'], [1, '#1e3a8a']],
                    labels={'churn_rate_pct': 'Churn Rate (%)'},
                    hover_data={'area': True, 'n': True, 'churn_rate_pct': True},
                )
                fig_map.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white', font_color='#111827',
                    height=370, margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_colorbar=dict(title='Churn %', thickness=12),
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption('Areas with multiple sub-regions mapped to the same state reflect combined averages.')
            else:
                st.info('Area column not found in dataset.')
        else:
            st.info('Run the pipeline to enable regional analysis.')
    except Exception as e:
        st.info(f'Regional map not available: {e}')

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── High-Risk Prioritisation List ────────────────────────
    if 'drivers' not in data:
        st.info('High-risk customer list not yet available. Run the pipeline first.')
    else:
        st.markdown('<div class="section-title">High-Risk Customer Prioritisation List</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="callout">'
            'Top 100 accounts ranked by predicted churn probability. Each row shows the three '
            'most influential model signals driving that customer\'s risk score. Use this list '
            'to brief call centre teams and account managers on priority outreach targets.'
            '</div>',
            unsafe_allow_html=True,
        )

        fc1, fc2 = st.columns([2, 2])
        with fc1:
            min_prob_pct = st.slider('Minimum risk score', 0, 100, 0, 5, format='%d%%', help='Show only customers at or above this predicted churn probability')
            min_prob = min_prob_pct / 100.0
        with fc2:
            driver_search = st.text_input(
                'Filter by risk driver', '',
                placeholder='e.g. eqpdays, drop_rate, months…',
            )

        _rows = []
        for _d in data['drivers']:
            _drv = _d.get('top_drivers', [])
            _d1 = f"{_drv[0]['feature']} ({_drv[0]['shap_value']:+.3f})" if len(_drv) > 0 else ''
            _d2 = f"{_drv[1]['feature']} ({_drv[1]['shap_value']:+.3f})" if len(_drv) > 1 else ''
            _d3 = f"{_drv[2]['feature']} ({_drv[2]['shap_value']:+.3f})" if len(_drv) > 2 else ''
            _rows.append({
                'Rank':           _d['rank'],
                'Risk Score':     _d['churn_probability'],
                'Risk %':         round(_d['churn_probability'] * 100, 1),
                'Primary Driver': _d1,
                'Driver 2':       _d2,
                'Driver 3':       _d3,
                '_s':             f"{_d1} {_d2} {_d3}".lower(),
            })

        _ddf = pd.DataFrame(_rows)
        if min_prob > 0:
            _ddf = _ddf[_ddf['Risk Score'] >= min_prob]
        if driver_search.strip():
            _ddf = _ddf[_ddf['_s'].str.contains(driver_search.strip().lower(), na=False)]

        st.caption(
            f"{len(_ddf)} account(s) · "
            "Positive SHAP values (+) increase churn risk · Negative values (−) decrease it"
        )

        if len(_ddf) == 0:
            st.info('No accounts match the current filters.')
        else:
            st.dataframe(
                _ddf.drop(columns=['_s']),
                column_config={
                    'Rank':          st.column_config.NumberColumn('Rank',           width='small'),
                    'Risk Score':    st.column_config.ProgressColumn('Risk Score',
                                                                     min_value=0, max_value=1,
                                                                     width='medium'),
                    'Risk %':        st.column_config.NumberColumn('Risk %',         format='%.1f%%',
                                                                   width='small'),
                    'Primary Driver':st.column_config.TextColumn('Primary Driver',   width='large'),
                    'Driver 2':      st.column_config.TextColumn('Driver 2',         width='large'),
                    'Driver 3':      st.column_config.TextColumn('Driver 3',         width='large'),
                },
                use_container_width=True,
                hide_index=True,
                height=420,
            )


# ============================================================
# TAB 3 — Explainability & Drivers
# ============================================================
with tab3:

    # ── Aggregated SHAP ──────────────────────────────────────
    st.markdown('<div class="section-title">What Is Driving Churn?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'Feature importance is aggregated across the top 100 highest-risk accounts using SHAP '
        '(SHapley Additive exPlanations) — a game-theoretic method that attributes each model '
        'prediction fairly across all input signals. Features with higher importance consistently '
        'appear as the decisive factor for customers most likely to leave.'
        '</div>',
        unsafe_allow_html=True,
    )

    if feature_counts:
        _df_feat = pd.DataFrame([
            {'Feature': k, 'Importance': v}
            for k, v in top_features[:15]
        ])
        fig_feat = px.bar(
            _df_feat, x='Importance', y='Feature', orientation='h',
            color='Importance',
            color_continuous_scale=[[0, '#dbeafe'], [1, '#1d4ed8']],
        )
        fig_feat.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font_color='#111827',
            showlegend=False, coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Aggregated SHAP importance (top 100 high-risk accounts)',
            yaxis_title='',
            height=440,
            margin=dict(l=170, r=30, t=20, b=50),
        )
        fig_feat.update_xaxes(showgrid=True, gridcolor='#f3f4f6')
        fig_feat.update_yaxes(showgrid=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Key Risk Patterns ────────────────────────────────────
    st.markdown('<div class="section-title">Key Risk Patterns — Operational Implications</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'The dominant model signals are translated into plain-language business insights below. '
        'Each pattern is accompanied by a recommended retention intervention.'
        '</div>',
        unsafe_allow_html=True,
    )

    _matched = [
        (feat, FEATURE_MAP[feat])
        for feat, _ in top_features
        if feat in FEATURE_MAP
    ][:4]

    if not _matched:
        _matched = [
            ('eqpdays',            FEATURE_MAP['eqpdays']),
            ('recent_usage_delta', FEATURE_MAP['recent_usage_delta']),
            ('drop_rate',          FEATURE_MAP['drop_rate']),
            ('months',             FEATURE_MAP['months']),
        ]

    _pc = st.columns(len(_matched))
    for i, (feat, info) in enumerate(_matched):
        with _pc[i]:
            insight_card(
                title=info[0],
                description=info[2],
                action=info[3],
                accent=info[1],
            )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SHAP Summary Image ───────────────────────────────────
    _ss = os.path.join(OUTPUT_DIR, 'shap_summary.png')
    if os.path.exists(_ss):
        st.markdown('<div class="section-title">SHAP Feature Impact — Full Model View</div>',
                    unsafe_allow_html=True)
        st.caption(
            'Each dot represents one customer. Colour indicates feature value '
            '(red = high, blue = low). Position on the x-axis shows contribution direction.'
        )
        _, _sc, _ = st.columns([1, 6, 1])
        with _sc:
            st.image(_ss, use_container_width=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Individual Customer Waterfall ────────────────────────
    _wf = os.path.join(OUTPUT_DIR, 'shap_waterfall.png')
    if os.path.exists(_wf):
        st.markdown('<div class="section-title">Individual Customer Risk Profile</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="callout">'
            'This chart deconstructs why the highest-risk customer was scored as a likely churner. '
            'Red bars push the prediction toward churn; blue bars push it toward retention. '
            'The base value E[f(x)] is the model\'s average prediction across all customers — '
            'each feature\'s bar shows how far that customer deviates from the base.'
            '</div>',
            unsafe_allow_html=True,
        )
        _, _wc, _ = st.columns([1, 5, 1])
        with _wc:
            st.image(_wf, use_container_width=True)


# ============================================================
# TAB 4 — Technical Appendix
# ============================================================
with tab4:

    st.markdown(
        '<div class="callout" style="margin-top:0.5rem;">'
        '<strong>Technical Appendix</strong> — Model validation plots and exploratory analysis '
        'for data scientists and technical reviewers. Business users should refer to the '
        'Executive Overview and Customer Segmentation tabs for operational insights.'
        '</div>',
        unsafe_allow_html=True,
    )

    _sel = data.get('selection', {})
    _winner = _sel.get('selected_model', model_name)
    _reason = _sel.get('selection_reason', '')
    if _reason:
        st.markdown(
            f'<div class="callout"><strong>Selected model:</strong> {_winner} — {_reason}</div>',
            unsafe_allow_html=True,
        )

    # ── Discrimination & Calibration ─────────────────────────
    st.markdown('<div class="section-title">Discrimination & Calibration</div>',
                unsafe_allow_html=True)
    _vl, _vr = st.columns(2)

    with _vl:
        _roc = os.path.join(OUTPUT_DIR, 'roc_curve.png')
        if os.path.exists(_roc):
            st.markdown('<p class="tech-label">ROC Curve — Ranking Quality</p>',
                        unsafe_allow_html=True)
            st.image(_roc, use_container_width=True)
            st.caption('AUC-ROC measures ranking quality across all thresholds. '
                       'Scores above 0.75 indicate useful discrimination; 0.5 = random.')

        _thr = os.path.join(OUTPUT_DIR, 'precision_recall_vs_threshold.png')
        if os.path.exists(_thr):
            st.markdown('<p class="tech-label">Threshold Optimisation</p>',
                        unsafe_allow_html=True)
            st.image(_thr, use_container_width=True)
            st.caption('The business threshold is selected where precision ≥ 60% '
                       'at maximum recall, minimising false-positive campaign contacts.')

    with _vr:
        _pr = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
        if os.path.exists(_pr):
            st.markdown('<p class="tech-label">Precision–Recall Curve</p>',
                        unsafe_allow_html=True)
            st.image(_pr, use_container_width=True)
            st.caption('More informative than ROC for imbalanced classes (~14% churn rate). '
                       'Average Precision summarises performance across all operating points.')

        _cal = os.path.join(OUTPUT_DIR, 'calibration_curve.png')
        if os.path.exists(_cal):
            st.markdown('<p class="tech-label">Probability Calibration</p>',
                        unsafe_allow_html=True)
            st.image(_cal, use_container_width=True)
            st.caption('A calibrated model means predicted probabilities reflect true empirical '
                       'frequencies — essential for reliable threshold selection and expected '
                       'value calculations.')

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Confusion Matrices ───────────────────────────────────
    st.markdown('<div class="section-title">Decision Boundaries — Confusion Matrices</div>',
                unsafe_allow_html=True)
    _cm1, _cm2 = st.columns(2)
    with _cm1:
        _cm = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        if os.path.exists(_cm):
            st.markdown('<p class="tech-label">Default Threshold (0.5)</p>',
                        unsafe_allow_html=True)
            st.image(_cm, use_container_width=True)
    with _cm2:
        _cmb = os.path.join(OUTPUT_DIR, 'confusion_matrix_business_threshold.png')
        if os.path.exists(_cmb):
            _cm_label = f'Business Threshold ({biz_thr:.2f})' if biz_thr else 'Business Threshold'
            st.markdown(f'<p class="tech-label">{_cm_label}</p>', unsafe_allow_html=True)
            st.image(_cmb, use_container_width=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SHAP Bar (full model) ────────────────────────────────
    _sb = os.path.join(OUTPUT_DIR, 'shap_bar.png')
    if os.path.exists(_sb):
        st.markdown('<div class="section-title">SHAP — Mean Absolute Feature Importance (Full Model)</div>',
                    unsafe_allow_html=True)
        _, _sbc, _ = st.columns([1, 5, 1])
        with _sbc:
            st.image(_sb, use_container_width=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── Exploratory Data Analysis ────────────────────────────
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        'Raw dataset analysis conducted prior to model training. These plots informed '
        'feature selection decisions and confirmed the domain-motivated feature engineering choices '
        '(equipment age threshold, usage delta, call drop rate).'
        '</div>',
        unsafe_allow_html=True,
    )

    _eda_plots = [
        ('01_target_distribution.png',  'Target Distribution',           'Class balance of the churn label.'),
        ('02_missing_values.png',        'Missing Values',                'Columns > 50% missing were dropped; 20–50% received was_missing flags.'),
        ('03_numeric_distributions.png', 'Numeric Distributions',        'Top 12 numeric features by correlation with churn, split by class.'),
        ('04_outlier_detection.png',     'Outlier Detection',             'Z-score method — features with > 1% of values beyond |z| = 3.'),
        ('05_churn_by_categorical.png',  'Churn by Categorical Feature',  'Churn rate per category vs. global average.'),
        ('06_correlation_heatmap.png',   'Correlation Heatmap',           'Feature intercorrelation for the top 20 features by churn correlation.'),
        ('07_top_features_boxplot.png',  'Top Features by Churn',         'Boxplots with Mann-Whitney U significance test.'),
    ]

    for i in range(0, len(_eda_plots), 2):
        _left  = _eda_plots[i]
        _right = _eda_plots[i + 1] if i + 1 < len(_eda_plots) else None
        _ca, _cb = st.columns(2)
        with _ca:
            _p = os.path.join(EDA_DIR, _left[0])
            if os.path.exists(_p):
                st.markdown(f'<p class="tech-label">{_left[1]}</p>', unsafe_allow_html=True)
                st.caption(_left[2])
                st.image(_p, use_container_width=True)
        with _cb:
            if _right:
                _p = os.path.join(EDA_DIR, _right[0])
                if os.path.exists(_p):
                    st.markdown(f'<p class="tech-label">{_right[1]}</p>', unsafe_allow_html=True)
                    st.caption(_right[2])
                    st.image(_p, use_container_width=True)
        st.markdown('')

    # EDA summary JSON
    _eda_json = os.path.join(EDA_DIR, 'eda_summary.json')
    if os.path.exists(_eda_json):
        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
        with open(_eda_json) as f:
            _es = json.load(f)

        _s1, _s2, _s3, _s4 = st.columns(4)
        _s1.metric('Total Rows',     f"{_es.get('n_rows', 0):,}")
        _s2.metric('Total Columns',  f"{_es.get('n_cols', 0):,}")
        _s3.metric('Churn Rate',     f"{_es.get('churn_rate', 0):.1%}")
        _s4.metric('Duplicate Rows', f"{_es.get('duplicate_row_count', 0):,}")

        _tc = _es.get('top_10_corr_with_churn', {})
        if _tc:
            st.markdown('<div class="section-title">Top Correlations with Churn</div>',
                        unsafe_allow_html=True)
            _cdf = pd.DataFrame([
                {'Feature': k, 'Pearson |r|': round(abs(v), 4)}
                for k, v in list(_tc.items())[:10]
            ]).sort_values('Pearson |r|', ascending=False)
            st.dataframe(_cdf, use_container_width=False, hide_index=True, width=420)

        _cc = _es.get('categorical_cardinality', {})
        if _cc:
            st.markdown('<div class="section-title">Categorical Cardinality</div>',
                        unsafe_allow_html=True)
            _ccdf = pd.DataFrame([
                {'Feature': k, 'Unique Values': v}
                for k, v in sorted(_cc.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(_ccdf, use_container_width=False, hide_index=True, width=380)
