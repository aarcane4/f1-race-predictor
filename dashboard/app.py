import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.predict import get_model_comparison, load_model, load_scaler
from features.feature_engineering import engineer_features, get_feature_columns

st.set_page_config(page_title="F1 RACE PREDICTOR", page_icon="🏎️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Barlow+Condensed:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Barlow:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: #080808 !important;
    color: #f0f0f0;
    font-family: 'Barlow', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(225,6,0,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(225,6,0,.04) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridpulse 8s ease-in-out infinite alternate;
}
@keyframes gridpulse {
    0%   { opacity: .4; }
    100% { opacity: 1; }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1e1e1e !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem !important; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 900;
    color: #E10600; letter-spacing: .15em;
    text-transform: uppercase;
    border-bottom: 2px solid #E10600;
    padding-bottom: .75rem; margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 10px;
}
.sidebar-logo span { color: #fff; }

/* ── Sidebar inputs ── */
.stSelectbox label, .stRadio label, .stSlider label,
.stNumberInput label { 
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: .75rem !important; font-weight: 600 !important;
    color: #666 !important; letter-spacing: .1em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
.stRadio > div { gap: 6px !important; }
.stRadio > div > label {
    background: #111 !important; border: 1px solid #222 !important;
    border-radius: 4px !important; padding: 6px 12px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: .8rem !important; color: #aaa !important;
    transition: all .2s !important;
}
.stRadio > div > label:has(input:checked) {
    background: #E10600 !important; border-color: #E10600 !important;
    color: #fff !important;
}
.stSlider > div > div > div { background: #E10600 !important; }
div[data-testid="stNumberInput"] input {
    background: #111 !important; border: 1px solid #222 !important;
    color: #f0f0f0 !important; border-radius: 4px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}

/* ── Hero header ── */
.hero-wrap {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #0d0000 0%, #130000 50%, #0d0d0d 100%);
    border: 1px solid #1e1e1e; border-left: 4px solid #E10600;
    border-radius: 8px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.hero-wrap::after {
    content: '';
    position: absolute; right: -30px; top: 50%; transform: translateY(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(225,6,0,.15) 0%, transparent 70%);
    pointer-events: none;
}
.race-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .7rem; font-weight: 700; letter-spacing: .2em;
    color: #E10600; text-transform: uppercase; margin-bottom: 6px;
}
.race-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem; font-weight: 800;
    color: #fff; line-height: 1.1; margin-bottom: 8px;
    text-shadow: 0 0 40px rgba(225,6,0,.3);
}
.race-meta {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .85rem; font-weight: 400; color: #666;
    letter-spacing: .05em;
}
.race-meta strong { color: #aaa; }
.upcoming-pill {
    display: inline-block;
    background: #E10600; color: #fff;
    font-family: 'Orbitron', monospace;
    font-size: .6rem; font-weight: 700; letter-spacing: .1em;
    padding: 3px 10px; border-radius: 2px;
    text-transform: uppercase; margin-left: 10px;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50% { opacity: .5; }
}

/* ── Metric cards ── */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.metric-card {
    background: #0d0d0d; border: 1px solid #1a1a1a;
    border-top: 2px solid #E10600;
    border-radius: 4px; padding: 1rem 1.25rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(225,6,0,.08), transparent);
}
.svg-wm { position:absolute;inset:0;width:100%;height:100%;opacity:.055;pointer-events:none; }
.wi { position:relative;z-index:1; }
.hero-svg { position:absolute;right:0;top:0;bottom:0;width:340px;opacity:.065;pointer-events:none; }
.metric-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem; font-weight: 700;
    color: #E10600; line-height: 1;
}
.metric-lbl {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .65rem; font-weight: 600;
    color: #444; text-transform: uppercase;
    letter-spacing: .12em; margin-top: 6px;
}
.metric-sub {
    font-family: 'Barlow', sans-serif;
    font-size: .72rem; color: #333; margin-top: 2px;
}

/* ── Section titles ── */
.sec-title {
    font-family: 'Orbitron', monospace;
    font-size: .6rem; font-weight: 700;
    color: #E10600; text-transform: uppercase;
    letter-spacing: .2em; margin-bottom: 1rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a1a1a;
    display: flex; align-items: center; gap: 8px;
}
.sec-title::before {
    content: ''; display: inline-block;
    width: 16px; height: 2px; background: #E10600;
}

/* ── Podium cards ── */
.podium-card {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 4px; padding: 12px 16px;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 12px;
    transition: border-color .2s, background .2s;
    position: relative; overflow: hidden;
}
.podium-card.p1 {
    background: linear-gradient(90deg, #130e00, #0d0d0d);
    border-color: #c8891c;
}
.podium-card.p2 { border-color: #888; }
.podium-card.p3 { border-color: #a0522d; }
.podium-pos {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem; font-weight: 900;
    min-width: 36px; text-align: center;
}
.p1 .podium-pos { color: #FFD700; }
.p2 .podium-pos { color: #C0C0C0; }
.p3 .podium-pos { color: #CD7F32; }
.podium-info { flex: 1; }
.podium-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #fff; letter-spacing: .02em;
}
.podium-team {
    font-family: 'Barlow', sans-serif;
    font-size: .72rem; color: #555; margin-top: 2px;
}
.podium-prob {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem; font-weight: 700;
    color: #E10600; text-align: right;
}
.podium-prob-lbl {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .6rem; color: #444;
    text-transform: uppercase; letter-spacing: .1em;
    text-align: right; margin-top: 2px;
}

/* ── Team color stripe ── */
.team-stripe {
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; border-radius: 4px 0 0 4px;
}

/* ── DNF rows ── */
.dnf-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #111;
    font-family: 'Barlow Condensed', sans-serif;
}
.dnf-row:last-child { border-bottom: none; }
.dnf-code { font-size: .95rem; font-weight: 700; color: #ccc; min-width: 40px; }
.dnf-team { font-size: .78rem; color: #444; flex: 1; }
.dnf-badge {
    font-family: 'Orbitron', monospace;
    font-size: .6rem; font-weight: 700;
    padding: 3px 10px; border-radius: 2px;
    text-transform: uppercase; letter-spacing: .05em;
}
.risk-hi  { background: rgba(225,6,0,.15);   color: #E10600; border: 1px solid rgba(225,6,0,.3); }
.risk-med { background: rgba(239,159,39,.1); color: #EF9F27; border: 1px solid rgba(239,159,39,.3); }
.risk-lo  { background: rgba(59,109,17,.1);  color: #6dbf3e; border: 1px solid rgba(59,109,17,.3); }

/* ── Warning banner ── */
.warn-banner {
    background: rgba(239,159,39,.05);
    border: 1px solid rgba(239,159,39,.2);
    border-left: 3px solid #EF9F27;
    border-radius: 4px; padding: 10px 16px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .85rem; color: #EF9F27;
    margin-bottom: 1.25rem; letter-spacing: .02em;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Orbitron', monospace !important;
    font-size: .65rem !important; color: #666 !important;
    letter-spacing: .15em !important;
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 4px !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1a1a1a !important; border-radius: 4px !important; }
.stDataFrame thead th {
    background: #0d0d0d !important; color: #E10600 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: .55rem !important; letter-spacing: .1em !important;
}

/* ── Caption ── */
.f1-caption {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .7rem; color: #2a2a2a;
    text-align: center; letter-spacing: .15em;
    text-transform: uppercase; padding-top: 1rem;
    border-top: 1px solid #111; margin-top: 1rem;
}

/* ── Sidebar section headers ── */
.sb-section {
    font-family: 'Orbitron', monospace;
    font-size: .55rem; font-weight: 700;
    color: #E10600; letter-spacing: .2em;
    text-transform: uppercase;
    margin: 1rem 0 .5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    'mercedes':'#27F4D2','ferrari':'#E8002D','mclaren':'#FF8000',
    'red bull':'#3671C6','aston martin':'#229971','alpine':'#FF87BC',
    'williams':'#64C4FF','racing bulls':'#5E8FAA','haas':'#B6BABD',
    'audi':'#E4002B','cadillac':'#0066CC',
}
def tc(team):
    t = str(team).lower()
    for k,v in TEAM_COLORS.items():
        if k in t: return v
    return '#555'

SCHEDULE_2026 = [
    {'Round':1, 'RaceName':'Australian Grand Prix',     'Circuit':'Albert Park',       'Date':'15 MAR 2026','Status':'past'},
    {'Round':2, 'RaceName':'Chinese Grand Prix',        'Circuit':'Shanghai',           'Date':'22 MAR 2026','Status':'past'},
    {'Round':3, 'RaceName':'Japanese Grand Prix',       'Circuit':'Suzuka',             'Date':'29 MAR 2026','Status':'upcoming'},
    {'Round':4, 'RaceName':'Bahrain Grand Prix',        'Circuit':'Sakhir',             'Date':'19 APR 2026','Status':'upcoming'},
    {'Round':5, 'RaceName':'Saudi Arabian Grand Prix',  'Circuit':'Jeddah',             'Date':'26 APR 2026','Status':'upcoming'},
    {'Round':6, 'RaceName':'Miami Grand Prix',          'Circuit':'Miami Int\'l Autodrome','Date':'10 MAY 2026','Status':'upcoming'},
    {'Round':7, 'RaceName':'Emilia Romagna Grand Prix', 'Circuit':'Imola',              'Date':'24 MAY 2026','Status':'upcoming'},
    {'Round':8, 'RaceName':'Monaco Grand Prix',         'Circuit':'Circuit de Monaco',  'Date':'31 MAY 2026','Status':'upcoming'},
    {'Round':9, 'RaceName':'Spanish Grand Prix',        'Circuit':'Barcelona',          'Date':'14 JUN 2026','Status':'upcoming'},
    {'Round':10,'RaceName':'Canadian Grand Prix',       'Circuit':'Montréal',           'Date':'21 JUN 2026','Status':'upcoming'},
    {'Round':11,'RaceName':'Austrian Grand Prix',       'Circuit':'Red Bull Ring',      'Date':'05 JUL 2026','Status':'upcoming'},
    {'Round':12,'RaceName':'British Grand Prix',        'Circuit':'Silverstone',        'Date':'19 JUL 2026','Status':'upcoming'},
    {'Round':13,'RaceName':'Belgian Grand Prix',        'Circuit':'Spa-Francorchamps',  'Date':'02 AUG 2026','Status':'upcoming'},
    {'Round':14,'RaceName':'Hungarian Grand Prix',      'Circuit':'Hungaroring',        'Date':'23 AUG 2026','Status':'upcoming'},
    {'Round':15,'RaceName':'Dutch Grand Prix',          'Circuit':'Zandvoort',          'Date':'30 AUG 2026','Status':'upcoming'},
    {'Round':16,'RaceName':'Italian Grand Prix',        'Circuit':'Monza',              'Date':'06 SEP 2026','Status':'upcoming'},
    {'Round':17,'RaceName':'Azerbaijan Grand Prix',     'Circuit':'Baku',               'Date':'20 SEP 2026','Status':'upcoming'},
    {'Round':18,'RaceName':'Singapore Grand Prix',      'Circuit':'Marina Bay',         'Date':'04 OCT 2026','Status':'upcoming'},
    {'Round':19,'RaceName':'United States Grand Prix',  'Circuit':'COTA',               'Date':'18 OCT 2026','Status':'upcoming'},
    {'Round':20,'RaceName':'Mexico City Grand Prix',    'Circuit':'Autodromo Hermanos Rodriguez','Date':'25 OCT 2026','Status':'upcoming'},
    {'Round':21,'RaceName':'São Paulo Grand Prix',      'Circuit':'Interlagos',         'Date':'08 NOV 2026','Status':'upcoming'},
    {'Round':22,'RaceName':'Las Vegas Grand Prix',      'Circuit':'Las Vegas Strip',    'Date':'22 NOV 2026','Status':'upcoming'},
    {'Round':23,'RaceName':'Qatar Grand Prix',          'Circuit':'Lusail',             'Date':'29 NOV 2026','Status':'upcoming'},
    {'Round':24,'RaceName':'Abu Dhabi Grand Prix',      'Circuit':'Yas Marina',         'Date':'06 DEC 2026','Status':'upcoming'},
]

DRIVERS_2026 = [
    {'Abbreviation':'RUS','FullName':'George Russell',    'TeamName':'Mercedes',        'DriverNumber':'63'},
    {'Abbreviation':'ANT','FullName':'Kimi Antonelli',    'TeamName':'Mercedes',        'DriverNumber':'12'},
    {'Abbreviation':'LEC','FullName':'Charles Leclerc',   'TeamName':'Ferrari',         'DriverNumber':'16'},
    {'Abbreviation':'HAM','FullName':'Lewis Hamilton',    'TeamName':'Ferrari',         'DriverNumber':'44'},
    {'Abbreviation':'NOR','FullName':'Lando Norris',      'TeamName':'McLaren',         'DriverNumber':'4'},
    {'Abbreviation':'PIA','FullName':'Oscar Piastri',     'TeamName':'McLaren',         'DriverNumber':'81'},
    {'Abbreviation':'VER','FullName':'Max Verstappen',    'TeamName':'Red Bull Racing', 'DriverNumber':'1'},
    {'Abbreviation':'LAW','FullName':'Liam Lawson',       'TeamName':'Red Bull Racing', 'DriverNumber':'30'},
    {'Abbreviation':'ALO','FullName':'Fernando Alonso',   'TeamName':'Aston Martin',    'DriverNumber':'14'},
    {'Abbreviation':'STR','FullName':'Lance Stroll',      'TeamName':'Aston Martin',    'DriverNumber':'18'},
    {'Abbreviation':'GAS','FullName':'Pierre Gasly',      'TeamName':'Alpine',          'DriverNumber':'10'},
    {'Abbreviation':'DOO','FullName':'Jack Doohan',       'TeamName':'Alpine',          'DriverNumber':'7'},
    {'Abbreviation':'ALB','FullName':'Alexander Albon',   'TeamName':'Williams',        'DriverNumber':'23'},
    {'Abbreviation':'SAI','FullName':'Carlos Sainz',      'TeamName':'Williams',        'DriverNumber':'55'},
    {'Abbreviation':'HUL','FullName':'Nico Hulkenberg',   'TeamName':'Audi',            'DriverNumber':'27'},
    {'Abbreviation':'BOR','FullName':'Gabriel Bortoleto', 'TeamName':'Audi',            'DriverNumber':'5'},
    {'Abbreviation':'TSU','FullName':'Yuki Tsunoda',      'TeamName':'Racing Bulls',    'DriverNumber':'22'},
    {'Abbreviation':'HAD','FullName':'Isack Hadjar',      'TeamName':'Racing Bulls',    'DriverNumber':'6'},
    {'Abbreviation':'OCO','FullName':'Esteban Ocon',      'TeamName':'Haas F1 Team',    'DriverNumber':'31'},
    {'Abbreviation':'BEA','FullName':'Oliver Bearman',    'TeamName':'Haas F1 Team',    'DriverNumber':'87'},
    {'Abbreviation':'PER','FullName':'Sergio Perez',      'TeamName':'Cadillac',        'DriverNumber':'11'},
    {'Abbreviation':'BOT','FullName':'Valtteri Bottas',   'TeamName':'Cadillac',        'DriverNumber':'77'},
]

DRIVER_FORM = {
    'VER':1.8,'LEC':3.2,'NOR':3.5,'HAM':3.8,'RUS':4.1,'PIA':4.3,
    'SAI':4.8,'ALO':5.2,'ANT':5.5,'GAS':7.1,'TSU':7.8,'STR':8.2,
    'LAW':8.5,'OCO':9.1,'ALB':9.4,'HUL':9.8,'DOO':11.2,'BOR':11.8,
    'HAD':12.1,'BEA':13.0,'PER':7.5,'BOT':10.2,
}
TEAM_FORM = {
    'Red Bull Racing':1.9,'Ferrari':2.8,'McLaren':3.1,'Mercedes':3.8,
    'Aston Martin':6.2,'Alpine':7.8,'Williams':8.5,'Racing Bulls':8.9,
    'Haas F1 Team':11.2,'Audi':12.1,'Cadillac':13.5,
}

@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__),'..','data','f1_raw_data.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_comparison():
    return get_model_comparison()

def predict_upcoming(race_name, quali_positions, weather, hist_df, model_name):
    hist = hist_df.copy()
    driver_stats = hist.groupby('Abbreviation').agg(
        win_rate   =('Winner','mean'),
        podium_rate=('Podium','mean'),
        dnf_rate   =('DNF','mean'),
        avg_pos    =('Position','mean'),
    ).reset_index()

    rows = []
    for d in DRIVERS_2026:
        abbr = d['Abbreviation']
        qpos = quali_positions.get(abbr, 15)
        ds   = driver_stats[driver_stats['Abbreviation']==abbr]
        if not ds.empty:
            wr  = float(ds['win_rate'].iloc[0])
            pr  = float(ds['podium_rate'].iloc[0])
            dr  = float(ds['dnf_rate'].iloc[0])
            ap  = float(ds['avg_pos'].iloc[0])
        else:
            f   = DRIVER_FORM.get(abbr, 12.0)
            wr  = max(0, .05 - f*.004)
            pr  = max(0, .25 - f*.018)
            dr  = .12
            ap  = f

        pole = min(quali_positions.values())
        rows.append({
            'Abbreviation':abbr,'FullName':d['FullName'],'TeamName':d['TeamName'],
            'QualiPosition':qpos,'QualiGapToPole':qpos-pole,
            'win_rate':wr,'podium_rate':pr,'dnf_rate':dr,'avg_pos':ap,
            'team_form':TEAM_FORM.get(d['TeamName'],10.),
            'AvgTrackTemp':weather['AvgTrackTemp'],
            'AvgRainfall':weather['AvgRainfall'],
        })

    df = pd.DataFrame(rows)
    df['score'] = (
        df['QualiPosition']*3.0 + df['avg_pos']*1.5 +
        df['team_form']*1.0 + df['QualiGapToPole']*.8 -
        df['win_rate']*20.0 - df['podium_rate']*5.0
    )
    if weather['AvgRainfall'] > .5:
        for a, b in {'VER':2,'HAM':2,'ALO':1,'NOR':1,'RUS':1}.items():
            df.loc[df['Abbreviation']==a,'score'] -= b

    df['snorm']  = (df['score'] - df['score'].min()) / (df['score'].max()-df['score'].min()+1e-9)
    df['inv']    = 1.0 / (df['snorm'] + .08)
    df['Winner_Prob']  = (df['inv']/df['inv'].sum()*100).round(1)
    df['Podium_Prob']  = (df['Winner_Prob']*2.8).clip(upper=88).round(1)
    df['DNF_Prob']     = (df['dnf_rate']*100 + np.random.uniform(0,5,len(df))).clip(4,60).round(1)
    df = df.sort_values('Winner_Prob',ascending=False).reset_index(drop=True)
    df['PredictedPosition'] = df.index+1
    return df

def predict_historical(race_df, model_name):
    from models.predict import predict_race as ml_predict
    return ml_predict(race_df, model_name=model_name)

df_all        = load_data()
df_comparison = load_comparison()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏎️ F1 <span>PREDICTOR</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">ML ENGINE</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("", ['XGBoost','RandomForest','NeuralNet'],
                                label_visibility='collapsed')

    st.markdown('<div class="sb-section">SEASON MODE</div>', unsafe_allow_html=True)
    season_tab = st.radio("", ['🔴  2026 SEASON','📋  HISTORICAL'],
                          label_visibility='collapsed')
    is_2026 = '2026' in season_tab

    if is_2026:
        st.markdown('<div class="sb-section">RACE CALENDAR</div>', unsafe_allow_html=True)
        race_labels = [
            f"{'🔴' if r['Status']=='upcoming' else '✅'}  R{r['Round']:02d} · {r['RaceName'].replace(' Grand Prix','').upper()}"
            for r in SCHEDULE_2026
        ]
        default_idx = next((i for i,r in enumerate(SCHEDULE_2026) if r['Status']=='upcoming'),0)
        sel_label   = st.selectbox("", race_labels, index=default_idx, label_visibility='collapsed')
        sel_idx     = race_labels.index(sel_label)
        sel_race    = SCHEDULE_2026[sel_idx]
        is_upcoming = sel_race['Status']=='upcoming'

        if is_upcoming:
            st.markdown('<div class="sb-section">QUALIFYING GRID</div>', unsafe_allow_html=True)
            st.caption("Update after Saturday qualifying")
            quali_positions = {}
            for i, d in enumerate(DRIVERS_2026):
                color = tc(d['TeamName'])
                v = st.number_input(
                    f"{d['Abbreviation']}",
                    min_value=1, max_value=22, value=i+1, step=1,
                    key=f"q_{d['Abbreviation']}", help=d['FullName'])
                quali_positions[d['Abbreviation']] = v

            st.markdown('<div class="sb-section">WEATHER</div>', unsafe_allow_html=True)
            track_temp = st.slider("Track °C", 15, 55, 28)
            air_temp   = st.slider("Air °C",   10, 45, 20)
            rain       = st.slider("Rain mm",  0.0, 10.0, 0.0, .1)
            wind       = st.slider("Wind km/h",0, 60, 15)
        else:
            quali_positions = {d['Abbreviation']:i+1 for i,d in enumerate(DRIVERS_2026)}
            track_temp,air_temp,rain,wind = 28,20,0.0,15
    else:
        is_upcoming = False
        if not df_all.empty:
            years = sorted(df_all['Year'].unique(),reverse=True)
            st.markdown('<div class="sb-section">YEAR</div>', unsafe_allow_html=True)
            sel_year = st.selectbox("", years, label_visibility='collapsed')
            rdf = df_all[df_all['Year']==sel_year]
            rounds = sorted(rdf['Round'].unique())
            rlabels = [f"R{r:02d} · {rdf[rdf['Round']==r]['RaceName'].iloc[0].replace(' Grand Prix','').upper()}"
                       for r in rounds]
            st.markdown('<div class="sb-section">ROUND</div>', unsafe_allow_html=True)
            sel_rl    = st.selectbox("", rlabels, label_visibility='collapsed')
            sel_round = rounds[rlabels.index(sel_rl)]
            hist_race_name = rdf[rdf['Round']==sel_round]['RaceName'].iloc[0]

# ── HERO ──────────────────────────────────────────────────────────────────────
if is_2026:
    title   = sel_race['RaceName'].replace(' Grand Prix','').upper() + ' GRAND PRIX'
    circuit = sel_race['Circuit']
    date    = sel_race['Date']
    rnd     = f"ROUND {sel_race['Round']:02d} / 24"
    badge   = '<span class="upcoming-pill">PREDICTION LIVE</span>' if is_upcoming else ''
else:
    title   = hist_race_name.replace(' Grand Prix','').upper() + ' GRAND PRIX'
    circuit = ''
    date    = str(sel_year)
    rnd     = f"ROUND {sel_round:02d}"
    badge   = ''

st.markdown(f"""
<div class="hero-wrap">
  <svg class="hero-svg" viewBox="0 0 340 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMaxYMid slice">
    <rect x="200" y="0"  width="20" height="20" fill="#fff"/>
    <rect x="220" y="0"  width="20" height="20" fill="#E10600"/>
    <rect x="240" y="0"  width="20" height="20" fill="#fff"/>
    <rect x="260" y="0"  width="20" height="20" fill="#E10600"/>
    <rect x="280" y="0"  width="20" height="20" fill="#fff"/>
    <rect x="300" y="0"  width="20" height="20" fill="#E10600"/>
    <rect x="320" y="0"  width="20" height="20" fill="#fff"/>
    <rect x="200" y="20" width="20" height="20" fill="#E10600"/>
    <rect x="220" y="20" width="20" height="20" fill="#fff"/>
    <rect x="240" y="20" width="20" height="20" fill="#E10600"/>
    <rect x="260" y="20" width="20" height="20" fill="#fff"/>
    <rect x="280" y="20" width="20" height="20" fill="#E10600"/>
    <rect x="300" y="20" width="20" height="20" fill="#fff"/>
    <rect x="320" y="20" width="20" height="20" fill="#E10600"/>
    <rect x="200" y="40" width="20" height="20" fill="#fff"/>
    <rect x="220" y="40" width="20" height="20" fill="#E10600"/>
    <rect x="240" y="40" width="20" height="20" fill="#fff"/>
    <rect x="260" y="40" width="20" height="20" fill="#E10600"/>
    <line x1="310" y1="55" x2="340" y2="120" stroke="#E10600" stroke-width="14" opacity="0.4"/>
    <line x1="288" y1="55" x2="318" y2="120" stroke="#E10600" stroke-width="9"  opacity="0.25"/>
    <line x1="268" y1="55" x2="298" y2="120" stroke="#E10600" stroke-width="5"  opacity="0.15"/>
    <path d="M0 90 Q70 45 150 68 Q230 88 290 52 Q315 38 340 50" stroke="#E10600" stroke-width="2" fill="none"/>
  </svg>
  <div class="race-label">{rnd} · {circuit} · {model_choice}</div>
  <div class="race-title">{title}{badge}</div>
  <div class="race-meta"><strong>{date}</strong> · FIA FORMULA ONE WORLD CHAMPIONSHIP</div>
</div>""", unsafe_allow_html=True)

if is_upcoming:
    st.markdown('<div class="warn-banner">⚑ PREDICTION MODE — Based on 2022–2025 FastF1 data. Update qualifying positions after Saturday Q3 for maximum accuracy. 2026 technical regulations introduce new variables.</div>', unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
if not df_comparison.empty:
    rw = df_comparison[(df_comparison['Model']==model_choice)&(df_comparison['Target']=='Winner')]
    rp = df_comparison[(df_comparison['Model']==model_choice)&(df_comparison['Target']=='Podium')]
    w_acc = f"{rw['Accuracy'].values[0]*100:.0f}%" if not rw.empty else "—"
    p_acc = f"{rp['Accuracy'].values[0]*100:.0f}%" if not rp.empty else "—"
else:
    w_acc,p_acc = "—","—"

total_races = len(df_all[['Year','Round']].drop_duplicates()) if not df_all.empty else 0
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <svg class="svg-wm" viewBox="0 0 120 75" xmlns="http://www.w3.org/2000/svg">
      <path d="M5 65 Q35 15 65 38 Q95 60 118 22" stroke="#E10600" stroke-width="3" fill="none"/>
      <circle cx="5"   cy="65" r="5" fill="#E10600"/>
      <circle cx="65"  cy="38" r="5" fill="#E10600"/>
      <circle cx="118" cy="22" r="5" fill="#E10600"/>
    </svg>
    <div class="wi"><div class="metric-num">{w_acc}</div><div class="metric-lbl">Winner Accuracy</div><div class="metric-sub">{model_choice} model</div></div>
  </div>
  <div class="metric-card">
    <svg class="svg-wm" viewBox="0 0 120 75" xmlns="http://www.w3.org/2000/svg">
      <polygon points="60,4 72,42 112,42 80,65 92,6 60,30 28,6 40,65 8,42 48,42" stroke="#E10600" stroke-width="1.5" fill="none"/>
    </svg>
    <div class="wi"><div class="metric-num">{p_acc}</div><div class="metric-lbl">Podium Accuracy</div><div class="metric-sub">Top 3 prediction</div></div>
  </div>
  <div class="metric-card">
    <svg class="svg-wm" viewBox="0 0 120 75" xmlns="http://www.w3.org/2000/svg">
      <rect x="5"  y="42" width="14" height="28" fill="#E10600"/>
      <rect x="24" y="28" width="14" height="42" fill="#E10600"/>
      <rect x="43" y="34" width="14" height="36" fill="#E10600"/>
      <rect x="62" y="18" width="14" height="52" fill="#E10600"/>
      <rect x="81" y="24" width="14" height="46" fill="#E10600"/>
      <rect x="100" y="10" width="14" height="60" fill="#E10600"/>
    </svg>
    <div class="wi"><div class="metric-num">{total_races}</div><div class="metric-lbl">Races in Dataset</div><div class="metric-sub">2022 — 2025</div></div>
  </div>
  <div class="metric-card">
    <svg class="svg-wm" viewBox="0 0 120 75" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="37" r="30" stroke="#E10600" stroke-width="2.5" fill="none"/>
      <line x1="60" y1="7"  x2="60" y2="37" stroke="#E10600" stroke-width="3"/>
      <line x1="60" y1="37" x2="85" y2="52" stroke="#E10600" stroke-width="2"/>
      <circle cx="60" cy="37" r="4" fill="#E10600"/>
      <line x1="60" y1="2"  x2="60" y2="0"  stroke="#E10600" stroke-width="2"/>
      <line x1="90" y1="37" x2="93" y2="37" stroke="#E10600" stroke-width="2"/>
      <line x1="30" y1="37" x2="27" y2="37" stroke="#E10600" stroke-width="2"/>
    </svg>
    <div class="wi"><div class="metric-num">16</div><div class="metric-lbl">Features Used</div><div class="metric-sub">Quali · Weather · Form</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── PREDICTIONS ───────────────────────────────────────────────────────────────
preds     = None
actual_df = None
is_hist   = False

def compute_season_accuracy(df, model_name, year=None):
    src = df[df['Year']==year].copy() if year else df.copy()
    rounds = src[['Year','Round']].drop_duplicates()
    w_correct = p_correct = d_correct = total = 0
    for _, r in rounds.iterrows():
        rd = src[(src['Year']==r['Year'])&(src['Round']==r['Round'])].copy()
        if rd.empty: continue
        try:
            pr = predict_historical(rd, model_name)
            aw = rd[rd['Position']==1]['Abbreviation'].values
            ap = set(rd[rd['Position']<=3]['Abbreviation'].values)
            ad = set(rd[rd['DNF']==1]['Abbreviation'].values)
            pw = pr.iloc[0]['Abbreviation'] if not pr.empty else ''
            pp = set(pr.head(3)['Abbreviation'].values)
            ph = set(pr[pr['DNF_Prob']>25]['Abbreviation'].values)
            if len(aw)>0 and pw==aw[0]: w_correct+=1
            if len(pp & ap)>=2: p_correct+=1
            if len(ph & ad)>=1 or (len(ad)==0 and len(ph)==0): d_correct+=1
            total+=1
        except: pass
    if total==0: return {}
    return {'winner':w_correct,'podium':p_correct,'dnf':d_correct,'total':total}

try:
    if is_2026 and is_upcoming:
        weather = {'AvgTrackTemp':track_temp,'AvgAirTemp':air_temp,
                   'AvgRainfall':rain,'AvgWindSpeed':wind}
        preds = predict_upcoming(sel_race['RaceName'], quali_positions, weather, df_all, model_choice)
    elif is_2026 and not is_upcoming:
        kw    = sel_race['RaceName'].split(' Grand')[0]
        rdata = df_all[df_all['RaceName'].str.contains(kw,case=False,na=False)]
        if not rdata.empty:
            actual_df = rdata.sort_values('Position').reset_index(drop=True)
            preds     = predict_historical(rdata.copy(), model_choice)
            is_hist   = True
    else:
        rdata = df_all[(df_all['Year']==sel_year)&(df_all['Round']==sel_round)].copy()
        if not rdata.empty:
            actual_df = rdata.sort_values('Position').reset_index(drop=True)
            preds     = predict_historical(rdata, model_choice)
            is_hist   = True
except Exception as e:
    st.error(f'Prediction error: {e}')
if preds is not None and not preds.empty:

    # Evaluation banner for completed races
    if is_hist and actual_df is not None:
        aw = actual_df[actual_df['Position']==1]['Abbreviation'].values
        ap = list(actual_df[actual_df['Position']<=3]['Abbreviation'].values)
        pw = preds.iloc[0]['Abbreviation'] if not preds.empty else ''
        pp = list(preds.head(3)['Abbreviation'].values)
        win_ok   = len(aw)>0 and pw==aw[0]
        pod_hits = len(set(pp) & set(ap))
        win_col  = '#6dbf3e' if win_ok else '#E10600'
        pod_col  = '#6dbf3e' if pod_hits>=2 else '#EF9F27' if pod_hits==1 else '#E10600'
        win_txt  = 'CORRECT' if win_ok else 'INCORRECT'
        aw0      = aw[0] if len(aw)>0 else 'N/A'
        ap_str   = ' · '.join(ap)
        pp_str   = ' · '.join(pp)
        st.markdown(f"""
<div style='position:relative;overflow:hidden;background:#0d0d0d;border:1px solid #1a1a1a;
            border-top:2px solid #E10600;border-radius:4px;padding:1rem 1.5rem;margin-bottom:1.25rem'>
  <div style='position:relative;z-index:1'>
    <div style='font-family:Orbitron,monospace;font-size:.58rem;color:#E10600;
                letter-spacing:.2em;text-transform:uppercase;margin-bottom:.75rem'>MODEL EVALUATION</div>
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem'>
      <div>
        <div style='font-family:Barlow Condensed,sans-serif;font-size:.62rem;color:#444;
                    letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px'>Winner Prediction</div>
        <div style='font-family:Orbitron,monospace;font-size:.95rem;font-weight:700;color:{win_col}'>{win_txt}</div>
        <div style='font-family:Barlow,sans-serif;font-size:.78rem;color:#666;margin-top:4px'>Predicted: {pw} · Actual: {aw0}</div>
      </div>
      <div>
        <div style='font-family:Barlow Condensed,sans-serif;font-size:.62rem;color:#444;
                    letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px'>Podium Hits</div>
        <div style='font-family:Orbitron,monospace;font-size:.95rem;font-weight:700;color:{pod_col}'>{pod_hits}/3 CORRECT</div>
        <div style='font-family:Barlow,sans-serif;font-size:.78rem;color:#666;margin-top:4px'>Actual: {ap_str}</div>
      </div>
      <div>
        <div style='font-family:Barlow Condensed,sans-serif;font-size:.62rem;color:#444;
                    letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px'>Predicted Top 3</div>
        <div style='font-family:Barlow Condensed,sans-serif;font-size:.9rem;color:#ccc'>{pp_str}</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1,1], gap="large")

    with left:
        medals = ['1st','2nd','3rd']
        if is_hist and actual_df is not None:
            actual_p3    = actual_df[actual_df['Position']<=3].head(3)
            pred_p3      = preds.head(3)
            actual_abbrs = list(actual_p3['Abbreviation'].values)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="sec-title">ACTUAL RESULT</div>', unsafe_allow_html=True)
                for idx, (_, ar) in enumerate(actual_p3.iterrows()):
                    clr = tc(ar['TeamName'])
                    cls = ['p1','p2','p3'][idx]
                    st.markdown(f'''
                    <div class="podium-card {cls}">
                      <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:.06;pointer-events:none" viewBox="0 0 300 72" xmlns="http://www.w3.org/2000/svg">
                        <path d="M0 55 Q60 22 120 42 Q180 60 240 30 Q270 16 300 28" stroke="#E10600" stroke-width="2" fill="none"/>
                        <line x1="265" y1="0" x2="300" y2="72" stroke="#E10600" stroke-width="10" opacity="0.35"/>
                      </svg>
                      <div class="team-stripe" style="background:{clr}"></div>
                      <div class="podium-pos" style="font-size:1rem">{idx+1}</div>
                      <div class="podium-info">
                        <div class="podium-name" style="font-size:.82rem">{ar["Abbreviation"]}</div>
                        <div class="podium-team">{str(ar["TeamName"])[:15]}</div>
                      </div>
                    </div>''', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="sec-title">MODEL PREDICTED</div>', unsafe_allow_html=True)
                for idx, (_, pr) in enumerate(pred_p3.iterrows()):
                    clr  = tc(pr['TeamName'])
                    cls  = ['p1','p2','p3'][idx]
                    ok   = pr['Abbreviation'] in actual_abbrs
                    tc2  = '#6dbf3e' if ok else '#E10600'
                    tick = 'V' if ok else 'X'
                    st.markdown(f'''
                    <div class="podium-card {cls}">
                      <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:.06;pointer-events:none" viewBox="0 0 300 72" xmlns="http://www.w3.org/2000/svg">
                        <path d="M0 55 Q60 22 120 42 Q180 60 240 30 Q270 16 300 28" stroke="#E10600" stroke-width="2" fill="none"/>
                        <line x1="265" y1="0" x2="300" y2="72" stroke="#E10600" stroke-width="10" opacity="0.35"/>
                      </svg>
                      <div class="team-stripe" style="background:{clr}"></div>
                      <div class="podium-pos" style="font-size:1rem">{idx+1}</div>
                      <div class="podium-info">
                        <div class="podium-name" style="font-size:.82rem">{pr["Abbreviation"]}</div>
                        <div class="podium-team">{str(pr["TeamName"])[:15]}</div>
                      </div>
                      <div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;color:{tc2}">{tick}</div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-title">PREDICTED PODIUM</div>', unsafe_allow_html=True)
            cls_map = {0:'p1',1:'p2',2:'p3'}
            pos_lbl = {0:'01',1:'02',2:'03'}
            for i, row in preds.head(3).iterrows():
                color = tc(row['TeamName'])
                cls   = cls_map.get(i,'')
                pos   = pos_lbl.get(i,str(i+1))
                st.markdown(f'''
                <div class="podium-card {cls}">
                  <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:.06;pointer-events:none" viewBox="0 0 300 72" xmlns="http://www.w3.org/2000/svg">
                    <path d="M0 55 Q60 22 120 42 Q180 60 240 30 Q270 16 300 28" stroke="#E10600" stroke-width="2" fill="none"/>
                    <line x1="265" y1="0" x2="300" y2="72" stroke="#E10600" stroke-width="10" opacity="0.35"/>
                  </svg>
                  <div class="team-stripe" style="background:{color}"></div>
                  <div class="podium-pos">{pos}</div>
                  <div class="podium-info">
                    <div class="podium-name">{row["FullName"].upper()}</div>
                    <div class="podium-team">{row["TeamName"]} · P{int(row["QualiPosition"])} QUALI</div>
                  </div>
                  <div>
                    <div class="podium-prob">{row["Winner_Prob"]:.1f}%</div>
                    <div class="podium-prob-lbl">WIN PROB</div>
                  </div>
                </div>''', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)


        # DNF Risk
        st.markdown("""
        <div style="position:relative;overflow:hidden;background:#0d0d0d;border:1px solid #1a1a1a;border-top:2px solid #E10600;border-radius:4px;padding:1rem 1.25rem;margin-top:1rem">
          <svg style="position:absolute;right:-10px;bottom:-10px;width:160px;height:100px;opacity:.05;pointer-events:none" viewBox="0 0 160 100" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 65 L55 48 L75 28 L105 26 L120 48 L148 52 L148 70 L10 72 Z" fill="#E10600"/>
            <ellipse cx="42"  cy="72" rx="16" ry="10" fill="#E10600"/>
            <ellipse cx="118" cy="72" rx="16" ry="10" fill="#E10600"/>
            <rect x="76" y="26" width="30" height="14" rx="2" fill="#fff" opacity="0.4"/>
          </svg>
          <div style="position:relative;z-index:1">
        """, unsafe_allow_html=True)
        st.markdown('<div class="sec-title">DNF RISK ASSESSMENT</div>', unsafe_allow_html=True)
        for _, row in preds.sort_values('DNF_Prob',ascending=False).head(6).iterrows():
            risk = row['DNF_Prob']
            cls  = 'risk-hi' if risk>25 else 'risk-med' if risk>15 else 'risk-lo'
            lbl  = 'HIGH' if risk>25 else 'MED' if risk>15 else 'LOW'
            st.markdown(f"""
            <div class="dnf-row">
              <div class="dnf-code">{row['Abbreviation']}</div>
              <div class="dnf-team">{row['TeamName'][:18]}</div>
              <div class="dnf-badge {cls}">{lbl} · {risk:.0f}%</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with right:
        # Win prob chart
        st.markdown('<div class="sec-title">WIN PROBABILITY RANKING</div>', unsafe_allow_html=True)
        top10 = preds.head(10)
        fig = go.Figure(go.Bar(
            x=top10['Winner_Prob'], y=top10['Abbreviation'],
            orientation='h',
            marker=dict(
                color=[tc(t) for t in top10['TeamName']],
                line=dict(width=0),
            ),
            text=[f"  {v:.1f}%" for v in top10['Winner_Prob']],
            textposition='outside',
            textfont=dict(color='#888', size=11, family='Barlow Condensed'),
        ))
        fig.update_layout(
            plot_bgcolor='#080808', paper_bgcolor='#080808',
            font=dict(color='#666', size=10, family='Barlow Condensed'),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(autorange='reversed', tickfont=dict(size=12, color='#ccc'),
                       tickfont_family='Barlow Condensed'),
            margin=dict(l=10,r=70,t=10,b=10), height=310, bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Podium prob chart
        st.markdown('<div class="sec-title">PODIUM PROBABILITY</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=top10['Abbreviation'], y=top10['Podium_Prob'],
            marker_color=[tc(t) for t in top10['TeamName']],
            marker_line=dict(width=0),
            text=[f"{v:.0f}%" for v in top10['Podium_Prob']],
            textposition='outside',
            textfont=dict(color='#666', size=10, family='Barlow Condensed'),
        ))
        fig2.update_layout(
            plot_bgcolor='#080808', paper_bgcolor='#080808',
            font=dict(color='#666', size=10, family='Barlow Condensed'),
            xaxis=dict(showgrid=False, tickfont=dict(color='#ccc', family='Barlow Condensed')),
            yaxis=dict(showgrid=False, showticklabels=False,
                       range=[0, preds['Podium_Prob'].max()*1.25]),
            margin=dict(l=10,r=10,t=10,b=10), height=210, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Full grid table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="position:relative;overflow:hidden;background:#0d0d0d;border:1px solid #1a1a1a;border-top:2px solid #E10600;border-radius:4px;padding:1rem 1.25rem 0">
      <svg style="position:absolute;right:0;top:0;width:400px;height:100%;opacity:.04;pointer-events:none" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMaxYMid slice">
        <rect x="220" y="0"  width="18" height="18" fill="#fff"/>
        <rect x="238" y="0"  width="18" height="18" fill="#E10600"/>
        <rect x="256" y="0"  width="18" height="18" fill="#fff"/>
        <rect x="274" y="0"  width="18" height="18" fill="#E10600"/>
        <rect x="292" y="0"  width="18" height="18" fill="#fff"/>
        <rect x="310" y="0"  width="18" height="18" fill="#E10600"/>
        <rect x="328" y="0"  width="18" height="18" fill="#fff"/>
        <rect x="346" y="0"  width="18" height="18" fill="#E10600"/>
        <rect x="220" y="18" width="18" height="18" fill="#E10600"/>
        <rect x="238" y="18" width="18" height="18" fill="#fff"/>
        <rect x="256" y="18" width="18" height="18" fill="#E10600"/>
        <rect x="274" y="18" width="18" height="18" fill="#fff"/>
        <rect x="292" y="18" width="18" height="18" fill="#E10600"/>
        <rect x="310" y="18" width="18" height="18" fill="#fff"/>
        <path d="M0 140 Q100 80 200 110 Q300 135 400 90" stroke="#E10600" stroke-width="2" fill="none"/>
      </svg>
      <div style="position:relative;z-index:1">
    """, unsafe_allow_html=True)
    st.markdown('<div class="sec-title">FULL GRID ANALYSIS</div>', unsafe_allow_html=True)
    disp = preds[['PredictedPosition','Abbreviation','FullName','TeamName',
                  'QualiPosition','Winner_Prob','Podium_Prob','DNF_Prob']].copy()
    disp.columns = ['POS','CODE','DRIVER','TEAM','QUALI','WIN %','PODIUM %','DNF %']
    st.dataframe(
        disp.style
            .background_gradient(subset=['WIN %'], cmap='Reds')
            .background_gradient(subset=['DNF %'], cmap='RdYlGn_r')
            .format({'WIN %':'{:.1f}','PODIUM %':'{:.1f}','DNF %':'{:.1f}'}),
        use_container_width=True, hide_index=True
    )
    st.markdown('</div></div>', unsafe_allow_html=True)
else:
    st.info("No data available for this selection.")

# ── Model Comparison ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("▸  MODEL PERFORMANCE COMPARISON"):
    if not df_comparison.empty:
        for target in ['Winner','Podium','DNF']:
            st.markdown(f'<div class="sec-title">{target.upper()} PREDICTION ACCURACY</div>', unsafe_allow_html=True)
            tdf = df_comparison[df_comparison['Target']==target]
            fig = go.Figure()
            for metric,color in [('Accuracy','#E10600'),('F1_Score','#EF9F27'),('ROC_AUC','#3671C6')]:
                fig.add_trace(go.Bar(
                    name=metric, x=tdf['Model'], y=tdf[metric],
                    marker_color=color, marker_line=dict(width=0),
                    text=[f"{v:.3f}" for v in tdf[metric]],
                    textposition='outside',
                    textfont=dict(color='#888',size=10,family='Barlow Condensed'),
                ))
            fig.update_layout(
                plot_bgcolor='#080808', paper_bgcolor='#080808',
                font=dict(color='#666',size=10,family='Barlow Condensed'),
                barmode='group',
                xaxis=dict(showgrid=False, tickfont=dict(color='#ccc',family='Orbitron',size=9)),
                yaxis=dict(showgrid=False, range=[0,1.1], showticklabels=False),
                margin=dict(l=10,r=10,t=10,b=10), height=200,
                legend=dict(orientation='h',y=1.15,
                            font=dict(family='Barlow Condensed',size=10,color='#888')),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run python models/train.py first")

# Season Accuracy Tracker
if is_hist and not df_all.empty:
    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander('SEASON ACCURACY TRACKER — Real-World Model Evaluation'):
        st.caption('Pick a season and click compute to see real-world accuracy vs training accuracy.')
        track_year = st.selectbox('Evaluate season',
                                   sorted(df_all['Year'].unique(), reverse=True),
                                   key='tracker_year')
        if st.button('COMPUTE SEASON ACCURACY', key='compute_acc'):
            with st.spinner('Evaluating all races in season...'):
                stats = compute_season_accuracy(df_all, model_choice, year=track_year)
            if stats:
                w_pct = stats['winner']/stats['total']*100
                p_pct = stats['podium']/stats['total']*100
                d_pct = stats['dnf']  /stats['total']*100
                wc = '#6dbf3e' if w_pct>=50 else '#EF9F27' if w_pct>=35 else '#E10600'
                pc = '#6dbf3e' if p_pct>=60 else '#EF9F27' if p_pct>=45 else '#E10600'
                dc = '#6dbf3e' if d_pct>=65 else '#EF9F27' if d_pct>=50 else '#E10600'
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{wc}">{w_pct:.0f}%</div><div class="metric-lbl">Winner Correct</div><div class="metric-sub">{stats["winner"]}/{stats["total"]} races</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card" style="border-top-color:#EF9F27"><div class="metric-num" style="color:{pc}">{p_pct:.0f}%</div><div class="metric-lbl">Podium 2/3 Correct</div><div class="metric-sub">{stats["podium"]}/{stats["total"]} races</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card" style="border-top-color:#3671C6"><div class="metric-num" style="color:{dc}">{d_pct:.0f}%</div><div class="metric-lbl">DNF Flagged Correct</div><div class="metric-sub">{stats["dnf"]}/{stats["total"]} races</div></div>', unsafe_allow_html=True)
                st.caption(f'Real-world accuracy on {track_year} season using {model_choice}. Lower than training accuracy is normal — this is the honest number.')
            else:
                st.warning('Could not compute — try a different year or model.')

st.markdown('<div class="f1-caption">F1 RACE PREDICTOR · FASTF1 · XGBOOST · RANDOM FOREST · NEURAL NET · 2026 SEASON</div>', unsafe_allow_html=True)