"""
app.py — Ransomware Forensic Triage · console UI
Explainable, human-in-the-loop prioritisation of ransomware forensic artefacts.
Loads the real trained XGBoost model, SHAP explainer and operating point.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

# ------------------------------------------------------------------ palette
BG0="#090C11"; PANEL="#0F141C"; RAISED="#131A24"; BORDER="#1E2A38"
TEXT="#E6EDF3"; MUTED="#8A97A6"; BRAND1="#2DD4BF"; BRAND2="#7C6FF0"
ACCENT="#38BDF8"; GREEN="#34D399"; AMBER="#F59E0B"; RED="#F26D6D"
DEC_COL={"AUTO-CLEAR":GREEN,"ESCALATE":AMBER,"AUTO-FLAG":RED}

st.set_page_config(page_title="Ransomware Forensic Triage", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
:root {{
  --bg0:{BG0}; --panel:{PANEL}; --raised:{RAISED}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --b1:{BRAND1}; --b2:{BRAND2};
  --green:{GREEN}; --amber:{AMBER}; --red:{RED};
}}
.stApp {{
  background:
    radial-gradient(1200px 600px at 15% -5%, rgba(45,212,191,.10), transparent 60%),
    radial-gradient(1000px 600px at 95% 0%, rgba(124,111,240,.12), transparent 55%),
    var(--bg0);
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
}}
#MainMenu, header[data-testid="stHeader"], footer {{visibility:hidden; height:0;}}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#0C111A,#0A0E14); border-right:1px solid var(--border); }}
.block-container {{ padding-top:1.2rem; max-width:1250px; }}
h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif; letter-spacing:.2px; }}

/* hero */
.hero {{ padding:6px 2px 2px; }}
.hero .kick {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:3px;
  color:var(--muted); text-transform:uppercase; }}
.hero .title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.5rem; line-height:1.05;
  margin:.15rem 0 .1rem; background:linear-gradient(92deg,var(--b1),var(--b2)); -webkit-background-clip:text;
  background-clip:text; color:transparent; }}
.hero .sub {{ color:#B9C4D0; font-size:1.02rem; }}
.pill {{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:2px;
  text-transform:uppercase; color:var(--b1); border:1px solid rgba(45,212,191,.35);
  background:rgba(45,212,191,.06); padding:5px 12px; border-radius:999px; }}

/* section header */
.sec {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; letter-spacing:3px; text-transform:uppercase;
  color:var(--muted); margin:.2rem 0 .6rem; display:flex; align-items:center; gap:8px; }}
.sec b {{ color:var(--b1); }}

/* tiles */
.tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:.2rem 0 .3rem; }}
.tile {{ background:linear-gradient(180deg,var(--raised),var(--panel)); border:1px solid var(--border);
  border-radius:16px; padding:16px 18px; position:relative; overflow:hidden; }}
.tile:before {{ content:""; position:absolute; left:0; top:0; height:3px; width:100%;
  background:linear-gradient(90deg,var(--b1),var(--b2)); opacity:.9; }}
.tile .lab {{ font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:2px; text-transform:uppercase;
  color:var(--muted); }}
.tile .val {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.9rem; margin-top:4px;
  color:var(--text); }}
.tile .val.g {{ color:var(--green);}} .tile .val.a {{ color:var(--amber);}} .tile .val.t {{ color:var(--b1);}}

/* generic card */
.card {{ background:linear-gradient(180deg,var(--raised),var(--panel)); border:1px solid var(--border);
  border-radius:16px; padding:18px 20px; }}

/* verdict badge */
.badge {{ display:inline-flex; align-items:center; gap:8px; font-family:'Space Grotesk',sans-serif;
  font-weight:700; font-size:1.05rem; padding:9px 16px; border-radius:12px; letter-spacing:.5px; }}
.badge.clear {{ color:#062; background:rgba(52,211,153,.16); border:1px solid rgba(52,211,153,.5); box-shadow:0 0 22px rgba(52,211,153,.18);}}
.badge.esc   {{ color:#7a5; background:rgba(245,158,11,.14); border:1px solid rgba(245,158,11,.5); box-shadow:0 0 22px rgba(245,158,11,.16);}}
.badge.flag  {{ color:#e88; background:rgba(242,109,109,.14); border:1px solid rgba(242,109,109,.55); box-shadow:0 0 22px rgba(242,109,109,.18);}}
.badge .dot {{ width:9px; height:9px; border-radius:50%; }}
.clear .dot{{background:var(--green);}} .esc .dot{{background:var(--amber);}} .flag .dot{{background:var(--red);}}

/* probability meter */
.meter {{ position:relative; height:16px; border-radius:10px; overflow:hidden; border:1px solid var(--border);
  display:flex; margin:10px 0 6px; }}
.meter .zg {{ background:rgba(52,211,153,.5);}} .meter .za {{ background:rgba(245,158,11,.5);}} .meter .zf {{ background:rgba(242,109,109,.55);}}
.mark {{ position:relative; height:26px; margin-top:-3px;}}
.mark i {{ position:absolute; top:0; width:2px; height:22px; background:#fff; box-shadow:0 0 8px #fff; }}
.mark i:after {{ content:attr(data-v); position:absolute; top:-18px; left:50%; transform:translateX(-50%);
  font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:#fff; white-space:nowrap;}}
.mlab {{ display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  color:var(--muted); }}

/* reason chips */
.reason {{ display:flex; align-items:center; gap:10px; padding:8px 12px; border:1px solid var(--border);
  border-radius:10px; background:var(--panel); margin-bottom:7px; font-size:.9rem;}}
.reason .tag {{ font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:1px; text-transform:uppercase;
  padding:3px 8px; border-radius:6px; }}
.reason .up {{ color:var(--red); background:rgba(242,109,109,.12);}}
.reason .dn {{ color:var(--green); background:rgba(52,211,153,.12);}}
.reason .ft {{ font-family:'IBM Plex Mono',monospace; color:var(--text);}}

/* steps */
.steps {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
.step {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; }}
.step .n {{ font-family:'Space Grotesk'; font-weight:700; color:var(--b1); font-size:1.2rem;}}
.step .t {{ font-weight:600; margin:3px 0 2px;}} .step .d {{ color:var(--muted); font-size:.82rem;}}

/* disclaimer */
.disc {{ border:1px solid rgba(245,158,11,.4); background:rgba(245,158,11,.06); border-radius:12px;
  padding:12px 16px; font-family:'IBM Plex Mono',monospace; font-size:.8rem; color:#d7c08a;}}

/* footer */
.foot {{ text-align:center; font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:2px;
  color:var(--muted); text-transform:uppercase; margin:26px 0 6px; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono',monospace; font-size:.8rem; letter-spacing:1px;
  text-transform:uppercase; color:var(--muted); background:transparent; border-radius:8px 8px 0 0; padding:8px 14px;}}
.stTabs [aria-selected="true"] {{ color:var(--b1); border-bottom:2px solid var(--b1); }}

/* buttons */
.stButton>button, [data-testid="stDownloadButton"] button {{
  font-family:'IBM Plex Mono',monospace; letter-spacing:1px; text-transform:uppercase; font-weight:600;
  border-radius:10px; border:1px solid rgba(45,212,191,.4);
  background:linear-gradient(92deg,rgba(45,212,191,.16),rgba(124,111,240,.16)); color:var(--text);}}
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover {{ border-color:var(--b1); }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ helpers
def sec(text):
    st.markdown(f"<div class='sec'><b>◆</b> {text}</div>", unsafe_allow_html=True)

def tiles(items):
    cells="".join(
        f"<div class='tile'><div class='lab'>{l}</div><div class='val {c}'>{v}</div></div>"
        for l,v,c in items)
    st.markdown(f"<div class='tiles'>{cells}</div>", unsafe_allow_html=True)

def badge(dec):
    cls={"AUTO-CLEAR":"clear","ESCALATE":"esc","AUTO-FLAG":"flag"}[dec]
    st.markdown(f"<span class='badge {cls}'><span class='dot'></span>{dec}</span>", unsafe_allow_html=True)

def meter(p,low,high):
    g,a,f=low*100,(high-low)*100,(1-high)*100
    st.markdown(
      f"<div class='meter'><div class='zg' style='width:{g}%'></div>"
      f"<div class='za' style='width:{a}%'></div><div class='zf' style='width:{f}%'></div></div>"
      f"<div class='mark'><i data-v='{p:.1%}' style='left:{min(max(p,0),1)*100:.1f}%'></i></div>"
      f"<div class='mlab'><span>0%</span><span>auto-clear ▸ {low:.1%}</span>"
      f"<span>{high:.1%} ◂ auto-flag</span><span>100%</span></div>", unsafe_allow_html=True)

def dark_ax(fig,ax):
    fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for s in ax.spines.values(): s.set_color(BORDER)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

def align(df,feats):
    x=df.copy(); x.columns=[str(c).strip() for c in x.columns]
    for f in feats:
        if f not in x.columns: x[f]=0
    return x[feats].apply(pd.to_numeric,errors="coerce").fillna(0)

# ------------------------------------------------------------------ load
@st.cache_resource(show_spinner="Loading model…")
def get_bundle(): return core.get_bundle()

try:
    bundle,how=get_bundle()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

model=bundle["model"]; feats=bundle["feature_names"]; bank=bundle["sample_bank"]
meta=bundle.get("meta",{}); rep=meta.get("reported",{}); low0,high0=bundle["thresholds"]
explainer=bundle.get("explainer")

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class='hero'>
  <div class='kick'>MSc Cyber Security Dissertation · CN7000</div>
  <div class='title'>🛡 Ransomware Forensic Triage</div>
  <div class='sub'>Explainable, human-in-the-loop prioritisation of forensic artefacts — XGBoost · SHAP · MLRan</div>
  <div style='margin-top:10px'><span class='pill'>Real trained model · loaded live</span></div>
</div>""", unsafe_allow_html=True)
st.write("")

tiles([
 ("Detection accuracy", f"{rep.get('detection_accuracy',0)*100:.2f}%","t"),
 ("Workload auto-handled", f"{rep.get('queue_reduction',0)*100:.2f}%","g"),
 ("Escalated to analyst", f"{rep.get('escalation_rate',0)*100:.2f}%","a"),
 ("Behavioural features", f"{meta.get('n_features','—')}",""),
])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<div class='sec'><b>◆</b> Control</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='card' style='padding:12px 14px'>"
                f"<div class='tile' style='border:none;padding:0'><div class='lab'>Model</div>"
                f"<div class='val t' style='font-size:1.1rem'>{meta.get('best_model','—')}</div></div>"
                f"<div style='color:{MUTED};font-family:IBM Plex Mono;font-size:.72rem;margin-top:6px'>"
                f"{'loaded from disk' if how=='real' else 'trained on launch'} · "
                f"{meta.get('live_test',{}).get('n','—')} hold-out samples</div></div>",
                unsafe_allow_html=True)
    st.write("")
    sec("Triage thresholds")
    st.caption("Below **low** → auto-clear · between → escalate · at/above **high** → auto-flag.")
    low=st.slider("Low (auto-clear ceiling)",0.0,1.0,float(low0),0.005)
    high=st.slider("High (auto-flag floor)",0.0,1.0,float(high0),0.005)
    if low>=high: st.warning("Low must be below High."); low,high=low0,high0
    st.markdown("<div class='disc'>⚠ Behavioural features only — no malware is executed.</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------ tabs
t1,t2,t3=st.tabs(["🔍 Triage a sample","📋 Investigation queue","📊 Model card"])

# ---- Tab 1 ----
with t1:
    L,Rp=st.columns([1,1.25],gap="large")
    with L:
        sec("Select artefact")
        src=st.radio("src",["Held-out sample bank","Upload one row (CSV)"],label_visibility="collapsed")
        x_row=None; true_lbl=None
        if src=="Held-out sample bank" and bank is not None:
            idx=st.selectbox("Sample",list(range(len(bank))),
                format_func=lambda i:f"#{i:03d} · true = "
                f"{'ransomware' if int(bank.iloc[i]['_true'])==1 else 'goodware'}")
            x_row=bank.iloc[[idx]][feats]; true_lbl=int(bank.iloc[idx]["_true"])
        else:
            up=st.file_uploader("Upload one row of MLRan features",type=["csv"])
            if up is not None: x_row=align(pd.read_csv(up).iloc[[0]],feats)
    with Rp:
        if x_row is not None:
            p=float(model.predict_proba(x_row)[:,1][0]); dec=core.triage_decision(p,low,high)
            sec("Verdict")
            c1,c2=st.columns([1,1])
            with c1: st.markdown(f"<div class='tile'><div class='lab'>Ransomware probability</div>"
                                 f"<div class='val'>{p:.1%}</div></div>",unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='lab' style='font-family:IBM Plex Mono;font-size:.66rem;"
                            "letter-spacing:2px;color:#8A97A6;text-transform:uppercase'>Decision</div>",
                            unsafe_allow_html=True)
                badge(dec)
                if true_lbl is not None:
                    st.caption(f"ground truth: {'ransomware' if true_lbl==1 else 'goodware'}")
            meter(p,low,high)
            sec("Why — top behaviours (SHAP)")
            top=core.shap_top_contributions(explainer,x_row,feats,7)
            rows=""
            for cbr in top:
                cls="up" if cbr["direction"]=="ransomware" else "dn"
                pres="present" if cbr["value"] else "absent"
                rows+=(f"<div class='reason'><span class='tag {cls}'>{cbr['direction']}</span>"
                       f"<span class='ft'>{cbr['feature'][:46]}</span>"
                       f"<span style='margin-left:auto;color:#8A97A6;font-size:.75rem'>{pres}</span></div>")
            st.markdown(rows,unsafe_allow_html=True)
        else:
            st.info("Pick a sample to see its verdict and explanation.")

# ---- Tab 2 ----
with t2:
    sec("Risk-ordered analyst queue")
    use=st.radio("u",["Held-out sample bank","Upload batch CSV"],horizontal=True,label_visibility="collapsed")
    batch=None; truth=None
    if use=="Held-out sample bank" and bank is not None:
        batch=bank[feats].reset_index(drop=True); truth=bank["_true"].reset_index(drop=True)
    else:
        up=st.file_uploader("Upload batch CSV",type=["csv"],key="b")
        if up is not None: batch=align(pd.read_csv(up),feats)
    if batch is not None and len(batch):
        probs=model.predict_proba(batch)[:,1]
        decs=[core.triage_decision(pp,low,high) for pp in probs]
        out=pd.DataFrame({"artefact":[f"#{i:03d}" for i in range(len(batch))],
                          "ransomware_prob":np.round(probs,4),"decision":decs})
        if truth is not None: out["true_label"]=np.where(truth.values==1,"ransomware","goodware")
        out=out.sort_values("ransomware_prob",ascending=False).reset_index(drop=True)
        n=len(out); nc=int((out.decision=="AUTO-CLEAR").sum()); ne=int((out.decision=="ESCALATE").sum())
        nf=int((out.decision=="AUTO-FLAG").sum())
        tiles([("Auto-cleared",nc,"g"),("Escalated",ne,"a"),("Auto-flagged",nf,""),
               ("Workload auto-handled",f"{100*(nc+nf)/n:.1f}%","t")])
        def _sty(r):
            c=DEC_COL[r.decision]; return [f"color:{c};font-weight:700" if col=="decision" else "" for col in r.index]
        st.dataframe(out.style.apply(_sty,axis=1),width='stretch',height=420)
        st.download_button("⬇ Download triage results (CSV)",out.to_csv(index=False).encode(),
                           "triage_results.csv","text/csv")
    else:
        st.info("Choose the sample bank or upload a batch CSV.")

# ---- Tab 3 ----
with t3:
    sec("Reported performance (full test split)")
    tiles([("Detection accuracy",f"{rep.get('detection_accuracy',0)*100:.2f}%","t"),
           ("Queue reduction",f"{rep.get('queue_reduction',0)*100:.2f}%","g"),
           ("Escalation rate",f"{rep.get('escalation_rate',0)*100:.2f}%","a"),
           ("Ransomware missed",f"{rep.get('ransomware_missed','—')}","")])
    st.write("")
    cA,cB=st.columns([1.15,1],gap="large")
    with cA:
        sec("SHAP importance by behavioural group")
        gi=rep.get("shap_group_importance",{})
        if gi:
            g=pd.DataFrame({"k":list(gi.keys()),"v":list(gi.values())}).sort_values("v")
            fig,ax=plt.subplots(figsize=(5.6,3.2),dpi=150)
            ax.barh(g.k,g.v,color=BRAND1)
            for i,(kk,vv) in enumerate(zip(g.k,g.v)): ax.text(vv+0.5,i,f"{vv:.1f}%",va="center",color=TEXT,fontsize=8)
            ax.set_xlim(0,max(g.v)+6); dark_ax(fig,ax); fig.tight_layout()
            st.pyplot(fig)
    with cB:
        sec("How it works")
        st.markdown("""
        <div class='steps' style='grid-template-columns:1fr 1fr'>
          <div class='step'><div class='n'>1</div><div class='t'>Score</div><div class='d'>Calibrated XGBoost on 483 behavioural features.</div></div>
          <div class='step'><div class='n'>2</div><div class='t'>Explain</div><div class='d'>SHAP attributes each verdict to named behaviours.</div></div>
          <div class='step'><div class='n'>3</div><div class='t'>Triage</div><div class='d'>Operating point → auto-clear / escalate / auto-flag.</div></div>
          <div class='step'><div class='n'>4</div><div class='t'>Human</div><div class='d'>Analyst owns the uncertain, consequential cases.</div></div>
        </div>""",unsafe_allow_html=True)
        st.write("")
        st.markdown(f"<div class='disc'>Operating point in use — low {low:.3f} · high {high:.3f}. "
                    "MLRan provides extracted behavioural indicators, not live malware; this app never "
                    "executes ransomware.</div>",unsafe_allow_html=True)

st.markdown("<div class='foot'>◆ Ransomware Forensic Triage · CN7000 · XGBoost · MLRan · "
            f"{rep.get('detection_accuracy',0)*100:.2f}% accuracy · {rep.get('queue_reduction',0)*100:.2f}% workload cut ◆</div>",
            unsafe_allow_html=True)
