"""
Explainable AI for Early Mental Health Risk Prediction and Personalized Intervention
Interactive Streamlit app  --  run with:  streamlit run app.py

Needs in the same folder:  risk_model_final.pkl, label_encoder_final.pkl, presets.json
"""

import base64
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="Mental Health Risk Explorer", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {min-width: 420px; max-width: 480px;}
    .risk-card {padding: 22px; border-radius: 16px; text-align: center;}
    .risk-card .lbl {font-size: 15px; opacity: 0.9;}
    .risk-card .val {font-size: 46px; font-weight: 800; line-height: 1.1;}
    .tip {padding: 12px 16px; border-left: 5px solid #4C9AFF; background: rgba(76,154,255,0.14);
          border-radius: 6px; margin-bottom: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("risk_model_final.pkl")
    label_encoder = joblib.load("label_encoder_final.pkl")
    explainer = shap.TreeExplainer(model)
    presets = {}
    if os.path.exists("presets.json"):
        with open("presets.json") as f:
            presets = json.load(f)
    return model, label_encoder, explainer, presets



# ---------------------------------------------------------------------
# Splash screen: logo shown for 3 seconds at the start of each session.
# Put your logo next to app.py as logo.png (jpg / jpeg / webp / svg also work).
# ---------------------------------------------------------------------
SPLASH_SECONDS = 3
LOGO_FILES = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"]


def find_logo():
    for name in LOGO_FILES:
        if os.path.exists(name):
            return name
    return None


def logo_data_uri(path):
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "svg": "image/svg+xml"}[path.rsplit(".", 1)[1].lower()]
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


_logo = find_logo()
splash, _t0 = None, time.time()
if _logo and not st.session_state.get("splash_done"):
    splash = st.empty()
    splash.markdown(
        "<style>@keyframes fadein{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}"
        "@keyframes fill{from{width:0}to{width:100%}}</style>"
        "<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "height:80vh;animation:fadein .8s ease'>"
        "<div style='background:#ffffff;border-radius:24px;padding:20px 28px;"
        "box-shadow:0 6px 30px rgba(0,0,0,.28)'>"
        f"<img src='{logo_data_uri(_logo)}' style='max-height:260px;max-width:70vw;display:block'/></div>"
        "<div style='margin-top:24px;font-size:26px;font-weight:700;text-align:center'>"
        "Explainable Mental Health Risk Explorer</div>"
        "<div style='margin-top:18px;width:220px;height:4px;border-radius:4px;"
        "background:rgba(128,128,128,.35);overflow:hidden'>"
        f"<div style='height:100%;background:#4ADE80;animation:fill {SPLASH_SECONDS}s linear forwards'></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

model, label_encoder, explainer, PRESETS = load_artifacts()

if splash is not None:
    time.sleep(max(0.0, SPLASH_SECONDS - (time.time() - _t0)))
    splash.empty()
    st.session_state["splash_done"] = True
FEATURES = list(model.feature_names_in_)
CLASSES = list(label_encoder.classes_)

# ---------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------
YNM = ["Yes", "No", "Maybe"]
YNM_NA = ["Yes", "No", "Maybe", "Not applicable"]
YN_NA = ["Yes", "No", "Not applicable"]
YN_DK_NA = ["Yes", "No", "I don't know", "Not applicable"]
BINARY_KEYS = {"self_employed", "tech_company", "has_previous_employers",
               "sought_treatment", "medical_coverage_mh"}

QUESTIONS = {
    "Mental health history": [
        ("diagnosed_by_professional", "Diagnosed with a mental health condition by a professional?", ["Yes", "No"]),
        ("past_mh_disorder", "Have you had a mental health disorder in the past?", YNM),
        ("sought_treatment", "Ever sought treatment from a mental health professional?", "binary"),
        ("interferes_when_treated", "Does it interfere with your work when effectively treated?",
         ["Never", "Rarely", "Sometimes", "Often", "Not applicable to me"]),
        ("productivity_affected", "Is your productivity ever affected by a mental health issue?",
         ["Yes", "No", "Unsure", "Not applicable to me"]),
        ("willing_share_family", "How open would you be sharing a mental illness with friends/family?",
         ["Very open", "Somewhat open", "Neutral", "Somewhat not open", "Not open at all",
          "Not applicable to me (I do not have a mental illness)"]),
    ],
    "Your workplace": [
        ("self_employed", "Are you self-employed?", "binary"),
        ("company_size", "Company size (employees)",
         ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000", "Not applicable"]),
        ("tech_company", "Is your employer primarily a tech company?", "binary"),
        ("remote_work", "Do you work remotely?", ["Always", "Sometimes", "Never"]),
        ("has_previous_employers", "Do you have previous employers?", "binary"),
        ("mh_benefits", "Employer provides mental health benefits?",
         ["Yes", "No", "I don't know", "Not eligible for coverage / N/A", "Not applicable"]),
        ("mh_care_options_known", "Know the care options under employer coverage?",
         ["Yes", "No", "I am not sure", "Not applicable"]),
        ("mh_discussed_by_employer", "Employer formally discussed mental health?", YN_DK_NA),
        ("mh_resources_offered", "Employer offers mental health learning/help resources?", YN_DK_NA),
        ("anonymity_protected", "Anonymity protected when using employer resources?", YN_DK_NA),
        ("medical_coverage_mh", "Medical coverage includes mental health treatment?", "binary"),
        ("know_resources", "Know local/online resources for help?",
         ["Yes, I know several", "I know some", "No, I don't know any", "Not applicable"]),
        ("leave_ease", "How easy is requesting mental-health medical leave?",
         ["Very easy", "Somewhat easy", "Neither easy nor difficult", "Somewhat difficult",
          "Very difficult", "I don't know", "Not applicable"]),
        ("employer_takes_mh_seriously", "Employer takes mental health as seriously as physical?",
         ["Yes", "No", "I don't know", "Not applicable"]),
    ],
    "Attitudes & experiences": [
        ("mh_disorder_negative_consequence", "Discussing a mental health disorder with employer has negative consequences?", YNM_NA),
        ("phys_disorder_negative_consequence", "Discussing a physical health issue with employer has negative consequences?", YNM_NA),
        ("comfortable_coworkers", "Comfortable discussing it with coworkers?", YNM_NA),
        ("comfortable_supervisor", "Comfortable discussing it with your supervisor?", YNM_NA),
        ("observed_negative_consequences", "Seen negative consequences for open coworkers?", YN_NA),
        ("bring_up_physical_interview", "Would raise a physical health issue in an interview?", YNM),
        ("bring_up_mh_interview", "Would raise a mental health issue in an interview?", YNM),
        ("career_hurt_belief", "Being identified would hurt your career?",
         ["Yes, it has", "Yes, I think it would", "Maybe", "No, I don't think it would", "No, it has not"]),
        ("coworker_negative_view_belief", "Coworkers would view you more negatively?",
         ["Yes, they do", "Yes, I think they would", "Maybe", "No, I don't think they would", "No, they do not"]),
        ("observed_bad_response", "Unsupportive response to mental health at work?",
         ["Yes, I experienced", "Yes, I observed", "Maybe/Not sure", "No", "Not applicable"]),
    ],
}
ALL_Q = {q[0]: q for g in QUESTIONS.values() for q in g}

SHORT = {
    "diagnosed_by_professional": "Professional diagnosis", "past_mh_disorder": "Past mental health disorder",
    "sought_treatment": "Sought treatment before", "interferes_when_treated": "Work interference (treated)",
    "productivity_affected": "Productivity affected", "willing_share_family": "Openness with family/friends",
    "self_employed": "Self-employed", "company_size": "Company size", "tech_company": "Tech employer",
    "remote_work": "Remote work", "has_previous_employers": "Has previous employers",
    "mh_benefits": "Employer MH benefits", "mh_care_options_known": "Knows care options",
    "mh_discussed_by_employer": "Employer discussed MH", "mh_resources_offered": "Employer resources",
    "anonymity_protected": "Anonymity protected", "medical_coverage_mh": "Medical coverage for MH",
    "know_resources": "Knows help resources", "leave_ease": "Ease of medical leave",
    "employer_takes_mh_seriously": "Employer takes MH seriously",
    "mh_disorder_negative_consequence": "Fears consequences (MH)",
    "phys_disorder_negative_consequence": "Fears consequences (physical)",
    "comfortable_coworkers": "Comfort with coworkers", "comfortable_supervisor": "Comfort with supervisor",
    "observed_negative_consequences": "Saw negative consequences",
    "bring_up_physical_interview": "Raise physical issue in interview",
    "bring_up_mh_interview": "Raise MH in interview", "career_hurt_belief": "Career-hurt belief",
    "coworker_negative_view_belief": "Fears coworker judgement",
    "observed_bad_response": "Unsupportive response at work",
    "age": "Age", "gender": "Gender", "country_live": "Country", "work_position": "Work role",
}
DEMOGRAPHIC = {"age", "gender", "country_live", "work_position"}

# Workplace/support questions that the "what-if" explorer is allowed to vary
LEVER_KEYS = ["mh_benefits", "mh_care_options_known", "mh_discussed_by_employer", "mh_resources_offered",
              "anonymity_protected", "medical_coverage_mh", "know_resources", "leave_ease",
              "employer_takes_mh_seriously", "comfortable_coworkers", "comfortable_supervisor",
              "mh_disorder_negative_consequence", "career_hurt_belief", "coworker_negative_view_belief"]

BASES = sorted(list(ALL_Q) + ["gender", "country_live", "work_position"], key=len, reverse=True)


def base_of(col):
    if col == "age":
        return "age"
    for b in BASES:
        if col == b or col.startswith(b + "_"):
            return b
    return col


def options_for(key):
    opts = ALL_Q[key][2]
    return ["Yes", "No"] if opts == "binary" else opts


INTERVENTIONS = {
    "diagnosed_by_professional_No": "You have not been formally diagnosed. Consider an evaluation with a licensed mental health professional to get clarity on how you are doing.",
    "diagnosed_by_professional_Yes": "You have a formal diagnosis. Staying consistent with your treatment plan and check-ins matters, even when you feel stable.",
    "past_mh_disorder_Yes": "You have a history of a mental health disorder. Regular check-ins with a professional, even in stable periods, can help catch early warning signs.",
    "past_mh_disorder_Maybe": "You are unsure whether you have had a mental health disorder before. A short conversation with a counselor can help you make sense of past experiences.",
    "sought_treatment": "You have sought treatment before. Continuing or returning to that support can help maintain your wellbeing.",
    "interferes_when_treated_Often": "Your condition often interferes with work even when treated. Consider discussing treatment adjustments with your provider and workplace accommodations with HR.",
    "interferes_when_treated_Sometimes": "Your condition sometimes interferes with work. Coping strategies or periodic check-ins with a counselor may help.",
    "productivity_affected_Yes": "You feel your productivity is affected. Talking with a professional and adjusting workload or routines can help.",
    "comfortable_supervisor_No": "You may not feel comfortable talking to your supervisor about mental health. An anonymous Employee Assistance Program (EAP) can be an alternative channel.",
    "comfortable_coworkers_No": "You may feel isolated from coworkers on this topic. Peer support groups, in person or online, can offer a more comfortable space.",
    "mh_benefits_No": "Your employer may not offer mental health benefits. Look into community clinics, sliding-scale therapy, or free helplines.",
    "mh_benefits_I don't know": "You are unsure what mental health benefits exist. Ask HR; many employers offer benefits staff are not aware of.",
    "mh_care_options_known_No": "You do not know the care options your coverage includes. HR or your insurer can walk you through them.",
    "mh_care_options_known_I am not sure": "You are not sure what care options your coverage includes. HR or your insurer can walk you through them.",
    "know_resources_No, I don't know any": "You do not know where to find help. Government helplines, campus or community counseling, and online directories are good starting points.",
    "employer_takes_mh_seriously_No": "Your workplace may not prioritise mental health. External support (helplines, community clinics) may be more reliable for now.",
    "career_hurt_belief_Yes, it has": "You feel your career has been affected by stigma. Career counselling alongside mental health support may help with both.",
    "career_hurt_belief_Yes, I think it would": "You worry that disclosure could hurt your career. A confidential professional can support you without involving your workplace.",
    "observed_bad_response_Yes, I experienced": "You have experienced an unsupportive response before. Support from outside your workplace, such as a therapist or support group, may help.",
    "willing_share_family_Not open at all": "You may be hesitant to share with family. A therapist offers a confidential space that does not require family disclosure.",
    "willing_share_family_Somewhat not open": "You may be hesitant to share with family. A therapist offers a confidential space that does not require family disclosure.",
    "leave_ease_Very difficult": "Taking medical leave looks hard at your workplace. Ask HR about the formal policy, or get a professional's note on what accommodations are possible.",
    "mh_resources_offered_No": "Your employer offers no mental health resources. Public helplines and community services can fill that gap.",
}
TIER_MESSAGE = {
    "Low": "Your answers point to a lower risk profile. Keeping up healthy routines, sleep, and social connection is a good ongoing practice.",
    "Medium": "Your answers point to a moderate risk profile. Consider proactive steps such as periodic self-check-ins, guided journaling, or a preventive conversation with a counselor.",
    "High": "Your answers point to an elevated risk profile. We recommend speaking with a licensed mental health professional for a proper evaluation.",
}

# ---------------------------------------------------------------------
# Theme-aware, high-contrast palette (all colours >= 5.4:1 against the page background)
# ---------------------------------------------------------------------
def theme_type():
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return None


_theme = theme_type()
if _theme is None and not st.session_state.get("theme_retry"):
    st.session_state["theme_retry"] = True      # the browser reports its theme after the first run
    st.rerun()

if _theme == "light":
    PAL = {"tier": {"Low": "#1B7A3D", "Medium": "#B54708", "High": "#B71C1C"},
           "on_bar": "#FFFFFF", "pos": "#B71C1C", "neg": "#1565C0", "text": "#1A1F27"}
else:
    PAL = {"tier": {"Low": "#4ADE80", "Medium": "#FFB347", "High": "#FF6B6B"},
           "on_bar": "#0B0F14", "pos": "#FF6B6B", "neg": "#6CB4FF", "text": "#F2F5F9"}
TIER_COLOR = PAL["tier"]


def callout(html, color, alpha=0.16):
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;border-left:6px solid {color};"
        f"background:rgba({r},{g},{b},{alpha});line-height:1.55'>{html}</div>",
        unsafe_allow_html=True)


def show_fig(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=PAL["text"], size=14),
                      legend=dict(font=dict(color=PAL["text"], size=14)))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.25)", zerolinecolor="rgba(128,128,128,0.6)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
    st.plotly_chart(fig, width="stretch")


DISCLAIMER = (
    "**Research prototype, not a diagnostic or crisis tool.** The prediction comes from a model trained on the "
    "OSMI 2016 workplace survey (about 71% accuracy on held-out data) and cannot diagnose anyone. "
    "If you are struggling or in crisis, please contact a licensed professional, a local helpline "
    "(in India, Tele-MANAS: 14416), or your local emergency number."
)

DISCLAIMER_HTML = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", DISCLAIMER)

ROLE_OPTIONS = sorted({r for f in FEATURES if f.startswith("work_position_")
                       for r in f[len("work_position_"):].split("|")})
COUNTRIES = sorted(f[len("country_live_"):] for f in FEATURES if f.startswith("country_live_"))


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------
def best_role_column(selected):
    selected = set(selected)
    if not selected:
        return None
    best, best_score = None, 0.0
    for f in FEATURES:
        if f.startswith("work_position_"):
            roles = set(f[len("work_position_"):].split("|"))
            score = len(roles & selected) / len(roles | selected)
            if score > best_score:
                best, best_score = f, score
    return best


def row_values(answers):
    row = dict.fromkeys(FEATURES, 0.0)
    for key, value in answers.items():
        if key == "age":
            row["age"] = float(value)
        elif key == "work_position":
            col = best_role_column(value)
            if col:
                row[col] = 1.0
        elif key in BINARY_KEYS:
            row[key] = 1.0 if value == "Yes" else 0.0
        else:
            col = f"{key}_{value}"
            if col in row:
                row[col] = 1.0
    return row


def to_frame(list_of_answers):
    return pd.DataFrame([row_values(a) for a in list_of_answers])[FEATURES]


def class_shap(x, class_idx):
    sv = explainer.shap_values(x)
    if isinstance(sv, list):
        return np.asarray(sv[class_idx])[0]
    return np.asarray(sv)[0, :, class_idx]


def answer_text(answers, base):
    v = answers.get(base)
    if isinstance(v, list):
        return ", ".join(v) if v else "none selected"
    return str(v)


# ---------------------------------------------------------------------
# Session state: defaults + presets
# ---------------------------------------------------------------------
DEFAULTS = {
    "age": 30, "gender": "Male", "country_live": "United States of America", "work_position": [],
}
for key, (_, _, opts) in ALL_Q.items():
    DEFAULTS[key] = "No" if opts == "binary" else options_for(key)[0]
DEFAULTS.update({"diagnosed_by_professional": "No", "past_mh_disorder": "No",
                 "productivity_affected": "No", "willing_share_family": "Neutral",
                 "interferes_when_treated": "Not applicable to me", "has_previous_employers": "Yes",
                 "tech_company": "Yes", "medical_coverage_mh": "Yes"})

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    data = PRESETS.get(name)
    if data:
        for k, v in data.items():
            st.session_state[k] = v


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------
# Sidebar: live inputs
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Your answers")
    st.caption("The results on the right update instantly as you change anything.")

    if PRESETS:
        st.write("**Load an example profile**")
        c1, c2, c3 = st.columns(3)
        c1.button("Low", on_click=apply_preset, args=("Low",), width="stretch")
        c2.button("Medium", on_click=apply_preset, args=("Medium",), width="stretch")
        c3.button("High", on_click=apply_preset, args=("High",), width="stretch")
    st.button("Reset all answers", on_click=reset_all, width="stretch")
    st.divider()

    with st.expander("About you", expanded=False):
        st.slider("Age", 18, 75, key="age")
        st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
        st.selectbox("Country", COUNTRIES, key="country_live")
        st.multiselect("Work role(s)", ROLE_OPTIONS, key="work_position")

    for group, qs in QUESTIONS.items():
        with st.expander(group, expanded=(group == "Mental health history")):
            for key, label, opts in qs:
                if opts == "binary":
                    st.radio(label, ["Yes", "No"], horizontal=True, key=key)
                else:
                    st.selectbox(label, opts, key=key)

answers = {k: st.session_state[k] for k in DEFAULTS}

# ---------------------------------------------------------------------
# Predict (live)
# ---------------------------------------------------------------------
x = to_frame([answers])
proba = model.predict_proba(x)[0]
pred_idx = int(np.argmax(proba))
tier = CLASSES[pred_idx]
P = {c: float(proba[CLASSES.index(c)]) for c in ["Low", "Medium", "High"]}

st.title("Explainable Mental Health Risk Explorer")
st.caption("Early risk prediction with the reasons behind it, personalised next steps, and a what-if explorer.")

tab_res, tab_why, tab_tips, tab_what = st.tabs(
    ["Result", "Why this result", "Suggestions", "What-if explorer"])

# ---------------- Result ----------------
with tab_res:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown(
            f"<div class='risk-card' style='background:{TIER_COLOR[tier]}29;border:2px solid {TIER_COLOR[tier]}'>"
            f"<div class='lbl'>Estimated risk level</div>"
            f"<div class='val' style='color:{TIER_COLOR[tier]}'>{tier}</div>"
            f"<div class='lbl'>{P[tier]:.0%} model confidence</div></div>",
            unsafe_allow_html=True)
        st.write("")
        callout(TIER_MESSAGE[tier], TIER_COLOR[tier])
    with c2:
        fig = go.Figure()
        for cls in ["Low", "Medium", "High"]:
            fig.add_trace(go.Bar(
                y=["Probability"], x=[P[cls]], name=f"{cls} {P[cls]:.0%}", orientation="h",
                marker_color=TIER_COLOR[cls], text=(f"{cls} {P[cls]:.0%}" if P[cls] >= 0.12 else ""),
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color=PAL["on_bar"], size=15),
                hovertemplate=f"{cls}: %{{x:.1%}}<extra></extra>"))
        fig.update_layout(barmode="stack", height=170, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(visible=False),
                          legend=dict(orientation="h", y=-0.35))
        show_fig(fig)
        st.caption("How the model splits its confidence across the three tiers. Hover for exact values.")
    callout(DISCLAIMER_HTML, "#E0A100", 0.14)

# ---------------- Why ----------------
contrib = pd.DataFrame({"feature": FEATURES, "shap": class_shap(x, pred_idx), "value": x.iloc[0].values})
contrib["base"] = contrib["feature"].map(base_of)

with tab_why:
    st.subheader(f"What pushed the model toward “{tier}”")
    grouped = contrib.groupby("base")["shap"].sum()
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(10)[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=[SHORT.get(b, b) for b in top.index], orientation="h",
        marker_color=[PAL["pos"] if v > 0 else PAL["neg"] for v in top.values],
        customdata=[answer_text(answers, b) for b in top.index],
        hovertemplate="<b>%{y}</b><br>Your answer: %{customdata}<br>Effect: %{x:+.3f}<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title=f"Push toward (+) / away from (−) “{tier}”  (SHAP value)")
    show_fig(fig)
    st.caption("Red bars pushed the model toward this result; blue bars pushed away. Hover a bar to see your answer. "
               "Demographic items (age, gender, country, role) are shown for transparency only and never drive suggestions.")

# ---------------- Suggestions ----------------
with tab_tips:
    st.subheader("Personalised suggestions")
    if tier == "Low":
        st.success("Nothing in your answers calls for specific action right now. If that changes, or you simply "
                   "want to talk to someone, a counselor is always a good option.")
    else:
        active = contrib[(contrib["value"] == 1) & (contrib["shap"] > 0)
                         & (~contrib["base"].isin(DEMOGRAPHIC))].sort_values("shap", ascending=False)
        shown = 0
        for _, r in active.iterrows():
            tip = INTERVENTIONS.get(r["feature"])
            if tip:
                st.markdown(f"<div class='tip'><b>{SHORT.get(r['base'], r['base'])}</b><br>{tip}</div>",
                            unsafe_allow_html=True)
                shown += 1
            if shown == 5:
                break
        if shown == 0:
            st.write("No specific factor-based suggestions matched, so the general guidance applies.")
    st.caption(DISCLAIMER)

# ---------------- What-if ----------------
with tab_what:
    st.subheader("What-if explorer")
    st.write("Change one workplace or support factor and see how the model's estimate would move. "
             "This shows the **model's sensitivity**, not proof that the change would cause a real-world difference.")

    qkey = st.selectbox("Pick a factor to change", LEVER_KEYS, format_func=lambda k: SHORT[k], key="whatif_q")
    current = answers[qkey]
    new_val = st.selectbox(f"What if the answer were…  (currently: {current})",
                           [o for o in options_for(qkey) if o != current] or [current], key=f"whatif_v_{qkey}")

    alt = dict(answers)
    alt[qkey] = new_val
    p_new = model.predict_proba(to_frame([alt]))[0]
    Pn = {c: float(p_new[CLASSES.index(c)]) for c in ["Low", "Medium", "High"]}

    m1, m2, m3 = st.columns(3)
    for col, cls in zip((m1, m2, m3), ["Low", "Medium", "High"]):
        col.metric(f"{cls} probability", f"{Pn[cls]:.0%}", f"{(Pn[cls] - P[cls]) * 100:+.1f} pts",
                   delta_color="inverse" if cls == "High" else ("normal" if cls == "Low" else "off"))
    new_tier = CLASSES[int(np.argmax(p_new))]
    if new_tier != tier:
        st.success(f"Predicted tier would change from **{tier}** to **{new_tier}**.")
    else:
        st.caption(f"Predicted tier would stay **{tier}**.")

    st.divider()
    st.write("**Biggest levers for you** — the single answer change that most lowers the High-risk probability:")
    rows, meta = [], []
    for k in LEVER_KEYS:
        for opt in options_for(k):
            if opt != answers[k]:
                a2 = dict(answers)
                a2[k] = opt
                rows.append(a2)
                meta.append((k, opt))
    probs = model.predict_proba(to_frame(rows))
    hi = CLASSES.index("High")
    best = {}
    for (k, opt), p in zip(meta, probs):
        delta = (p[hi] - P["High"]) * 100
        if k not in best or delta < best[k][1]:
            best[k] = (opt, delta)
    lev = pd.DataFrame([(SHORT[k], v[0], v[1]) for k, v in best.items()],
                       columns=["Factor", "Best alternative answer", "Change in High-risk probability (pts)"])
    lev = lev.sort_values("Change in High-risk probability (pts)").head(6)
    fig = go.Figure(go.Bar(
        x=lev["Change in High-risk probability (pts)"][::-1], y=lev["Factor"][::-1], orientation="h",
        marker_color=PAL["neg"], customdata=lev["Best alternative answer"][::-1],
        hovertemplate="<b>%{y}</b><br>If answered: %{customdata}<br>Change: %{x:+.1f} pts<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title="Change in High-risk probability (percentage points)")
    show_fig(fig)
    st.dataframe(lev, hide_index=True, width="stretch")
