import streamlit as st
import numpy as np
import tensorflow as tf
from auth import verify_login
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os
from datetime import datetime

from db import (
    init_db, create_batch, create_sample, record_human_decision,
    record_size_assessment, get_batch_summary, get_all_batches, get_batch_samples
)
from grading import load_rules, apply_grading
from translations import t
from report_generator import generate_batch_report

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_multiclass.keras"
CLASS_NAMES = ["damaged", "healthy", "rotten"]
CONFIDENCE_THRESHOLD = 0.60
UPLOAD_DIR = "uploads"
GRADING_RULES = load_rules("grading_rules.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()

if "logged_in_officer" not in st.session_state:
    st.session_state.logged_in_officer = None
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "current_batch" not in st.session_state:
    st.session_state.current_batch = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

lang = st.session_state.lang

st.set_page_config(page_title="AgroNex", page_icon="🧅", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #FAF9F6;
    }
    section[data-testid="stSidebar"] {
        background-color: #1B3A2B;
    }
    section[data-testid="stSidebar"] * {
        color: #F1EFE6 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 6px 4px;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.08);
    }

    h1, h2, h3 {
        color: #22331F;
        font-weight: 700;
    }

    div.stButton > button {
        background-color: #C1502E;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.55rem 1.3rem;
        font-weight: 600;
        transition: 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #A13F22;
        color: white;
    }

    [data-testid="stMetricValue"] {
        color: #C1502E;
        font-weight: 800;
    }
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E3E8DE;
        border-radius: 14px;
        padding: 16px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    div[data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #E3E8DE !important;
        border-radius: 14px !important;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 22px;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stAlert {
        border-radius: 10px;
    }

    .agronex-badge {
        display: inline-block;
        background-color: #F4E3B2;
        color: #8A6A1B;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }

    hr {
        margin: 2rem 0 !important;
        border-color: #E3E8DE !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------
if st.session_state.logged_in_officer is None:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🧅 AgroNex")
        st.caption("AI-assisted onion procurement inspection")
        with st.container(border=True):
            st.subheader("Officer Login")
            officer_id = st.text_input("Officer ID")
            password = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                name = verify_login(officer_id, password)
                if name:
                    st.session_state.logged_in_officer = {"id": officer_id, "name": name}
                    st.rerun()
                else:
                    st.error("Invalid Officer ID or password.")
    st.stop()

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

model = load_model()
class_key_map = {"healthy": "healthy", "damaged": "damaged", "rotten": "rotten"}

# ---------------------------------------------------------------------------
# Sidebar: branding, nav, language, logout
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧅 AgroNex")
    st.caption("AI-Powered Onion Quality Assessment")
    st.markdown(f"**{st.session_state.logged_in_officer['name']}**")
    st.markdown("<hr style='margin:12px 0; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)

    st.session_state.page = st.radio(
        "Navigate",
        ["Dashboard", "New Inspection", "Batch History", "Reports", "Settings"],
        index=["Dashboard", "New Inspection", "Batch History", "Reports", "Settings"].index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:12px 0; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
    lang_display = {"English": "en", "தமிழ் (Tamil)": "ta", "हिन्दी (Hindi)": "hi"}
    lang_choice = st.selectbox(
        "Language",
        list(lang_display.keys()),
        index=list(lang_display.values()).index(st.session_state.lang),
    )
    st.session_state.lang = lang_display[lang_choice]
    lang = st.session_state.lang

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in_officer = None
        st.rerun()

all_batches = get_all_batches()
batch_options = [b["batch_id"] for b in all_batches]

# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
if st.session_state.page == "Dashboard":
    st.markdown('<span class="agronex-badge">DASHBOARD</span>', unsafe_allow_html=True)
    st.title("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Batches", len(all_batches))
    total_samples_all = sum(get_batch_summary(b["batch_id"])["total_samples"] for b in all_batches) if all_batches else 0
    c2.metric("Samples Inspected", total_samples_all)
    c3.metric("Current Batch", st.session_state.current_batch or "None selected")

    st.markdown("### Create a New Batch")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            supplier = st.text_input(t("supplier", lang))
            variety = st.text_input(t("variety", lang))
            unit = st.selectbox(t("unit", lang), ["kg", "quintal", "tonnes"])
        with col2:
            center = st.text_input(t("centre", lang))
            quantity = st.number_input(t("quantity", lang), min_value=0.0, value=0.0)
            intended_use = st.text_input(t("intended_use", lang), value="General")

        if st.button(t("create_batch_btn", lang)):
            if supplier and center:
                batch_id = create_batch(supplier, center, variety, quantity, unit, intended_use)
                st.session_state.current_batch = batch_id
                st.success(f"{t('batch_created', lang)}: {batch_id}")
                st.rerun()
            else:
                st.error(t("supplier_centre_required", lang))

    st.markdown("### Select Active Batch")
    if batch_options:
        selected = st.selectbox(
            t("active_batch", lang),
            batch_options,
            index=batch_options.index(st.session_state.current_batch) if st.session_state.current_batch in batch_options else 0,
        )
        st.session_state.current_batch = selected

        summary = get_batch_summary(selected)
        st.markdown(f"#### {t('batch_summary', lang)} — {selected}")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric(t("healthy", lang), f"{summary['healthy']} ({summary['healthy_pct']}%)")
        cc2.metric(t("damaged", lang), f"{summary['damaged']} ({summary['damaged_pct']}%)")
        cc3.metric(t("rotten", lang), f"{summary['rotten']} ({summary['rotten_pct']}%)")
        cc4.metric(t("human_corrections", lang), summary["human_corrections"])
    else:
        st.info("No batches yet — create one above to get started.")

# ===========================================================================
# PAGE: New Inspection
# ===========================================================================
elif st.session_state.page == "New Inspection":
    st.markdown('<span class="agronex-badge">INSPECTION</span>', unsafe_allow_html=True)
    st.title(t("inspect_sample", lang))

    if not st.session_state.current_batch:
        st.warning(t("select_batch_prompt", lang) + " (go to Dashboard first)")
        st.stop()

    st.caption(f"{t('active_batch', lang)}: **{st.session_state.current_batch}**")

    with st.container(border=True):
        tab1, tab3 = st.tabs([t("camera_tab", lang), "📦 Batch Upload"])
        img_file = None
        with tab1:
            camera_img = st.camera_input(t("take_photo", lang))
            if camera_img is not None:
                img_file = camera_img
        
        with tab3:
            st.caption("Upload multiple onion photos at once — each is predicted and saved as its own sample.")
            batch_files = st.file_uploader(
                "Select multiple images", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True, key="batch_uploader",
            )
            if batch_files:
                processed_key = f"batch_processed_{st.session_state.current_batch}"
                if processed_key not in st.session_state:
                    st.session_state[processed_key] = {}

                new_files = [f for f in batch_files if f.name not in st.session_state[processed_key]]
                if new_files:
                    progress = st.progress(0, text=f"Processing {len(new_files)} image(s)...")
                    for i, bf in enumerate(new_files):
                        img_b = Image.open(bf).convert("RGB")
                        img_resized_b = img_b.resize(IMG_SIZE)
                        img_array_b = np.expand_dims(np.array(img_resized_b), axis=0)
                        probs_b = model.predict(img_array_b, verbose=0)[0]
                        idx_b = np.argmax(probs_b)
                        pred_b = CLASS_NAMES[idx_b]
                        conf_b = float(probs_b[idx_b])
                        grade_b = apply_grading(pred_b, conf_b, GRADING_RULES)

                        fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{bf.name}"
                        path_b = os.path.join(UPLOAD_DIR, fname)
                        img_b.save(path_b)

                        sample_id_b = create_sample(
                            st.session_state.current_batch, path_b, pred_b, conf_b,
                            grade=grade_b["grade"], confidence_level=grade_b["confidence_level"],
                        )
                        st.session_state[processed_key][bf.name] = {
                            "sample_id": sample_id_b, "prediction": pred_b,
                            "confidence": conf_b, "grade": grade_b["grade"], "thumb": img_b,
                        }
                        progress.progress((i + 1) / len(new_files), text=f"Processed {i + 1}/{len(new_files)}")
                    progress.empty()
                    st.success(f"Added {len(new_files)} sample(s) to {st.session_state.current_batch}.")

                st.markdown("##### Batch upload results")
                for fname, r in st.session_state[processed_key].items():
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns([1, 2, 2])
                        with col_a:
                            st.image(r["thumb"], width=90)
                        with col_b:
                            st.write(f"**{t(class_key_map[r['prediction']], lang).upper()}**")
                            st.caption(f"{r['confidence']*100:.1f}% · {r['grade']}")
                        with col_c:
                            corr = st.selectbox(
                                t("correct_to", lang), CLASS_NAMES,
                                format_func=lambda c: t(class_key_map[c], lang),
                                index=CLASS_NAMES.index(r["prediction"]),
                                key=f"batch_correct_{r['sample_id']}",
                                label_visibility="collapsed",
                            )
                            if st.button(t("submit_correction", lang), key=f"batch_correct_btn_{r['sample_id']}"):
                                record_human_decision(r["sample_id"], corr)
                                st.success(t("corrected_to", lang) + f" {t(class_key_map[corr], lang)}")

                if st.button("✅ Confirm all AI results as correct", key="confirm_all_batch"):
                    for r in st.session_state[processed_key].values():
                        record_human_decision(r["sample_id"], r["prediction"])
                    st.success("All samples in this batch confirmed.")

    if img_file is not None:
        col_img, col_result = st.columns([1, 1.3])
        img = Image.open(img_file).convert("RGB")
        with col_img:
            st.image(img, caption=t("sample_image", lang), use_container_width=True)

        img_resized = img.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        probs = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])
        grade_info = apply_grading(predicted_class, confidence, GRADING_RULES)

        with col_result:
            with st.container(border=True):
                st.markdown(f"#### {t('ai_prediction', lang)}")
                st.markdown(f"## {t(class_key_map[predicted_class], lang).upper()}")
                st.progress(confidence, text=f"{t('confidence', lang)}: {confidence * 100:.1f}%")
                st.markdown(f"**Grade:** {grade_info['grade']}")
                if grade_info["manual_review_required"]:
                    st.caption("⚠️ Flagged for manual review")
                for i, name in enumerate(CLASS_NAMES):
                    st.caption(f"{t(class_key_map[name], lang)}: {probs[i] * 100:.1f}%")
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(t("low_confidence", lang))
                else:
                    st.success(t("high_confidence", lang))

        if "last_sample_id" not in st.session_state or st.session_state.get("last_img_name") != img_file.name:
            img_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{img_file.name}"
            img_path = os.path.join(UPLOAD_DIR, img_filename)
            img.save(img_path)
            sample_id = create_sample(
                st.session_state.current_batch, img_path, predicted_class, confidence,
                grade=grade_info["grade"],
                confidence_level=grade_info["confidence_level"],
            )
            st.session_state.last_sample_id = sample_id
            st.session_state.last_img_name = img_file.name
            st.session_state.last_prediction = predicted_class

        sample_id = st.session_state.last_sample_id

        st.markdown(f"#### {t('human_verification', lang)}")
        st.caption("Both the AI result and your assessment are recorded — your input doesn't erase the AI's original prediction, it's tracked alongside it for transparency.")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("confirm_ai", lang), key=f"confirm_{sample_id}", use_container_width=True):
                    record_human_decision(sample_id, predicted_class)
                    st.success(t("confirmed", lang))
            with col2:
                correction = st.selectbox(
                    t("correct_to", lang), CLASS_NAMES,
                    format_func=lambda c: t(class_key_map[c], lang),
                    key=f"correct_select_{sample_id}",
                )
                if st.button(t("submit_correction", lang), key=f"correct_btn_{sample_id}", use_container_width=True):
                    reason = st.session_state.get(f"reason_{sample_id}", "")
                    record_human_decision(sample_id, correction, correction_reason=reason)
                    st.success(f"{t('corrected_to', lang)} {t(class_key_map[correction], lang)}.")
            st.text_input(t("correction_reason", lang), key=f"reason_{sample_id}")

        st.markdown(f"#### {t('size_assessment', lang)}")
        with st.container(border=True):
            size_labels = [t("size_normal", lang), t("size_undersized", lang), t("size_not_assessed", lang)]
            size_internal = ["Normal", "Undersized", "Not Assessed"]
            size_choice_idx = st.radio(
                t("size_label", lang), range(3),
                format_func=lambda i: size_labels[i],
                index=2, key=f"size_{sample_id}", horizontal=True,
            )
            if st.button(t("save_size", lang), key=f"size_btn_{sample_id}"):
                record_size_assessment(sample_id, size_internal[size_choice_idx])
                st.success(f"{t('size_recorded', lang)}: {size_labels[size_choice_idx]}")

    st.divider()
    st.markdown("#### Batch Progress")
    live_summary = get_batch_summary(st.session_state.current_batch)
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric(t("total_samples", lang), live_summary["total_samples"])
    lc2.metric(t("healthy", lang), f"{live_summary['healthy_pct']}%")
    lc3.metric(t("damaged", lang), f"{live_summary['damaged_pct']}%")
    lc4.metric(t("rotten", lang), f"{live_summary['rotten_pct']}%")

# ===========================================================================
# PAGE: Batch History
# ===========================================================================
elif st.session_state.page == "Batch History":
    st.markdown('<span class="agronex-badge">HISTORY</span>', unsafe_allow_html=True)
    st.title("Batch History")

    if not all_batches:
        st.info("No batches yet.")
    else:
        for b in all_batches:
            summary = get_batch_summary(b["batch_id"])
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{b['batch_id']}** — {b.get('supplier_name', '')}")
                    st.caption(f"{b.get('procurement_centre', '')} · {b.get('date', '')} · {b.get('status', 'Active')}")
                with col2:
                    st.metric(t("total_samples", lang), summary["total_samples"])
                st.write(
                    f"{t('healthy', lang)}: {summary['healthy_pct']}%  ·  "
                    f"{t('damaged', lang)}: {summary['damaged_pct']}%  ·  "
                    f"{t('rotten', lang)}: {summary['rotten_pct']}%  ·  "
                    f"{t('human_corrections', lang)}: {summary['human_corrections']}"
                )
                if st.button("View / Set Active", key=f"view_{b['batch_id']}"):
                    st.session_state.current_batch = b["batch_id"]
                    st.session_state.page = "Dashboard"
                    st.rerun()

# ===========================================================================
# PAGE: Reports
# ===========================================================================
elif st.session_state.page == "Reports":
    st.markdown('<span class="agronex-badge">REPORTS</span>', unsafe_allow_html=True)
    st.title(t("generate_report", lang))

    if not st.session_state.current_batch:
        st.warning(t("select_batch_prompt", lang) + " (go to Dashboard first)")
        st.stop()

    st.caption(f"{t('active_batch', lang)}: **{st.session_state.current_batch}**")
    with st.container(border=True):
        if st.button(t("generate_pdf_btn", lang)):
            pdf_path = generate_batch_report(st.session_state.current_batch)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    t("download_pdf", lang),
                    data=f,
                    file_name=f"{st.session_state.current_batch}_report.pdf",
                    mime="application/pdf",
                )
            st.success(f"{t('report_generated', lang)}: {pdf_path}")

# ===========================================================================
# PAGE: Settings
# ===========================================================================
elif st.session_state.page == "Settings":
    st.markdown('<span class="agronex-badge">SETTINGS</span>', unsafe_allow_html=True)
    st.title("Settings")
    st.write(f"Logged in as **{st.session_state.logged_in_officer['name']}**")
    st.write(f"Officer ID: `{st.session_state.logged_in_officer['id']}`")