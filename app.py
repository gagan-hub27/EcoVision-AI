import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path

from config import APP_NAME, MODEL_PATH, CONFIDENCE_THRESHOLD
from database.mongo import save_detection, fetch_detections, get_db_status
from modules.detector import WasteDetector
from modules.waste_guide import WASTE_GUIDE, get_guidance

st.set_page_config(
    page_title=f"{APP_NAME} | Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = Path(__file__).parent / "assets" / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    return WasteDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)


ai = load_detector()
dbok, db_message = get_db_status()

PAGES = ["Overview", "Detect Waste", "Waste Guide", "History", "Analytics", "About"]

# Streamlit navigation state also allows the dashboard shortcuts to work.
if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Overview"


def metric(label, value, helper, icon):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top"><span>{icon}</span><span>{label}</span></div>
          <div class="metric-value">{value}</div>
          <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label, good):
    cls = "status-good" if good else "status-warn"
    return f'<span class="status-pill {cls}"><span class="status-dot"></span>{label}</span>'


def go_to(page):
    st.session_state.page_nav = page


def render_guide(category):
    guide = get_guidance(category)
    st.markdown(f"### {guide['emoji']} {guide['title']}")
    st.caption(f"{guide['group']}  ·  {guide['handling']}")
    st.write(guide["description"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Recommended steps**")
        for step in guide["steps"]:
            st.markdown(f"- {step}")
    with right:
        st.markdown("**Avoid**")
        for item in guide["avoid"]:
            st.markdown(f"- {item}")
    st.info(guide["note"])


with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">♻</div>
          <div><div class="brand-name">EcoVision <span>AI</span></div>
          <div class="brand-subtitle">SMART WASTE MANAGEMENT</div></div>
        </div>
        <div class="sidebar-label">MAIN MENU</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        PAGES,
        key="page_nav",
        label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown(status_pill("Database connected" if dbok else "Database offline", dbok), unsafe_allow_html=True)
    st.markdown(status_pill("AI model ready" if ai.available else "AI model not installed", ai.available), unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-footer">MCA PROJECT · DEEP LEARNING<br/>EcoVision AI v1.0</div>',
        unsafe_allow_html=True,
    )


if page == "Overview":
    records = pd.DataFrame(fetch_detections(5000))
    total_scans = len(records)
    categories_seen = records["category"].nunique() if not records.empty and "category" in records else 0

    st.markdown('<div class="page-kicker">WASTE INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.title("A cleaner tomorrow starts with smarter sorting.")
    st.markdown(
        '<p class="page-intro">Recognise waste, understand its impact, and make better disposal decisions — all in one place.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="welcome-panel">
          <div>
            <div class="welcome-eyebrow">WELCOME TO ECOVISION AI</div>
            <div class="welcome-title">Turn everyday waste into<br/>informed action.</div>
            <div class="welcome-copy">Upload an image to identify supported waste items and get practical guidance for reuse, recycling and responsible disposal.</div>
          </div>
          <div class="welcome-art"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="leaf-art">♻</div><div class="art-caption">RETHINK · REDUCE · RECYCLE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Project at a glance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric("TOTAL SCANS", f"{total_scans:,}", "Saved detection records", "⌁")
    with c2:
        metric("WASTE CATEGORIES", f"{len(WASTE_GUIDE)}", "Guidance categories available", "♻")
    with c3:
        metric("AI RECOGNITION", "Active" if ai.available else "Setup required", "YOLO deep-learning model", "◎")
    with c4:
        metric("DATABASE", "Connected" if dbok else "Not connected", "MongoDB Atlas storage", "▤")

    st.markdown('<div class="section-heading">Explore the workspace</div>', unsafe_allow_html=True)
    cards = st.columns(3, gap="medium")
    card_data = [
        ("📷", "Waste detection", "Analyse an image and view supported waste predictions.", "Detect Waste", "Open detector"),
        ("🌱", "Waste guide", "Find practical sorting, recycling and disposal advice.", "Waste Guide", "Explore guide"),
        ("📊", "Activity analytics", "Review saved records and understand category trends.", "Analytics", "View analytics"),
    ]
    for col, (icon, title, description, target, action) in zip(cards, card_data):
        with col:
            st.markdown(
                f'<div class="workspace-card"><div class="workspace-icon">{icon}</div><div class="workspace-title">{title}</div><div class="workspace-copy">{description}</div></div>',
                unsafe_allow_html=True,
            )
            st.button(action, key=f"goto_{target}", use_container_width=True, on_click=go_to, args=(target,))

    if not ai.available:
        st.markdown(
            """
            <div class="notice-panel">
              <strong>AI model setup pending</strong>
              <p>The dashboard is ready. To enable automatic image recognition, train a YOLO model and place the resulting <code>best.pt</code> file inside the <code>models</code> folder. Manual review is available in the meantime and is clearly labelled as manual.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if not dbok:
        st.warning("MongoDB Atlas is not connected. Add a valid MONGODB_URI in your local .env file to save records to the cloud.")
    if not records.empty:
        st.markdown('<div class="section-heading">Recent activity</div>', unsafe_allow_html=True)
        display_cols = [c for c in ["item_name", "category", "confidence", "source", "created_at"] if c in records.columns]
        st.dataframe(records[display_cols].head(5), use_container_width=True, hide_index=True)


elif page == "Detect Waste":
    st.markdown('<div class="page-kicker">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Waste detection")
    st.markdown('<p class="page-intro">Upload a clear image to inspect supported waste items and receive disposal guidance.</p>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">For best results, use a well-lit photo with the waste item clearly visible.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="visible")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            if ai.available:
                if st.button("Run AI detection", type="primary", use_container_width=True):
                    with st.spinner("Analysing image with the trained model..."):
                        st.session_state["predictions"] = ai.predict(image)
                        st.session_state["prediction_upload"] = uploaded.name
                if st.session_state.get("prediction_upload") == uploaded.name:
                    predictions = st.session_state.get("predictions", [])
                    if not predictions:
                        st.info("No supported objects were detected. Try a clearer image or a different angle.")
                    for idx, item in enumerate(predictions):
                        guide = get_guidance(item["category"])
                        with st.container(border=True):
                            st.markdown(f"**{item['item_name']}**")
                            st.caption(f"{guide['title']} · Confidence {item['confidence']:.1%}")
                            st.write(guide["description"])
                            st.markdown("**Recommended action**")
                            for step in guide["steps"]:
                                st.markdown(f"- {step}")
                            st.caption(guide["note"])
                            if st.button("Save detection", key=f"save_ai_{idx}"):
                                record = {
                                    **item,
                                    "status": "detected",
                                    "source": "ai",
                                    "recyclable": guide["recyclable"],
                                    "disposal_method": guide["handling"],
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                }
                                ok, message = save_detection(record)
                                st.success("Detection saved." if ok else f"Could not save: {message}")
            else:
                st.info("The trained model is not installed yet. You can test the record-saving workflow using manual review; this does not produce an AI prediction.")
                category = st.selectbox("Select the item category", list(WASTE_GUIDE), format_func=lambda key: WASTE_GUIDE[key]["title"])
                item_name = st.text_input("Item name (optional)", placeholder="e.g. plastic water bottle")
                if st.button("Save manual review", type="primary", use_container_width=True):
                    guide = get_guidance(category)
                    record = {
                        "category": category,
                        "item_name": item_name.strip() or guide["title"],
                        "confidence": None,
                        "status": "manual_review",
                        "source": "manual",
                        "recyclable": guide["recyclable"],
                        "disposal_method": guide["handling"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    ok, message = save_detection(record)
                    st.success("Manual review saved." if ok else f"Could not save: {message}")
    else:
        st.info("Your uploaded image and its analysis will appear here.")


elif page == "Waste Guide":
    st.markdown('<div class="page-kicker">KNOW YOUR WASTE</div>', unsafe_allow_html=True)
    st.title("Waste management guide")
    st.markdown('<p class="page-intro">Explore practical ways to separate, reuse, recycle and dispose of common waste materials.</p>', unsafe_allow_html=True)
    query = st.text_input("Search the guide", placeholder="Try plastic, battery, paper or food scraps...")
    categories = [
        key for key, value in WASTE_GUIDE.items()
        if not query or query.lower() in (key + value["title"] + value["description"]).lower()
    ]
    if not categories:
        st.info("No matching category found. Try another keyword.")
    else:
        cols = st.columns(2, gap="medium")
        for index, category in enumerate(categories):
            guide = WASTE_GUIDE[category]
            with cols[index % 2]:
                with st.container(border=True):
                    with st.expander(f"{guide['emoji']}  {guide['title']}  ·  {guide['group']}"):
                        render_guide(category)


elif page == "History":
    st.markdown('<div class="page-kicker">RECORDS</div>', unsafe_allow_html=True)
    st.title("Detection history")
    st.markdown('<p class="page-intro">A chronological record of AI detections and manually reviewed items.</p>', unsafe_allow_html=True)
    history = pd.DataFrame(fetch_detections(5000))
    if history.empty:
        st.info("No saved records yet. Start by reviewing an item in Waste detection.")
    else:
        st.markdown(f"**{len(history):,} records available**")
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Download history as CSV", history.to_csv(index=False).encode("utf-8"), "ecovision_detection_history.csv", "text/csv")


elif page == "Analytics":
    st.markdown('<div class="page-kicker">INSIGHTS & REPORTING</div>', unsafe_allow_html=True)
    st.title("Waste analytics")
    st.markdown('<p class="page-intro">Understand the distribution of saved waste records across categories.</p>', unsafe_allow_html=True)
    data = pd.DataFrame(fetch_detections(5000))
    if data.empty or "category" not in data:
        st.info("Analytics will appear after you save some detections.")
    else:
        counts = data["category"].value_counts().rename_axis("Category").reset_index(name="Scans")
        total = int(counts["Scans"].sum())
        top = str(counts.iloc[0]["Category"]) if not counts.empty else "—"
        a, b, c = st.columns(3)
        with a:
            metric("SAVED RECORDS", f"{total:,}", "Records in current history", "▤")
        with b:
            metric("CATEGORIES FOUND", f"{len(counts)}", "Distinct recorded categories", "♻")
        with c:
            metric("MOST COMMON", top.replace("_", " ").title(), "Highest record count", "⌁")
        chart1, chart2 = st.columns(2, gap="large")
        with chart1:
            st.plotly_chart(px.bar(counts, x="Category", y="Scans", title="Records by waste category", text_auto=True), use_container_width=True)
        with chart2:
            st.plotly_chart(px.pie(counts, names="Category", values="Scans", hole=0.58, title="Category distribution"), use_container_width=True)
        st.caption("Charts summarise saved project records. They do not measure verified recycling rates or environmental impact.")


elif page == "About":
    st.markdown('<div class="page-kicker">PROJECT INFORMATION</div>', unsafe_allow_html=True)
    st.title("About EcoVision AI")
    st.markdown('<p class="page-intro">An AI-assisted waste recognition and management system developed as an academic deep-learning project.</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="about-panel">
          <div class="about-heading">Project objective</div>
          EcoVision AI aims to support responsible waste handling by identifying supported waste objects from images and presenting clear, category-specific disposal guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Core features")
    feature_cols = st.columns(2, gap="medium")
    features = [
        ("01", "Image-based recognition", "YOLO object detection identifies supported waste categories when trained model weights are installed."),
        ("02", "Waste management guidance", "Provides practical recommendations for sorting, reuse, recycling and disposal."),
        ("03", "Cloud-backed history", "MongoDB Atlas stores detection records when a database connection is configured."),
        ("04", "Activity analytics", "Interactive charts summarise saved records by waste category."),
    ]
    for index, (number, heading, body) in enumerate(features):
        with feature_cols[index % 2]:
            st.markdown(f'<div class="feature-card"><span>{number}</span><strong>{heading}</strong><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown("### Deep-learning methodology")
    st.markdown(
        """
        1. **Image input and preprocessing** – an uploaded image is prepared for model inference.
        2. **CNN feature extraction** – convolutional layers learn visual patterns such as edges, textures and shapes.
        3. **Multi-scale feature processing** – visual features are combined to support object detection at different sizes.
        4. **Detection and classification** – the YOLO detection head predicts object locations, classes and confidence scores.
        5. **Post-processing** – confidence filtering and non-maximum suppression refine the predictions.
        6. **Guidance mapping** – the predicted category is linked to relevant waste-handling instructions.
        """
    )
    st.markdown("### Technology stack")
    st.markdown(
        """
        - **Programming language:** Python
        - **Frontend:** Streamlit
        - **Deep learning:** Ultralytics YOLO with CNN-based feature extraction
        - **Data processing and visualisation:** Pandas and Plotly
        - **Database:** MongoDB Atlas
        """
    )
    st.warning("This is an educational decision-support tool. Do not rely on image predictions for chemicals, medical waste, sharp objects or damaged batteries. Follow authorised local disposal guidance.")
    st.caption("Model predictions are only available after a suitable dataset has been prepared and a model has been trained and evaluated. Do not report model accuracy until it has been measured on a held-out test set.")
