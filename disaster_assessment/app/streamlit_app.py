"""
Streamlit frontend for the AI Disaster Damage Assessment system.

Communicates with the FastAPI backend running on http://localhost:8000.

Start the backend first:
    uvicorn app.main:app --reload

Then start this frontend:
    streamlit run streamlit_app.py
"""

import io
import json

import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"

DAMAGE_COLORS = {
    "no_damage": "🟢",
    "minor_damage": "🟡",
    "major_damage": "🟠",
    "destroyed": "🔴",
}

DAMAGE_BADGE_CSS = {
    "no_damage": "background:#2e7d32; color:white; padding:3px 10px; border-radius:12px;",
    "minor_damage": "background:#f9a825; color:black; padding:3px 10px; border-radius:12px;",
    "major_damage": "background:#e65100; color:white; padding:3px 10px; border-radius:12px;",
    "destroyed": "background:#b71c1c; color:white; padding:3px 10px; border-radius:12px;",
    "unknown": "background:#424242; color:white; padding:3px 10px; border-radius:12px;",
}


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Disaster Damage Assessment",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ AI Disaster Damage Assessment")
st.markdown(
    "Upload **before** and **after** satellite images to detect buildings "
    "and assess structural damage caused by floods, earthquakes, or hurricanes."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Backend status
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ System Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            st.success(f"✅ Backend: **{health['status'].upper()}**")
            st.info(f"Version: {health.get('version', 'N/A')}")

            if health.get("mock_mode"):
                st.warning(
                    "⚠️ **MOCK MODE**\n\n"
                    "Predictions are simulated.\n"
                    "No real AI models loaded."
                )
            else:
                st.success("🤖 Real AI models loaded")

            st.markdown("**Models:**")
            bd_status = "✅ Loaded" if health.get("building_detector_loaded") else "❌ Not loaded (mock)"
            dc_status = "✅ Loaded" if health.get("damage_classifier_loaded") else "❌ Not loaded (mock)"
            st.markdown(f"- Building Detector: {bd_status}")
            st.markdown(f"- Damage Classifier: {dc_status}")
        else:
            st.error(f"Backend error: HTTP {resp.status_code}")
    except requests.ConnectionError:
        st.error(
            "❌ **Backend not reachable**\n\n"
            "Start the FastAPI server:\n"
            "```\nuvicorn app.main:app --reload\n```"
        )

    st.markdown("---")
    st.markdown("**API Docs:** [/docs](http://localhost:8000/docs)")
    st.markdown("**Health:** [/health](http://localhost:8000/health)")


# ─────────────────────────────────────────────────────────────────────────────
# Main – Image upload
# ─────────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Before Image")
    before_file = st.file_uploader(
        "Upload satellite image BEFORE disaster",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="before",
    )
    if before_file:
        st.image(before_file, caption="Before Disaster", use_column_width=True)

with col2:
    st.subheader("📸 After Image")
    after_file = st.file_uploader(
        "Upload satellite image AFTER disaster",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="after",
    )
    if after_file:
        st.image(after_file, caption="After Disaster", use_column_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────────────────────────────────────

if st.button("🚀 Run Damage Assessment", type="primary", disabled=(not before_file or not after_file)):
    with st.spinner("Analysing images... please wait."):
        try:
            before_bytes = before_file.getvalue()
            after_bytes = after_file.getvalue()

            files = {
                "before_image": (before_file.name, io.BytesIO(before_bytes), "image/jpeg"),
                "after_image": (after_file.name, io.BytesIO(after_bytes), "image/jpeg"),
            }

            response = requests.post(
                f"{BACKEND_URL}/predict",
                files=files,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Analysis complete!")

                # ── Mock warning ─────────────────────────────────────────────
                if result.get("mock_mode"):
                    st.warning(result.get("mock_warning", "⚠️ MOCK MODE – simulated predictions"))

                # ── Summary metrics ──────────────────────────────────────────
                summary = result.get("summary", {})
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("🏢 Total Buildings", summary.get("total_buildings", 0))
                m2.metric("🟢 No Damage",      summary.get("no_damage_count", 0))
                m3.metric("🟡 Minor Damage",   summary.get("minor_damage_count", 0))
                m4.metric("🟠 Major Damage",   summary.get("major_damage_count", 0))
                m5.metric("🔴 Destroyed",      summary.get("destroyed_count", 0))

                st.markdown(
                    f"⏱️ **Processing time:** {result.get('processing_time_ms', 0):.1f} ms"
                )
                st.markdown("---")

                # ── Per-building results ─────────────────────────────────────
                st.subheader("🏘️ Building-level Results")
                buildings = result.get("buildings", [])

                for b in buildings:
                    damage_cls = b.get("damage_class", "unknown")
                    icon = DAMAGE_COLORS.get(damage_cls, "⚪")
                    badge_style = DAMAGE_BADGE_CSS.get(damage_cls, DAMAGE_BADGE_CSS["unknown"])

                    with st.expander(
                        f"{icon} Building #{b['building_id']} – {b.get('damage_severity', 'Unknown')}",
                        expanded=False,
                    ):
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.markdown("**Bounding Box**")
                            bbox = b.get("bounding_box", {})
                            st.code(
                                f"x1={bbox.get('x1')}  y1={bbox.get('y1')}\n"
                                f"x2={bbox.get('x2')}  y2={bbox.get('y2')}\n"
                                f"w={bbox.get('width')}   h={bbox.get('height')}",
                                language="text",
                            )
                        with bc2:
                            st.markdown("**Damage Assessment**")
                            st.markdown(
                                f"<span style='{badge_style}'>{b.get('damage_severity', 'Unknown')}</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"Detection confidence: **{b.get('detection_confidence', 0):.1%}**")
                            st.markdown(f"Damage confidence:   **{b.get('damage_confidence', 0):.1%}**")
                            if b.get("is_mock"):
                                st.caption("⚠️ Simulated result")

                # ── Raw JSON toggle ──────────────────────────────────────────
                with st.expander("🔍 View Raw JSON Response"):
                    st.json(result)

            else:
                try:
                    err = response.json()
                    st.error(f"❌ Prediction failed (HTTP {response.status_code}): {err.get('detail', 'Unknown error')}")
                except Exception:
                    st.error(f"❌ HTTP {response.status_code}: {response.text[:300]}")

        except requests.ConnectionError:
            st.error(
                "❌ Cannot reach the backend. "
                "Make sure the FastAPI server is running:\n"
                "```\nuvicorn app.main:app --reload\n```"
            )
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")

elif not before_file or not after_file:
    st.info("👆 Upload both a **before** and an **after** image to enable the assessment button.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "AI Disaster Damage Assessment | EPICS College Project | "
    "Built with FastAPI + Streamlit + PyTorch + OpenCV"
)
