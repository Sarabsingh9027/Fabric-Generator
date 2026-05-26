import streamlit as st
import requests
import base64
import io
from PIL import Image
import numpy as np

API_URL = "http://localhost:8000"   # change to deployed URL in production

st.set_page_config(page_title="Fabric Roll Detector", layout="wide")
st.title("🧵 Fabric Roll Detection & Color Variation")

tab1, tab2, tab3 = st.tabs(["Detect Fabrics", "Extract Colors", "Color Variations"])


# ── Tab 1: Detect ─────────────────────────────────────────
with tab1:
    st.header("Fabric Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="detect")
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(uploaded, use_column_width=True)
        with col2:
            st.subheader("Detected")
            with st.spinner("Running detection..."):
                resp = requests.post(
                    f"{API_URL}/detect",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Image type: **{data['image_type'].upper()}** — Found **{data['count']}** fabric roll(s)")
                img_bytes = base64.b64decode(data["annotated_image_b64"])
                st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)

                with st.expander("Box details"):
                    for i, (box, conf) in enumerate(zip(data["boxes"], data["confidences"])):
                        st.write(f"Roll #{i+1}: box={box}, confidence={conf:.2f}")
            else:
                st.error(f"API error: {resp.text}")


# ── Tab 2: Extract Colors ────────────────────────────────
with tab2:
    st.header("Color Extraction")
    uploaded_c = st.file_uploader("Upload a group image", type=["jpg","jpeg","png"], key="colors")
    if uploaded_c:
        with st.spinner("Extracting colors..."):
            resp = requests.post(
                f"{API_URL}/extract-colors",
                files={"file": (uploaded_c.name, uploaded_c.getvalue(), uploaded_c.type)}
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Found **{data['count']}** fabric(s) — type: {data['image_type'].upper()}")
            cols = st.columns(min(data["count"], 4))
            for i, color in enumerate(data["colors"]):
                with cols[i % 4]:
                    crop_bytes = base64.b64decode(color["crop_b64"])
                    st.image(Image.open(io.BytesIO(crop_bytes)), caption=f"Roll #{color['crop_id']}", use_column_width=True)
                    r, g, b = color["rgb"]
                    st.markdown(
                        f"<div style='background:{color['hex']};padding:8px;border-radius:6px;"
                        f"text-align:center;color:{'#fff' if r+g+b < 380 else '#222'};font-size:13px'>"
                        f"{color['hex']}<br>RGB({r},{g},{b})</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.error(f"API error: {resp.text}")


# ── Tab 3: Generate Variations ────────────────────────────
with tab3:
    st.header("Generate Color Variations")
    st.info("Upload a single fabric image and a group image. The app will recolor the single fabric using colors found in the group.")

    col_s, col_g = st.columns(2)
    with col_s:
        single_file = st.file_uploader("Single fabric image", type=["jpg","jpeg","png"], key="single")
        if single_file:
            st.image(single_file, caption="Single fabric", use_column_width=True)
    with col_g:
        group_file = st.file_uploader("Group image (color source)", type=["jpg","jpeg","png"], key="group")
        if group_file:
            st.image(group_file, caption="Group (color source)", use_column_width=True)

    if single_file and group_file:
        if st.button("Generate Variations", type="primary"):
            with st.spinner("Generating color variations..."):
                resp = requests.post(
                    f"{API_URL}/generate-variations",
                    files={
                        "single_file": (single_file.name, single_file.getvalue(), single_file.type),
                        "group_file":  (group_file.name, group_file.getvalue(), group_file.type),
                    }
                )
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Original fabric")
                orig_bytes = base64.b64decode(data["original_b64"])
                st.image(Image.open(io.BytesIO(orig_bytes)), width=300)

                st.subheader(f"Color variations ({len(data['variations'])} total)")
                var_cols = st.columns(min(len(data["variations"]), 4))
                for i, var in enumerate(data["variations"]):
                    with var_cols[i % 4]:
                        img_bytes = base64.b64decode(var["image_b64"])
                        r, g, b = var["rgb"]
                        st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)
                        st.markdown(
                            f"<div style='background:{var['hex']};padding:6px;border-radius:5px;"
                            f"text-align:center;color:{'#fff' if r+g+b < 380 else '#222'};font-size:12px'>"
                            f"{var['hex']}</div>",
                            unsafe_allow_html=True
                        )
                        # Download button
                        st.download_button(
                            label=f"↓ Download",
                            data=base64.b64decode(var["image_b64"]),
                            file_name=f"variation_{var['id']}_{var['hex'].replace('#','')}.jpg",
                            mime="image/jpeg",
                            key=f"dl_{i}"
                        )
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")