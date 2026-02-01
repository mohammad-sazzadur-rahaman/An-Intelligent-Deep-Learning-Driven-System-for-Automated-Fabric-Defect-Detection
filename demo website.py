import io
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# ---------- Page ----------
st.set_page_config(
    page_title="Fabric Defect Detector",
    page_icon="🧵",
    layout="wide",
)

# ---------- Ultra-compact + Premium styling ----------
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 15% 10%, #101a2a, #070b14 55%, #05070f); }

      .block-container { padding-top: 0.35rem; padding-bottom: 0.45rem; }

      section[data-testid="stSidebar"] > div { padding-top: 0.55rem !important; }
      section[data-testid="stSidebar"] { background: rgba(255,255,255,0.04) !important; }

      h1 { font-size: 2.00rem !important; margin: 0.05rem 0 0.18rem 0 !important; }
      h2 { font-size: 1.35rem !important; margin: 0.05rem 0 0.15rem 0 !important; }
      h3 { font-size: 1.10rem !important; margin: 0.05rem 0 0.12rem 0 !important; }

      h1, h2, h3, p, label, span, div { color: #eaf0ff !important; }

      .card{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 15px;
        padding: 10px 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.33);
      }

      .pill{
        display:inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        background: rgba(99,102,241,0.22);
        border: 1px solid rgba(99,102,241,0.42);
        margin-right: 6px;
        font-size: 0.80rem;
      }

      .muted{ opacity: 0.78; }

      div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.14rem; }

      section[data-testid="stFileUploaderDropzone"] { padding: 0.45rem !important; }

      .metric{
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 7px 9px;
      }

      hr { margin: 0.20rem 0 !important; }

      @media (max-width: 1100px) {
        div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
names = model.names

# ---------- Header ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("🧵 Fabric Defect Detection (Local YOLOv8)")
st.write("Upload image(s) → Run inference locally → Get boxes + defect name + probability.")
st.markdown(
    '<span class="pill">Local</span><span class="pill">Fast</span><span class="pill">Offline</span>',
    unsafe_allow_html=True
)
st.markdown('<p class="muted" style="margin:0;">Model: <b>best.pt</b></p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ---------- Sidebar Controls ----------
st.sidebar.title("🧵 Controls")

uploaded_files = st.sidebar.file_uploader(
    "Upload up to 5 images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.10, 0.95, 0.75, 0.05)

imgsz = st.sidebar.selectbox("YOLO Input Size (imgsz)", [640, 512, 416], index=0)

run = st.sidebar.button("🚀 Run Detection", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use imgsz=640 for best accuracy. Increase confidence to reduce false positives.")

# ---------- Validate max 5 ----------
if uploaded_files and len(uploaded_files) > 5:
    st.sidebar.error("❌ Maximum 5 images allowed. Please upload 5 or fewer.")
    uploaded_files = uploaded_files[:5]

# ---------- Main Layout ----------
col1, col2 = st.columns([1, 1.25], gap="small")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview (Original)")
    if uploaded_files:
        for idx, uf in enumerate(uploaded_files, start=1):
            img_in = Image.open(uf).convert("RGB")
            st.write(f"Image {idx}: **{uf.name}**")
            st.image(img_in, use_container_width=True)
    else:
        st.info("Upload up to 5 images using the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")

    if uploaded_files and run:
        all_rows = []

        for idx, uf in enumerate(uploaded_files, start=1):
            img_in = Image.open(uf).convert("RGB")

            # ✅ BEST: Don't stretch resize
            # YOLO will resize correctly using imgsz parameter
            results = model.predict(
                source=img_in,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )

            r = results[0]

            # Annotated image
            annotated_bgr = r.plot()
            annotated_rgb = annotated_bgr[..., ::-1]
            out_img = Image.fromarray(annotated_rgb)

            # Extract detections
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    score = float(b.conf.item())
                    all_rows.append({
                        "Image": uf.name,
                        "Defect": names.get(cls_id, str(cls_id)),
                        "Probability": round(score, 4),
                    })

            st.write(f"### ✅ Result {idx}: {uf.name}")
            st.image(out_img, use_container_width=True)

            # Download output image
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            st.download_button(
                label=f"⬇️ Download Output {idx}",
                data=buf.getvalue(),
                file_name=f"prediction_{idx}_{uf.name}.png",
                mime="image/png",
                use_container_width=True
            )

        # Combined Table
        st.markdown("### 📋 All Detections (sorted)")
        if len(all_rows) == 0:
            st.success("✅ No defects detected in all uploaded images.")
        else:
            df_all = pd.DataFrame(all_rows).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(df_all, use_container_width=True, height=240)

    else:
        st.info("Run detection from the sidebar after uploading image(s).")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    '<p class="muted" style="text-align:center; margin-top: 4px;">'
    'Made locally • YOLOv8 • Fabric Defect Detection</p>',
    unsafe_allow_html=True
)
