import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import time

st.set_page_config(
    page_title="LeafScan — Plant Disease AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0d1117;
    color: #e8f0e8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1200px; }
section[data-testid="stSidebar"] { display: none; }

.hero-wrap { position:relative; padding:4rem 0 2.5rem; text-align:center; }
.hero-bg { position:absolute; inset:0; background:radial-gradient(ellipse 80% 50% at 50% 0%,#0d3318 0%,transparent 70%); pointer-events:none; }
.hero-tag { display:inline-block; background:#0d3318; border:1px solid #1a6b30; color:#4ade80; font-size:0.72rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; padding:0.35rem 1rem; border-radius:20px; margin-bottom:1.2rem; }
.hero-title { font-family:'Playfair Display',serif; font-size:clamp(2.2rem,5vw,3.8rem); font-weight:700; color:#f0f9f0; line-height:1.15; margin-bottom:1rem; letter-spacing:-1px; }
.hero-title span { color:#4ade80; }
.hero-sub { font-size:1rem; color:#7aaa8a; font-weight:300; max-width:480px; margin:0 auto; line-height:1.7; }
.hero-stats { display:flex; justify-content:center; gap:2.5rem; margin-top:2rem; }
.stat { text-align:center; }
.stat-val { font-family:'Playfair Display',serif; font-size:1.8rem; color:#4ade80; font-weight:700; }
.stat-lbl { font-size:0.7rem; color:#5a8a6a; text-transform:uppercase; letter-spacing:1px; margin-top:0.1rem; }

.pipe-track { display:flex; align-items:center; background:#111a13; border:1px solid #1e3a22; border-radius:14px; padding:1.2rem 1.5rem; margin:2rem 0; gap:0; overflow-x:auto; }
.pipe-node { display:flex; flex-direction:column; align-items:center; gap:0.3rem; flex:1; min-width:90px; padding:0.5rem; border-radius:10px; transition:all 0.3s ease; }
.pipe-node.idle { opacity:0.3; }
.pipe-node.active { background:#0d3318; border:1px solid #22c55e; opacity:1; transform:scale(1.05); box-shadow:0 0 18px rgba(34,197,94,0.2); }
.pipe-node.done { opacity:0.75; }
.pn-num { width:30px; height:30px; border-radius:50%; border:1.5px solid #2a5a35; display:flex; align-items:center; justify-content:center; font-size:0.78rem; font-weight:600; color:#4ade80; background:#0d1a10; }
.pipe-node.active .pn-num { background:#16a34a; color:#fff; border-color:#22c55e; box-shadow:0 0 8px rgba(34,197,94,0.5); }
.pipe-node.done .pn-num { background:#16a34a; color:#fff; border-color:#16a34a; }
.pn-label { font-size:0.76rem; font-weight:500; color:#a0c8a8; text-align:center; }
.pipe-node.active .pn-label { color:#bbf7d0; font-weight:600; }
.pn-sub { font-size:0.64rem; color:#4a7a58; text-align:center; }
.pipe-sep { flex:0 0 auto; width:32px; height:1px; background:linear-gradient(90deg,#1e3a22,#2a5a35,#1e3a22); position:relative; }
.pipe-sep::after { content:'›'; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#2a5a35; font-size:1rem; background:#111a13; padding:0 3px; }

.card { background:#111a13; border:1px solid #1e3a22; border-radius:14px; padding:1.4rem; margin-bottom:1rem; }
.card-title { font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; color:#4ade80; margin-bottom:1rem; display:flex; align-items:center; gap:0.5rem; }
.card-title::before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; background:#4ade80; box-shadow:0 0 6px #4ade80; }

.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.8rem; }
.meta-pill { background:#0d1a10; border:1px solid #1e3a22; border-radius:8px; padding:0.55rem 0.8rem; }
.meta-key { font-size:0.63rem; color:#4a7a58; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.15rem; }
.meta-val { font-size:0.85rem; color:#bbf7d0; font-weight:500; font-family:monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

.pipe-step-row { display:flex; align-items:flex-start; gap:0.7rem; padding:0.6rem 0; border-bottom:1px solid #1a2e1d; }
.pipe-step-row:last-child { border-bottom:none; }
.step-dot { width:7px; height:7px; border-radius:50%; background:#22c55e; margin-top:5px; flex-shrink:0; box-shadow:0 0 5px rgba(34,197,94,0.5); }
.step-key { font-size:0.76rem; color:#4ade80; font-weight:600; margin-bottom:0.1rem; }
.step-val { font-size:0.8rem; color:#7aaa8a; font-family:'Courier New',monospace; line-height:1.4; }

.result-hero { border-radius:14px; padding:2rem; text-align:center; position:relative; overflow:hidden; margin-bottom:1rem; }
.result-healthy { background:linear-gradient(135deg,#052e14,#064e20); border:1px solid #22c55e; }
.result-moderate { background:linear-gradient(135deg,#2d1a00,#4a2e00); border:1px solid #f59e0b; }
.result-severe { background:linear-gradient(135deg,#2d0a0a,#4a1010); border:1px solid #ef4444; }
.r-glow { position:absolute; inset:0; pointer-events:none; }
.result-healthy .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(34,197,94,0.12),transparent 55%); }
.result-moderate .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(245,158,11,0.12),transparent 55%); }
.result-severe .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(239,68,68,0.12),transparent 55%); }
.r-plant { font-size:0.68rem; letter-spacing:3px; text-transform:uppercase; color:#7aaa8a; margin-bottom:0.4rem; }
.r-disease { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; color:#f0f9f0; margin-bottom:0.8rem; line-height:1.2; }
.r-conf-num { font-family:'Playfair Display',serif; font-size:3.2rem; font-weight:700; line-height:1; }
.result-healthy .r-conf-num { color:#4ade80; }
.result-moderate .r-conf-num { color:#fbbf24; }
.result-severe .r-conf-num { color:#f87171; }
.r-conf-unit { font-size:0.9rem; color:#5a8a6a; }
.badge { display:inline-block; padding:0.28rem 0.9rem; border-radius:20px; font-size:0.7rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-top:0.8rem; }
.badge-none { background:#052e14; color:#4ade80; border:1px solid #22c55e; }
.badge-medium { background:#2d1a00; color:#fbbf24; border:1px solid #f59e0b; }
.badge-high { background:#2d0a0a; color:#f87171; border:1px solid #ef4444; }

.info-panel { background:#0d1a10; border:1px solid #1e3a22; border-radius:10px; padding:0.9rem 1.1rem; margin:0.5rem 0; }
.info-key { font-size:0.65rem; color:#4ade80; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.35rem; font-weight:600; }
.info-val { font-size:0.86rem; color:#a0c8a8; line-height:1.6; }

.pred-row { display:flex; align-items:center; gap:0.7rem; padding:0.45rem 0; border-bottom:1px solid #1a2e1d; }
.pred-row:last-child { border-bottom:none; }
.pred-rank { font-size:0.7rem; color:#4a7a58; width:16px; text-align:center; flex-shrink:0; }
.pred-label { font-size:0.8rem; color:#a0c8a8; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pred-label.top { color:#bbf7d0; font-weight:500; }
.pred-bar-bg { width:100px; height:5px; background:#1a2e1d; border-radius:3px; overflow:hidden; flex-shrink:0; }
.pred-bar-fill { height:100%; border-radius:3px; }
.pred-pct { font-size:0.76rem; color:#5a8a6a; font-family:monospace; min-width:42px; text-align:right; }
.pred-pct.top { color:#4ade80; font-weight:600; }

.stButton>button { background:#16a34a !important; color:#fff !important; border:none !important; border-radius:10px !important; font-family:'Outfit',sans-serif !important; font-weight:600 !important; font-size:0.9rem !important; letter-spacing:0.5px; transition:all 0.2s !important; }
.stButton>button:hover { background:#15803d !important; transform:translateY(-1px) !important; box-shadow:0 4px 18px rgba(34,197,94,0.3) !important; }
hr { border-color:#1e3a22 !important; margin:1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── CLASSES (alphabetical — matches ImageFolder) ───────────────────
CLASSES = sorted([
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
    'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
    'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch',
    'Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
])

DB = {
    'Apple_scab':('medium','Fungal lesions causing dark scabby spots on leaves and fruit.','Apply fungicide. Remove infected leaves. Prune for better airflow.'),
    'Black_rot':('high','Fungal disease with circular brown spots and dark margins.','Prune infected branches. Apply copper fungicide. Remove mummified fruit.'),
    'Cedar_apple_rust':('medium','Orange rust spots on upper leaf surface, tubes below.','Apply myclobutanil in spring. Remove nearby cedar trees if possible.'),
    'Powdery_mildew':('medium','White powdery fungal coating on the leaf surface.','Apply sulfur or neem-based fungicide. Improve air circulation.'),
    'Cercospora_leaf_spot':('medium','Circular gray spots with dark purple borders.','Apply fungicide. Remove crop debris. Avoid overhead watering.'),
    'Common_rust':('medium','Orange-brown rust pustules on both leaf surfaces.','Apply fungicide early in season. Use resistant corn varieties.'),
    'Northern_Leaf_Blight':('high','Long tan cigar-shaped lesions with dark wavy borders.','Apply fungicide. Rotate crops annually. Use resistant varieties.'),
    'Esca':('high','Tiger-stripe yellowing pattern with dark wood streaks.','No cure available. Remove infected vines to prevent spread.'),
    'Leaf_blight':('high','Brown water-soaked lesions spreading rapidly across leaves.','Apply copper fungicide. Avoid wet foliage. Improve drainage.'),
    'Haunglongbing':('high','Yellow shoots and blotchy mottled asymmetric leaves.','No cure. Remove infected trees immediately to stop spread.'),
    'Bacterial_spot':('medium','Small water-soaked spots turning angular and brown.','Apply copper spray. Avoid working with plants when foliage is wet.'),
    'Early_blight':('medium','Dark concentric-ring spots — like a target on leaves.','Apply fungicide. Remove lower infected leaves. Mulch around base.'),
    'Late_blight':('high','Water-soaked lesions turning dark brown, white mold below.','Apply fungicide immediately. Remove and destroy infected plants.'),
    'Leaf_Mold':('medium','Pale yellow spots above, olive-green mold on underside.','Improve ventilation. Reduce humidity. Apply appropriate fungicide.'),
    'Septoria_leaf_spot':('medium','Small circular spots — dark border, light gray center.','Apply fungicide. Remove infected lower leaves promptly.'),
    'Spider_mites':('medium','Yellow stippling with fine silky webbing on leaves.','Apply miticide or neem oil. Increase ambient humidity around plants.'),
    'Target_Spot':('medium','Circular brown spots with distinct concentric rings.','Apply fungicide. Improve air circulation around the plant canopy.'),
    'Yellow_Leaf_Curl_Virus':('high','Upward leaf curl, yellowing margins, stunted growth.','No cure. Remove infected plants. Control whitefly vectors.'),
    'mosaic_virus':('high','Mosaic light-dark green mottled pattern, distorted leaves.','No cure. Remove infected plants. Control aphid vectors immediately.'),
    'Leaf_scorch':('medium','Brown scorched margins and leaf tips due to stress.','Improve irrigation consistency. Mulch around base of plant.'),
    'healthy':('none','No disease detected. The plant appears healthy.','Continue regular watering, fertilization, and routine monitoring.'),
}

def get_info(cls):
    plant, cond = cls.split('___')
    if cond == 'healthy':
        return plant, 'Healthy', DB['healthy']
    for key, val in DB.items():
        if key.lower().replace('_','') in cond.lower().replace('_','').replace(' ',''):
            return plant, cond.replace('_',' '), val
    return plant, cond.replace('_',' '), ('medium',f'Disease detected in {plant}.','Consult an agricultural expert for treatment advice.')

# ── MODEL — EfficientNet-B3 (exact match to notebook) ─────────────
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.network = models.efficientnet_b3(weights=None)
        in_features  = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, xb):
        return self.network(xb)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = EfficientNetB3(38)
    path   = 'src/plant-disease-model (1).pth'
    if not os.path.exists(path):
        return None, device, False
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device); model.eval()
        return model, device, True
    except:
        return None, device, False

# ── TRANSFORM — exact match to notebook ───────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict(image, model, device):
    t = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    top5p, top5i = torch.topk(probs, 5)
    return [{'class':CLASSES[i.item()],'prob':p.item()} for p,i in zip(top5p.cpu(),top5i.cpu())]

# ── SESSION STATE ─────────────────────────────────────────────────
for k,v in [('stage',0),('results',None),('image',None),('fname',''),('fsize',0),('imgwh',(0,0))]:
    if k not in st.session_state: st.session_state[k]=v

model, device, loaded = load_model()
stage = st.session_state.stage

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-bg"></div>
  <div class="hero-tag">🌱 AI-Powered Plant Diagnostics</div>
  <h1 class="hero-title">Detect Plant <span>Disease</span><br>Instantly</h1>
  <p class="hero-sub">Upload a leaf photo. EfficientNet-B3 — trained on 70,295 images across 38 classes — delivers a diagnosis in seconds.</p>
  <div class="hero-stats">
    <div class="stat"><div class="stat-val">38</div><div class="stat-lbl">Disease Classes</div></div>
    <div class="stat"><div class="stat-val">70K+</div><div class="stat-lbl">Training Images</div></div>
    <div class="stat"><div class="stat-val">14</div><div class="stat-lbl">Plant Species</div></div>
    <div class="stat"><div class="stat-val">99%+</div><div class="stat-lbl">Val Accuracy</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE BAR ──────────────────────────────────────────────────
def nc(i): return "done" if stage>i else ("active" if stage==i else "idle")
steps = [(1,"Upload","Leaf photo"),(2,"Preprocess","300×300 · Normalize"),(3,"EfficientNet","B3 inference"),(4,"Output","Diagnosis")]
html  = '<div class="pipe-track">'
for i,(n,lbl,sub) in enumerate(steps):
    html += f'<div class="pipe-node {nc(i)}"><div class="pn-num">{n}</div><div class="pn-label">{lbl}</div><div class="pn-sub">{sub}</div></div>'
    if i<3: html += '<div class="pipe-sep"></div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drop your leaf image here", type=["jpg","jpeg","png"], label_visibility="visible")
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.session_state.update({'image':img,'fname':uploaded.name,'fsize':round(uploaded.size/1024,1),'imgwh':img.size,'stage':1,'results':None})
    stage = 1

st.markdown("<hr>", unsafe_allow_html=True)

if not st.session_state.image:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#2a5a35">
      <div style="font-size:3.5rem;opacity:0.3;margin-bottom:1rem">🍃</div>
      <div style="font-size:1rem;color:#3a6a45">Upload a leaf image above to begin</div>
      <div style="font-size:0.82rem;color:#2a4a30;margin-top:0.5rem">
        Apple · Grape · Tomato · Potato · Corn · Peach · Cherry · Pepper<br>
        Strawberry · Blueberry · Orange · Soybean · Squash · Raspberry
      </div>
    </div>""", unsafe_allow_html=True)
else:
    img  = st.session_state.image
    w, h = st.session_state.imgwh

    # ── ROW 1: Upload + Preprocessing ────────────────────────────
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><div class="card-title">① Original Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f"""
        <div class="meta-grid">
          <div class="meta-pill"><div class="meta-key">Filename</div><div class="meta-val">{st.session_state.fname}</div></div>
          <div class="meta-pill"><div class="meta-key">File size</div><div class="meta-val">{st.session_state.fsize} KB</div></div>
          <div class="meta-pill"><div class="meta-key">Dimensions</div><div class="meta-val">{w} × {h} px</div></div>
          <div class="meta-pill"><div class="meta-key">Color mode</div><div class="meta-val">RGB · 3 channels</div></div>
        </div></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">② Preprocessing Pipeline</div>', unsafe_allow_html=True)
        st.image(img.resize((300,300)), caption="After Resize → 300×300", use_container_width=True)
        st.markdown(f"""
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Resize</div><div class="step-val">{w}×{h} → 300×300 px</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">ToTensor</div><div class="step-val">PIL → torch.FloatTensor [0.0 – 1.0]</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Normalize (ImageNet)</div><div class="step-val">mean=[0.485, 0.456, 0.406]<br>std =[0.229, 0.224, 0.225]</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Final tensor</div><div class="step-val">torch.Size([1, 3, 300, 300]) → {str(device).upper()}</div></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── ROW 2: Model info + Inference button ─────────────────────
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(f"""
        <div class="card"><div class="card-title">③ EfficientNet-B3 Model</div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Architecture</div><div class="step-val">EfficientNet-B3 (pretrained ImageNet)</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Classifier head</div><div class="step-val">Dropout(0.3) → Linear(1536 → 38)</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Training config</div><div class="step-val">3 epochs · OneCycleLR · lr=0.001 · Adam · grad_clip=0.1</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Dataset</div><div class="step-val">PlantVillage · 70,295 images · 38 classes</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Output</div><div class="step-val">Softmax → 38 class probabilities</div></div></div>
        <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Device</div><div class="step-val">{str(device).upper()}</div></div></div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="card-title">Run inference</div>', unsafe_allow_html=True)
        if not loaded:
            st.error("⚠️ plant-disease-model.pth not found.")
            st.info("Copy your trained model file to the same folder as app.py")
        else:
            btn = st.button("▶  Analyze Leaf", use_container_width=True)
            if btn or st.session_state.results is not None:
                if st.session_state.results is None:
                    with st.spinner("Running EfficientNet-B3..."):
                        time.sleep(0.3)
                        st.session_state.results = predict(img, model, device)
                        st.session_state.stage   = 3
                        stage = 3

                if st.session_state.results:
                    top  = st.session_state.results[0]
                    conf = top['prob']*100
                    _, _, (sev,_,_) = get_info(top['class'])
                    clr  = "#4ade80" if sev=="none" else "#fbbf24" if sev=="medium" else "#f87171"
                    st.markdown(f"""
                    <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Status</div><div class="step-val" style="color:#4ade80">Inference complete ✓</div></div></div>
                    <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Top prediction</div><div class="step-val">{top['class']}</div></div></div>
                    <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Confidence</div><div class="step-val" style="color:{clr};font-size:1.05rem;font-weight:600">{conf:.2f}%</div></div></div>
                    <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Class index</div><div class="step-val">{CLASSES.index(top['class'])} / 37</div></div></div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 3: OUTPUT ─────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("<hr>", unsafe_allow_html=True)
        results = st.session_state.results
        top     = results[0]
        plant, disease, (sev, desc, treat) = get_info(top['class'])
        conf    = top['prob'] * 100
        is_ok   = 'healthy' in top['class']
        crd     = "result-healthy" if is_ok else ("result-severe" if sev=="high" else "result-moderate")
        bdg     = {"none":"badge-none","medium":"badge-medium","high":"badge-high"}[sev]
        btxt    = {"none":"✓ Healthy","medium":"⚠ Moderate","high":"⛔ Severe"}[sev]

        c5, c6 = st.columns(2, gap="large")
        with c5:
            st.markdown(f"""
            <div class="card-title">④ Diagnosis Output</div>
            <div class="result-hero {crd}">
              <div class="r-glow"></div>
              <div class="r-plant">{plant.replace('_',' ').replace('(','').replace(')','')}</div>
              <div class="r-disease">{disease}</div>
              <div style="margin:0.8rem 0">
                <span class="r-conf-num">{conf:.1f}</span>
                <span class="r-conf-unit">% confidence</span>
              </div>
              <span class="badge {bdg}">{btxt}</span>
            </div>
            <div class="info-panel"><div class="info-key">About this condition</div><div class="info-val">{desc}</div></div>
            <div class="info-panel"><div class="info-key">Recommended treatment</div><div class="info-val">{treat}</div></div>
            """, unsafe_allow_html=True)

        with c6:
            st.markdown('<div class="card"><div class="card-title">Top 5 Predictions</div>', unsafe_allow_html=True)
            for i, r in enumerate(results):
                p2, c2  = r['class'].split('___')
                lbl     = f"{p2} · {c2.replace('_',' ')[:26]}"
                pct     = r['prob']*100
                clr2    = "#4ade80" if c2=="healthy" else ("#f87171" if i==0 and sev=="high" else "#fbbf24" if i==0 else "#4a7a58")
                tc      = "top" if i==0 else ""
                st.markdown(f"""
                <div class="pred-row">
                  <div class="pred-rank">{'▶' if i==0 else i+1}</div>
                  <div class="pred-label {tc}">{lbl}</div>
                  <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{int(pct)}%;background:{clr2}"></div></div>
                  <div class="pred-pct {tc}">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄  Analyze another image", use_container_width=True):
                for k,v in [('stage',0),('results',None),('image',None),('fname',''),('fsize',0),('imgwh',(0,0))]:
                    st.session_state[k]=v
                st.rerun()

st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;color:#2a5a35;font-size:0.76rem;border-top:1px solid #1a2e1d;margin-top:3rem">
  EfficientNet-B3 · PlantVillage · 70,295 images · 38 classes · 3 epochs · OneCycleLR · Adam · lr=0.001
</div>
""", unsafe_allow_html=True)