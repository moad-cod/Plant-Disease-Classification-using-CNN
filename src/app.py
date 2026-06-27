import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pickle
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
html, body, [class*="css"] { font-family:'Outfit',sans-serif; background:#0d1117; color:#e8f0e8; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 2rem 4rem !important; max-width:1280px; }
section[data-testid="stSidebar"] { display:none; }

/* HERO */
.hero-wrap { position:relative; padding:3.5rem 0 2rem; text-align:center; }
.hero-bg { position:absolute; inset:0; background:radial-gradient(ellipse 80% 50% at 50% 0%,#0d3318 0%,transparent 70%); pointer-events:none; }
.hero-tag { display:inline-block; background:#0d3318; border:1px solid #1a6b30; color:#4ade80; font-size:0.72rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; padding:0.35rem 1rem; border-radius:20px; margin-bottom:1rem; }
.hero-title { font-family:'Playfair Display',serif; font-size:clamp(2rem,4vw,3.5rem); font-weight:700; color:#f0f9f0; line-height:1.15; margin-bottom:0.8rem; letter-spacing:-1px; }
.hero-title span { color:#4ade80; }
.hero-sub { font-size:0.95rem; color:#7aaa8a; font-weight:300; max-width:500px; margin:0 auto; line-height:1.7; }
.hero-stats { display:flex; justify-content:center; gap:2rem; margin-top:1.5rem; flex-wrap:wrap; }
.stat { text-align:center; }
.stat-val { font-family:'Playfair Display',serif; font-size:1.6rem; color:#4ade80; font-weight:700; }
.stat-lbl { font-size:0.68rem; color:#5a8a6a; text-transform:uppercase; letter-spacing:1px; }

/* PIPELINE */
.pipe-track { display:flex; align-items:center; background:#111a13; border:1px solid #1e3a22; border-radius:14px; padding:1rem 1.5rem; margin:1.5rem 0; gap:0; overflow-x:auto; }
.pipe-node { display:flex; flex-direction:column; align-items:center; gap:0.3rem; flex:1; min-width:85px; padding:0.5rem; border-radius:10px; transition:all 0.3s ease; }
.pipe-node.idle { opacity:0.3; }
.pipe-node.active { background:#0d3318; border:1px solid #22c55e; opacity:1; transform:scale(1.05); box-shadow:0 0 18px rgba(34,197,94,0.2); }
.pipe-node.done { opacity:0.8; }
.pn-num { width:28px; height:28px; border-radius:50%; border:1.5px solid #2a5a35; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; color:#4ade80; background:#0d1a10; }
.pipe-node.active .pn-num { background:#16a34a; color:#fff; border-color:#22c55e; box-shadow:0 0 8px rgba(34,197,94,0.5); }
.pipe-node.done .pn-num { background:#16a34a; color:#fff; border-color:#16a34a; }
.pn-label { font-size:0.74rem; font-weight:500; color:#a0c8a8; text-align:center; }
.pipe-node.active .pn-label { color:#bbf7d0; font-weight:600; }
.pn-sub { font-size:0.62rem; color:#4a7a58; text-align:center; }
.pipe-sep { flex:0 0 auto; width:28px; height:1px; background:linear-gradient(90deg,#1e3a22,#2a5a35,#1e3a22); position:relative; }
.pipe-sep::after { content:'›'; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#2a5a35; font-size:1rem; background:#111a13; padding:0 3px; }

/* CARDS */
.card { background:#111a13; border:1px solid #1e3a22; border-radius:14px; padding:1.3rem; margin-bottom:1rem; }
.card-title { font-size:0.66rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; color:#4ade80; margin-bottom:0.9rem; display:flex; align-items:center; gap:0.5rem; }
.card-title::before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; background:#4ade80; box-shadow:0 0 6px #4ade80; }

/* META */
.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.7rem; }
.meta-pill { background:#0d1a10; border:1px solid #1e3a22; border-radius:8px; padding:0.5rem 0.75rem; }
.meta-key { font-size:0.62rem; color:#4a7a58; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.12rem; }
.meta-val { font-size:0.83rem; color:#bbf7d0; font-weight:500; font-family:monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* PIPELINE STEPS */
.pipe-step-row { display:flex; align-items:flex-start; gap:0.7rem; padding:0.55rem 0; border-bottom:1px solid #1a2e1d; }
.pipe-step-row:last-child { border-bottom:none; }
.step-dot { width:7px; height:7px; border-radius:50%; background:#22c55e; margin-top:4px; flex-shrink:0; box-shadow:0 0 5px rgba(34,197,94,0.5); }
.step-dot.warn { background:#f59e0b; box-shadow:0 0 5px rgba(245,158,11,0.5); }
.step-dot.err  { background:#ef4444; box-shadow:0 0 5px rgba(239,68,68,0.5); }
.step-key { font-size:0.74rem; color:#4ade80; font-weight:600; margin-bottom:0.08rem; }
.step-key.warn { color:#fbbf24; }
.step-key.err  { color:#f87171; }
.step-val { font-size:0.78rem; color:#7aaa8a; font-family:'Courier New',monospace; line-height:1.4; }

/* ANOMALY ALERT */
.anomaly-card { border-radius:14px; padding:1.8rem; text-align:center; margin:1rem 0; background:linear-gradient(135deg,#1a0a0a,#2d1010); border:1px solid #ef4444; position:relative; overflow:hidden; }
.anomaly-card::before { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 50% -10%,rgba(239,68,68,0.1),transparent 60%); pointer-events:none; }
.anomaly-icon { font-size:2.5rem; margin-bottom:0.5rem; }
.anomaly-title { font-family:'Playfair Display',serif; font-size:1.8rem; color:#f87171; margin-bottom:0.5rem; }
.anomaly-reason { font-size:0.88rem; color:#fca5a5; margin-bottom:1rem; }
.anomaly-scores { display:flex; justify-content:center; gap:2rem; }
.score-box { text-align:center; }
.score-num { font-family:'Playfair Display',serif; font-size:1.4rem; color:#f87171; }
.score-lbl { font-size:0.65rem; color:#7a3a3a; text-transform:uppercase; letter-spacing:1px; }

/* RESULT */
.result-hero { border-radius:14px; padding:1.8rem; text-align:center; position:relative; overflow:hidden; margin-bottom:1rem; }
.result-healthy  { background:linear-gradient(135deg,#052e14,#064e20); border:1px solid #22c55e; }
.result-moderate { background:linear-gradient(135deg,#2d1a00,#4a2e00); border:1px solid #f59e0b; }
.result-severe   { background:linear-gradient(135deg,#2d0a0a,#4a1010); border:1px solid #ef4444; }
.r-glow { position:absolute; inset:0; pointer-events:none; }
.result-healthy  .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(34,197,94,0.12),transparent 55%); }
.result-moderate .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(245,158,11,0.12),transparent 55%); }
.result-severe   .r-glow { background:radial-gradient(ellipse at 50% -20%,rgba(239,68,68,0.12),transparent 55%); }
.r-plant { font-size:0.66rem; letter-spacing:3px; text-transform:uppercase; color:#7aaa8a; margin-bottom:0.3rem; }
.r-disease { font-family:'Playfair Display',serif; font-size:1.7rem; font-weight:700; color:#f0f9f0; margin-bottom:0.7rem; line-height:1.2; }
.r-conf-num { font-family:'Playfair Display',serif; font-size:3rem; font-weight:700; line-height:1; }
.result-healthy  .r-conf-num { color:#4ade80; }
.result-moderate .r-conf-num { color:#fbbf24; }
.result-severe   .r-conf-num { color:#f87171; }
.r-conf-unit { font-size:0.88rem; color:#5a8a6a; }
.badge { display:inline-block; padding:0.26rem 0.85rem; border-radius:20px; font-size:0.68rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-top:0.7rem; }
.badge-none   { background:#052e14; color:#4ade80; border:1px solid #22c55e; }
.badge-medium { background:#2d1a00; color:#fbbf24; border:1px solid #f59e0b; }
.badge-high   { background:#2d0a0a; color:#f87171; border:1px solid #ef4444; }

/* ANOMALY SCORES INSIDE NORMAL RESULT */
.scores-row { display:flex; gap:0.5rem; margin-top:0.6rem; }
.score-pill { flex:1; background:#0d1a10; border:1px solid #1e3a22; border-radius:8px; padding:0.5rem 0.7rem; text-align:center; }
.score-pill-lbl { font-size:0.6rem; color:#4a7a58; text-transform:uppercase; letter-spacing:1px; }
.score-pill-val { font-size:0.9rem; color:#4ade80; font-weight:600; font-family:monospace; }
.score-pill-val.warn { color:#fbbf24; }

/* INFO PANELS */
.info-panel { background:#0d1a10; border:1px solid #1e3a22; border-radius:10px; padding:0.85rem 1rem; margin:0.45rem 0; }
.info-key { font-size:0.63rem; color:#4ade80; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.3rem; font-weight:600; }
.info-val { font-size:0.84rem; color:#a0c8a8; line-height:1.6; }

/* TOP 5 */
.pred-row { display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0; border-bottom:1px solid #1a2e1d; }
.pred-row:last-child { border-bottom:none; }
.pred-rank { font-size:0.68rem; color:#4a7a58; width:16px; text-align:center; flex-shrink:0; }
.pred-label { font-size:0.78rem; color:#a0c8a8; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pred-label.top { color:#bbf7d0; font-weight:500; }
.pred-bar-bg { width:90px; height:5px; background:#1a2e1d; border-radius:3px; overflow:hidden; flex-shrink:0; }
.pred-bar-fill { height:100%; border-radius:3px; }
.pred-pct { font-size:0.74rem; color:#5a8a6a; font-family:monospace; min-width:40px; text-align:right; }
.pred-pct.top { color:#4ade80; font-weight:600; }

/* SECTION DIVIDER */
.sec-divider { display:flex; align-items:center; gap:1rem; margin:2rem 0 1rem; }
.sec-divider-line { flex:1; height:1px; background:#1e3a22; }
.sec-divider-txt { font-size:0.68rem; color:#4a7a58; text-transform:uppercase; letter-spacing:2px; white-space:nowrap; }

/* FOOTER */
.footer { text-align:center; padding:2.5rem 2rem 1.5rem; border-top:1px solid #1a2e1d; margin-top:2rem; }
.footer-tech { font-size:0.72rem; color:#2a5a35; line-height:1.8; }
.footer-links { display:flex; justify-content:center; gap:1.5rem; margin:1rem 0 1.25rem; flex-wrap:wrap; }
.footer-link { display:inline-flex; align-items:center; gap:0.4rem; color:#3a7a4a; text-decoration:none; font-size:0.82rem; font-weight:500; transition:color 0.2s; }
.footer-link:hover { color:#4ade80; }
.footer-link svg { width:18px; height:18px; flex-shrink:0; }
.footer-copy { font-size:0.68rem; color:#1e3a22; }


/* MODEL INSIGHTS TAB */
.insight-img { border-radius:10px; overflow:hidden; border:1px solid #1e3a22; }

/* BUTTONS */
.stButton>button { background:#16a34a !important; color:#fff !important; border:none !important; border-radius:10px !important; font-family:'Outfit',sans-serif !important; font-weight:600 !important; font-size:0.88rem !important; letter-spacing:0.5px; transition:all 0.2s !important; }
.stButton>button:hover { background:#15803d !important; transform:translateY(-1px) !important; box-shadow:0 4px 18px rgba(34,197,94,0.3) !important; }
hr { border-color:#1e3a22 !important; margin:1.5rem 0 !important; }

/* TABS */
[data-baseweb="tab-list"] { background:#111a13 !important; border-radius:10px !important; border:1px solid #1e3a22 !important; padding:4px !important; gap:4px !important; }
[data-baseweb="tab"] { background:transparent !important; color:#7aaa8a !important; border-radius:8px !important; font-size:0.82rem !important; }
[aria-selected="true"] { background:#16a34a !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────
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
    return plant, cond.replace('_',' '), ('medium', f'Disease detected in {plant}.', 'Consult an agricultural expert.')


# ─────────────────────────────────────────────
# GOAD TRANSFORMS — same as notebook
# ─────────────────────────────────────────────
_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
GOAD_TRANSFORMS = [
    transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((90,90)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((180,180)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((270,270)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((90,90)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
]
M = len(GOAD_TRANSFORMS)


# ─────────────────────────────────────────────
# MODEL — exact match to notebook
# ─────────────────────────────────────────────
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

    def extract_features(self, xb):
        features = self.network.features(xb)
        features = self.network.avgpool(features)
        features = features.flatten(1)
        logits   = self.network.classifier(features)
        return features, logits


# ─────────────────────────────────────────────
# LOAD MODEL + ANOMALY PARAMS
# ─────────────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model      = EfficientNetB3(38)
    model_path = os.path.join(BASE_DIR, 'Classification-based Anomaly Detection', 'plant-disease-model.pth')
    model_ok   = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            model_ok = True
            log.info('Model loaded successfully from %s', model_path)
        except Exception as e:
            log.error('Failed to load model from %s: %s', model_path, e)

    # Anomaly params
    anomaly_params = None
    anom_path      = os.path.join(BASE_DIR, 'Classification-based Anomaly Detection', 'anomaly_params.pkl')
    if os.path.exists(anom_path):
        try:
            with open(anom_path, 'rb') as f:
                anomaly_params = pickle.load(f)
            log.info('Anomaly params loaded from %s', anom_path)
        except Exception as e:
            log.error('Failed to load anomaly params from %s: %s', anom_path, e)

    return model, device, model_ok, anomaly_params


# ─────────────────────────────────────────────
# INFERENCE TRANSFORM
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# ─────────────────────────────────────────────
# GOAD SCORE — Mahalanobis distance
# ─────────────────────────────────────────────
def compute_goad_score(pil_image, model, device, cluster_centers, cov_inv, epsilon=0.1):
    log_probs = []
    for m_idx, t in enumerate(GOAD_TRANSFORMS):
        tensor = t(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat, _ = model.extract_features(tensor)
        feat = feat[0].cpu().numpy()
        mahal_dists = np.array([
            float((feat - cluster_centers[m_prime]) @ cov_inv @ (feat - cluster_centers[m_prime]))
            for m_prime in range(M)
        ])
        neg_d  = -mahal_dists
        exp_d  = np.exp(neg_d - neg_d.max())
        numer  = exp_d[m_idx] + epsilon
        denom  = exp_d.sum() + M * epsilon
        log_probs.append(np.log(numer / denom))
    return -sum(log_probs)


# ─────────────────────────────────────────────
# FULL PIPELINE — anomaly check + classification
# ─────────────────────────────────────────────
def run_pipeline(pil_image, model, device, anomaly_params):
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        feats, logits = model.extract_features(tensor)
        probs         = torch.softmax(logits, dim=1)[0]
        msp_conf      = probs.max().item()

    result = {
        'is_anomaly': False, 'reason': '',
        'msp_conf': msp_conf, 'goad_score': 0.0,
        'top5': None, 'plant': None, 'condition': None,
        'anomaly_checked': anomaly_params is not None
    }

    if anomaly_params:
        # MSP check
        if msp_conf < anomaly_params['msp_threshold']:
            result.update({
                'is_anomaly': True,
                'reason': f'MSP: confidence too low ({msp_conf*100:.1f}% < {anomaly_params["msp_threshold"]*100:.1f}%)',
            })
            return result

        # GOAD check
        g_score = compute_goad_score(
            pil_image, model, device,
            anomaly_params['cluster_centers'],
            anomaly_params['cov_inv'],
            anomaly_params.get('epsilon', 0.1)
        )
        result['goad_score'] = g_score

        if g_score > anomaly_params['goad_threshold']:
            result.update({
                'is_anomaly': True,
                'reason': f'GOAD: unusual pattern detected (score {g_score:.2f} > threshold {anomaly_params["goad_threshold"]:.2f})',
            })
            return result

    # Normal classification
    top5p, top5i = torch.topk(probs, 5)
    top5 = [{'class': CLASSES[i.item()], 'prob': p.item()}
            for p, i in zip(top5p.cpu(), top5i.cpu())]
    plant, condition = top5[0]['class'].split('___')
    result.update({'top5': top5, 'plant': plant, 'condition': condition})
    return result


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in [('stage',0),('result',None),('image',None),
             ('fname',''),('fsize',0),('imgwh',(0,0))]:
    if k not in st.session_state:
        st.session_state[k] = v

model, device, model_ok, anomaly_params = load_everything()
stage = st.session_state.stage


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
anom_status = "✓ Active" if anomaly_params else "○ Not loaded"
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-bg"></div>
  <div class="hero-tag">🌱 AI-Powered Plant Diagnostics · MSP + GOAD Anomaly Detection</div>
  <h1 class="hero-title">Detect Plant <span>Disease</span> Instantly</h1>
  <p class="hero-sub">EfficientNet-B3 trained on 70,295 images · 38 classes · Classification-based anomaly detection (Bergman & Hoshen, ICLR 2020)</p>
  <div class="hero-stats">
    <div class="stat"><div class="stat-val">38</div><div class="stat-lbl">Classes</div></div>
    <div class="stat"><div class="stat-val">70K+</div><div class="stat-lbl">Training Images</div></div>
    <div class="stat"><div class="stat-val">99%+</div><div class="stat-lbl">Val Accuracy</div></div>
    <div class="stat"><div class="stat-val">MSP+GOAD</div><div class="stat-lbl">Anomaly Detection</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PIPELINE BAR
# ─────────────────────────────────────────────
def nc(i): return "done" if stage>i else ("active" if stage==i else "idle")
steps = [
    (1,"Upload","Leaf photo"),
    (2,"Preprocess","300×300 · Normalize"),
    (3,"MSP+GOAD","Anomaly check"),
    (4,"EfficientNet","B3 inference"),
    (5,"Output","Diagnosis"),
]
html = '<div class="pipe-track">'
for i,(n,lbl,sub) in enumerate(steps):
    html += f'<div class="pipe-node {nc(i)}"><div class="pn-num">{n}</div><div class="pn-label">{lbl}</div><div class="pn-sub">{sub}</div></div>'
    if i < len(steps)-1: html += '<div class="pipe-sep"></div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_diag, tab_insights = st.tabs(["🔬 Diagnosis", "📊 Model Insights"])


# ═════════════════════════════════════════════
# TAB 1 — DIAGNOSIS
# ═════════════════════════════════════════════
with tab_diag:
    uploaded = st.file_uploader(
        "Drop your leaf image here",
        type=["jpg","jpeg","png"],
        label_visibility="visible"
    )
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.session_state.update({
            'image': img, 'fname': uploaded.name,
            'fsize': round(uploaded.size/1024, 1),
            'imgwh': img.size, 'stage': 1, 'result': None
        })
        stage = 1

    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.image:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#2a5a35">
          <div style="font-size:3rem;opacity:0.3;margin-bottom:1rem">🍃</div>
          <div style="font-size:1rem;color:#3a6a45">Upload a leaf image above to begin</div>
          <div style="font-size:0.82rem;color:#2a4a30;margin-top:0.5rem">
            Apple · Grape · Tomato · Potato · Corn · Peach · Cherry · Pepper<br>
            Strawberry · Blueberry · Orange · Soybean · Squash · Raspberry
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        img  = st.session_state.image
        w, h = st.session_state.imgwh

        # ── ROW 1: Image + Preprocessing ─────────────────────────
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
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Final tensor shape</div><div class="step-val">torch.Size([1, 3, 300, 300]) → {str(device).upper()}</div></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── ROW 2: Model + Anomaly + Run ─────────────────────────
        c3, c4 = st.columns(2, gap="large")

        with c3:
            st.markdown(f"""
            <div class="card"><div class="card-title">③ Anomaly Detection · MSP + GOAD</div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">MSP Branch</div><div class="step-val">Max softmax confidence · threshold = {f"{anomaly_params['msp_threshold']*100:.1f}%" if anomaly_params else "not loaded"}</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">GOAD Branch</div><div class="step-val">Mahalanobis distance · M=8 transforms · threshold = {f"{anomaly_params['goad_threshold']:.2f}" if anomaly_params else "not loaded"}</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Decision fusion</div><div class="step-val">Either branch flags → reject as anomaly</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Method</div><div class="step-val">Bergman & Hoshen, ICLR 2020</div></div></div>
            </div>
            <div class="card"><div class="card-title">④ EfficientNet-B3 Model</div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Architecture</div><div class="step-val">EfficientNet-B3 (pretrained ImageNet)</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Classifier head</div><div class="step-val">Dropout(0.3) → Linear(1536 → 38)</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Training</div><div class="step-val">3 epochs · OneCycleLR · lr=0.001 · Adam</div></div></div>
            <div class="pipe-step-row"><div class="step-dot"></div><div><div class="step-key">Device</div><div class="step-val">{str(device).upper()}</div></div></div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="card"><div class="card-title">⑤ Run Pipeline</div>', unsafe_allow_html=True)
            if not model_ok:
                st.error("⚠️ Model file not found.")
                st.info("Expected: plant-disease-model (5).pth")
            else:
                if not anomaly_params:
                    st.warning("⚠️ anomaly_params.pkl not found — running without anomaly detection.")

                btn = st.button("▶  Analyze Leaf", use_container_width=True)
                if btn or st.session_state.result is not None:
                    if st.session_state.result is None:
                        with st.spinner("Running MSP + GOAD + EfficientNet-B3..."):
                            time.sleep(0.2)
                            st.session_state.result = run_pipeline(
                                img, model, device, anomaly_params
                            )
                            st.session_state.stage = 3 if st.session_state.result['is_anomaly'] else 4
                            stage = st.session_state.stage

                    res = st.session_state.result
                    if res:
                        msp_clr  = "#4ade80" if res['msp_conf'] > (anomaly_params['msp_threshold'] if anomaly_params else 0.5) else "#f87171"
                        goad_clr = "#4ade80" if (not anomaly_params or res['goad_score'] < anomaly_params['goad_threshold']) else "#f87171"
                        status_txt = "⚠ ANOMALY DETECTED" if res['is_anomaly'] else "✓ Normal leaf — classified"
                        status_clr = "#f87171" if res['is_anomaly'] else "#4ade80"
                        st.markdown(f"""
                        <div class="pipe-step-row"><div class="step-dot {'err' if res['is_anomaly'] else ''}"></div><div>
                          <div class="step-key {'err' if res['is_anomaly'] else ''}">Status</div>
                          <div class="step-val" style="color:{status_clr};font-weight:600">{status_txt}</div>
                        </div></div>
                        <div class="pipe-step-row"><div class="step-dot {'warn' if res['msp_conf'] < (anomaly_params['msp_threshold'] if anomaly_params else 0) else ''}"></div><div>
                          <div class="step-key">MSP Confidence</div>
                          <div class="step-val" style="color:{msp_clr}">{res['msp_conf']*100:.2f}%</div>
                        </div></div>
                        <div class="pipe-step-row"><div class="step-dot {'err' if res['is_anomaly'] and 'GOAD' in res.get('reason','') else ''}"></div><div>
                          <div class="step-key">GOAD Score (Mahalanobis)</div>
                          <div class="step-val" style="color:{goad_clr}">{res['goad_score']:.4f}</div>
                        </div></div>
                        """, unsafe_allow_html=True)
                        if not res['is_anomaly'] and res['top5']:
                            st.markdown(f"""
                            <div class="pipe-step-row"><div class="step-dot"></div><div>
                              <div class="step-key">Top prediction</div>
                              <div class="step-val">{res['top5'][0]['class']} · {res['top5'][0]['prob']*100:.2f}%</div>
                            </div></div>
                            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── ROW 3: OUTPUT ─────────────────────────────────────────
        if st.session_state.result:
            st.markdown("<hr>", unsafe_allow_html=True)
            res = st.session_state.result

            # ── ANOMALY OUTPUT ─────────────────────────────────
            if res['is_anomaly']:
                st.markdown(f"""
                <div class="anomaly-card">
                  <div class="anomaly-icon">⚠️</div>
                  <div class="anomaly-title">Anomaly Detected</div>
                  <div class="anomaly-reason">{res['reason']}</div>
                  <div class="anomaly-scores">
                    <div class="score-box">
                      <div class="score-num">{res['msp_conf']*100:.1f}%</div>
                      <div class="score-lbl">MSP Confidence</div>
                    </div>
                    <div class="score-box">
                      <div class="score-num">{res['goad_score']:.3f}</div>
                      <div class="score-lbl">GOAD Score</div>
                    </div>
                  </div>
                  <div style="margin-top:1.2rem;font-size:0.82rem;color:#7a3a3a">
                    The image does not match any known plant leaf pattern in the training distribution.<br>
                    Please upload a clear, close-up photo of a plant leaf.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── NORMAL OUTPUT ──────────────────────────────────
            else:
                results = res['top5']
                top     = results[0]
                plant, disease, (sev, desc, treat) = get_info(top['class'])
                conf    = top['prob'] * 100
                is_ok   = 'healthy' in top['class']
                crd     = "result-healthy" if is_ok else ("result-severe" if sev=="high" else "result-moderate")
                bdg     = {"none":"badge-none","medium":"badge-medium","high":"badge-high"}[sev]
                btxt    = {"none":"✓ Healthy","medium":"⚠ Moderate","high":"⛔ Severe"}[sev]

                c5, c6 = st.columns(2, gap="large")

                with c5:
                    msp_ok_pct   = res['msp_conf']*100
                    goad_thresh  = anomaly_params['goad_threshold'] if anomaly_params else 0
                    goad_pct_of  = (res['goad_score']/goad_thresh*100) if goad_thresh else 0

                    st.markdown(f"""
                    <div class="card-title">⑤ Diagnosis Output</div>
                    <div class="result-hero {crd}">
                      <div class="r-glow"></div>
                      <div class="r-plant">{plant.replace('_',' ').replace('(','').replace(')','')}</div>
                      <div class="r-disease">{disease}</div>
                      <div style="margin:0.7rem 0">
                        <span class="r-conf-num">{conf:.1f}</span>
                        <span class="r-conf-unit">% confidence</span>
                      </div>
                      <span class="badge {bdg}">{btxt}</span>
                    </div>
                    <div class="scores-row">
                      <div class="score-pill">
                        <div class="score-pill-lbl">MSP Score</div>
                        <div class="score-pill-val">{msp_ok_pct:.1f}%</div>
                      </div>
                      <div class="score-pill">
                        <div class="score-pill-lbl">GOAD Score</div>
                        <div class="score-pill-val {'warn' if goad_pct_of > 70 else ''}">{res['goad_score']:.3f}</div>
                      </div>
                      <div class="score-pill">
                        <div class="score-pill-lbl">Anomaly</div>
                        <div class="score-pill-val">Not detected</div>
                      </div>
                    </div>
                    <div class="info-panel" style="margin-top:0.6rem"><div class="info-key">About this condition</div><div class="info-val">{desc}</div></div>
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
            if st.button("🔄  Analyze another image", use_container_width=False):
                for k, v in [('stage',0),('result',None),('image',None),('fname',''),('fsize',0),('imgwh',(0,0))]:
                    st.session_state[k] = v
                st.rerun()


# ═════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═════════════════════════════════════════════
with tab_insights:
    st.markdown("""
    <div class="sec-divider">
      <div class="sec-divider-line"></div>
      <div class="sec-divider-txt">Model Performance</div>
      <div class="sec-divider-line"></div>
    </div>""", unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    metrics = [
        (col_a, "Validation Accuracy", "99%+"),
        (col_b, "F1-Score (weighted)", "~0.99"),
        (col_c, "Training Epochs", "3"),
        (col_d, "Anomaly Method", "MSP+GOAD"),
    ]
    for col, lbl, val in metrics:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl" style="margin-top:0.3rem">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix + per-class F1 ──────────────────────────
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown('<div class="card"><div class="card-title">Confusion Matrix (38 classes)</div>', unsafe_allow_html=True)
        cm_path = 'Classification-based Anomaly Detection/confusion_matrix (4).png'
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("confusion_matrix.png not found in app directory")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="card"><div class="card-title">Per-class F1 Score</div>', unsafe_allow_html=True)
        f1_path = 'Classification-based Anomaly Detection/per_class_f1 (1).png'
        if os.path.exists(f1_path):
            st.image(f1_path, use_container_width=True)
        else:
            st.info("per_class_f1.png not found in app directory")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-divider">
      <div class="sec-divider-line"></div>
      <div class="sec-divider-txt">Training Analysis</div>
      <div class="sec-divider-line"></div>
    </div>""", unsafe_allow_html=True)

    # ── Overfitting + class dist ──────────────────────────────────
    c_ov, c_dist = st.columns(2, gap="large")

    with c_ov:
        st.markdown('<div class="card"><div class="card-title">Overfitting Analysis</div>', unsafe_allow_html=True)
        ov_path = 'Classification-based Anomaly Detection/overfitting_analysis (5).png'
        if os.path.exists(ov_path):
            st.image(ov_path, use_container_width=True)
        else:
            st.info("overfitting_analysis.png not found")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_dist:
        st.markdown('<div class="card"><div class="card-title">Class Distribution</div>', unsafe_allow_html=True)
        dist_path = 'Classification-based Anomaly Detection/class_distribution (2).png'
        if os.path.exists(dist_path):
            st.image(dist_path, use_container_width=True)
        else:
            st.info("class_distribution.png not found")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-divider">
      <div class="sec-divider-line"></div>
      <div class="sec-divider-txt">Anomaly Detection Thresholds</div>
      <div class="sec-divider-line"></div>
    </div>""", unsafe_allow_html=True)

    # ── MSP + GOAD distributions ──────────────────────────────────
    c_msp, c_goad = st.columns(2, gap="large")

    with c_msp:
        st.markdown('<div class="card"><div class="card-title">MSP Score Distribution</div>', unsafe_allow_html=True)
        msp_path = 'Classification-based Anomaly Detection/msp_distribution (1).png'
        if os.path.exists(msp_path):
            st.image(msp_path, use_container_width=True)
        else:
            st.info("msp_distribution.png not found")
        if anomaly_params:
            st.markdown(f"""
            <div class="info-panel">
              <div class="info-key">Threshold</div>
              <div class="info-val">{anomaly_params['msp_threshold']*100:.2f}% — images below this are rejected as anomalies</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_goad:
        st.markdown('<div class="card"><div class="card-title">GOAD Score Distribution (Mahalanobis)</div>', unsafe_allow_html=True)
        goad_path = 'Classification-based Anomaly Detection/goad_distribution (1).png'
        if os.path.exists(goad_path):
            st.image(goad_path, use_container_width=True)
        else:
            st.info("goad_distribution.png not found")
        if anomaly_params:
            st.markdown(f"""
            <div class="info-panel">
              <div class="info-key">Threshold</div>
              <div class="info-val">{anomaly_params['goad_threshold']:.4f} — images above this are rejected · M={anomaly_params.get('M',8)} transforms</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Anomaly params summary ────────────────────────────────────
    if anomaly_params:
        st.markdown("""
        <div class="sec-divider">
          <div class="sec-divider-line"></div>
          <div class="sec-divider-txt">Anomaly Detection Config</div>
          <div class="sec-divider-line"></div>
        </div>""", unsafe_allow_html=True)

        ca, cb, cc, cd = st.columns(4)
        for col, lbl, val in [
            (ca, "MSP Threshold", f"{anomaly_params['msp_threshold']*100:.2f}%"),
            (cb, "GOAD Threshold", f"{anomaly_params['goad_threshold']:.4f}"),
            (cc, "M Transforms",  str(anomaly_params.get('M',8))),
            (cd, "Distance Metric", anomaly_params.get('distance_metric','mahalanobis').title()),
        ]:
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center">
                  <div class="stat-val" style="font-size:1.3rem">{val}</div>
                  <div class="stat-lbl" style="margin-top:0.3rem">{lbl}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div class="footer-tech">
    EfficientNet-B3 · PlantVillage · 70,295 images · 38 classes · 3 epochs · OneCycleLR · Adam · lr=0.001<br>
    Anomaly Detection: MSP + GOAD (Mahalanobis) · Bergman & Hoshen, ICLR 2020
  </div>
  <div class="footer-links">
    <a class="footer-link" href="https://github.com/moad-cod/-Plant-Disease-Classification-using-CNN" target="_blank">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      GitHub
    </a>
    <a class="footer-link" href="https://x.com/MouadEl_AI" target="_blank">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
      @MouadEl_AI
    </a>
  </div>
  <div class="footer-copy">Built by Mouad Elbaz · © 2024</div>
</div>
""", unsafe_allow_html=True)