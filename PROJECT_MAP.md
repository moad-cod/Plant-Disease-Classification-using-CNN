# PROJECT

**Plant-Disease-Classification-using-CNN**

A deep learning-based plant disease classification system using EfficientNet-B3 with MSP + GOAD anomaly detection, deployed via a Streamlit web UI. Trained on the PlantVillage dataset (70K+ images, 38 classes). This repository contains only the **inference/deployment** layer — the training pipeline (notebooks) is not included.

- Author: Mouad Elbaz
- GitHub: https://github.com/moad-cod/-Plant-Disease-Classification-using-CNN

---

# STACK

| Category       | Technology                        |
|----------------|-----------------------------------|
| Language       | Python 3.12                       |
| Deep Learning  | PyTorch 2.12.1, torchvision 0.27.1 |
| Web UI         | Streamlit 1.58.0                  |
| Image          | Pillow 12.2.0                     |
| Numerical      | NumPy 2.5.0                       |
| GPU Support    | CUDA 13.x, cuDNN 9.20             |
| Model Arch     | EfficientNet-B3                   |
| Anomaly Det.   | MSP (Max Softmax Probability) + GOAD (Mahalanobis distance) |
| Version Ctrl   | Git, DVC (initialized, unused)    |
| IDE Config     | PyCharm/IntelliJ (.idea/)         |

---

# FOLDER_STRUCTURE

```
project-root/
├── .dvc/                          # DVC metadata (only .gitignore, no config)
├── .git/                          # Git repository data
├── .gitignore                     # 214-line Python gitignore
├── .idea/                         # PyCharm project files (tracked in git)
├── .venv/                         # Python 3.12 virtual env (2GB+, on disk only)
├── README.md                      # Project documentation (79 lines)
└── src/
    ├── app.py                     # Streamlit application (813 lines, dark theme)
    └── Classification-based Anomaly Detection/
    │   ├── __init__.py            # Public API exports
    │   ├── tokens.py              # CSS design system (~680 lines)
    │   ├── base.py                # Reusable HTML generators
    │   ├── layout.py              # Navbar, Hero, Footer
    │   ├── dashboard.py           # Upload, prediction, analytics
    │   ├── model_panel.py         # Model info, charts, disease library
    │   └── about.py               # About page
    └── Classification-based Anomaly Detection/
        ├── anomaly_params.pkl              # GOAD+MSP thresholds & cluster params (19 MB)
        ├── confusion_matrix.png            # 38-class confusion matrix (358 KB)
        ├── class_distribution.png          # Per-class sample count (68 KB)
        ├── goad_distribution .png          # GOAD score distribution (33 KB)
        ├── msp_distribution.png            # MSP score distribution (34 KB)
        ├── overfitting_analysis .png       # Train/val loss curves (64 KB)
        ├── per_class_f1.png                # Per-class F1 bar chart (79 KB)
        ├── plant-disease-model-complete.pth # Full checkpoint (42 MB, unused)
        └── plant-disease-model.pth          # Inference-only model (42 MB)
```

---

# ARCHITECTURE

## Standalone Streamlit App (`src/app.py`)

The application is a single-file Streamlit app (813 lines) with a dark theme design. The business logic (model definition, inference pipeline, anomaly detection) is co-located with the UI rendering in `src/app.py`. No module decomposition is used.

## Layers Identified

| Layer            | Present | Description                                    |
|------------------|---------|------------------------------------------------|
| Presentation    | ✓       | Premium SaaS UI (Inter font, glass nav, gradient blobs, conf ring, analytics cards, timeline) |
| Business Logic  | ✓       | Inference pipeline, anomaly detection logic    |
| AI/ML Layer     | ✓       | EfficientNet-B3, MSP, GOAD                     |
| Data Layer      | ✓       | Hardcoded classes, disease knowledge base      |
| Infrastructure  | ✗       | No Docker, no CI/CD, no deployment config      |
| Database        | ✗       | No database — all data is in-memory or files   |
| API Layer       | ✗       | No REST/gRPC API — Streamlit direct            |
| Shared/Core     | ✗       | Single-file app (no module decomposition)      |
| Configuration   | ✗       | No config files — hardcoded paths              |
| Logging         | ✓       | Python logging via `log.info/error/warning`    |
| Testing         | ✗       | Zero tests                                     |
| State Mgmt      | ✓       | Streamlit session state                        |
| Dependency Inj  | ✗       | Not used                                       |

---

# SYSTEM_FLOW

## Application Startup
1. Streamlit runs `app.py`
2. `st.set_page_config()` configures page
3. CSS is injected inline
4. `load_everything()` is called (cached):
   - Detects device (CUDA > CPU)
   - Instantiates `EfficientNetB3(38)`
   - Loads `.pth` weights from disk
   - Loads `anomaly_params.pkl` from disk
   - Returns model, device, status flags
5. Session state initialized (stage=0, result=None)
6. Hero section + pipeline bar + tabs rendered

## Inference Request Lifecycle
```
User uploads image
        │
        ▼
st.file_uploader → PIL.Image.open() → session_state
        │
        ▼
"Analyze Leaf" button clicked
        │
        ▼
run_pipeline():
  │
  ├─ TRANSFORM: Resize(300,300) → ToTensor → Normalize(ImageNet)
  │
  ├─ model.extract_features(tensor)
  │    ├─ features (1280-dim)  ← used by GOAD
  │    └─ logits (38-dim)      ← used by MSP + classification
  │
  ├─ MSP: softmax(logits).max() < threshold? → ANOMALY
  │
  ├─ GOAD: 8 transforms → Mahalanobis distance > threshold? → ANOMALY
  │
  └─ Normal: top-5 classes → lookup disease info → display
        │
        ▼
Result rendered in UI (hero card, top-5 bars, treatment)
```

## Data Flow
- **Input**: Image file (JPG/PNG) → PIL Image (RGB)
- **Transform**: 300×300 → [0,1] tensor → ImageNet normalization
- **Model**: EfficientNet-B3 → 1280-dim features + 38 logits
- **Anomaly**: MSP threshold test → GOAD Mahalanobis test
- **Output**: Classification result + disease info + treatment

## Authentication Flow
None. The application has no authentication mechanism.

## Error Flow
- Model file missing: `st.error()` shows "Model file not found." + `log.error()`
- Anomaly params missing: `st.warning()` shows warning + `log.warning()`; inference proceeds without anomaly detection.
- Model/anomaly load failure: Caught via `except Exception as e`, logged via `log.error()` + streamlit warning.
- Image decode failure: Streamlit native error.
- No explicit error boundaries for inference failures (e.g., GPU OOM).

---

# DATABASE

No database is used. All data is:
- **In-memory**: session state, model weights (RAM)
- **On-disk files**: `.pth` model weights, `.pkl` anomaly params, `.png` evaluation plots
- **Hardcoded**: 38 class names, 21 disease knowledge entries in `app.py`

---

# CONFIGURATION

No configuration files exist. Notable hardcoded values:

| Item              | Value                        | Location     |
|-------------------|------------------------------|--------------|
| Model path        | `BASE_DIR/Classification-based Anomaly Detection/plant-disease-model.pth` | app.py:510 |
| Anomaly params    | `BASE_DIR/Classification-based Anomaly Detection/anomaly_params.pkl`      | app.py:526 |
| Image size        | 300 × 300                    | app.py:462-469, 531 |
| Model arch        | EfficientNet-B3              | app.py:480 |
| Num classes       | 38                           | app.py:477 |
| Dropout           | 0.3                          | app.py:483 |
| Norm mean/std     | ImageNet ([0.485, ...])      | app.py:534 |
| GOAD transforms   | 8 (M=8)                      | app.py:461-471 |
| MSP threshold     | From anomaly_params.pkl      | app.py:580 |
| GOAD threshold    | From anomaly_params.pkl      | app.py:596 |
| Epsilon (GOAD)    | 0.1                          | app.py:592 |

---

# DEPENDENCIES

## Runtime (used in app.py)
| Package     | Min. Version | Purpose                  |
|-------------|-------------|--------------------------|
| streamlit   | ≥1.20       | Web UI framework         |
| torch       | ≥2.0        | Deep learning engine     |
| torchvision | ≥0.15       | Model + transforms       |
| Pillow      | ≥10.0       | Image loading            |
| numpy       | ≥1.24       | Array operations         |

## Training / Development (not in app.py, used for generating artifacts)
| Package     | Purpose                    |
|-------------|----------------------------|
| pandas      | Data analysis              |
| matplotlib  | Evaluation plots           |
| seaborn     | Statistical visualizations |
| scikit-learn| Metrics (F1, confusion matrix) |

---

# ENTRY_POINTS

| Entry Point | Command                          | Description            |
|-------------|----------------------------------|------------------------|
| Application | `streamlit run src/app.py`       | Start Streamlit server |

---

# LOGGING

- **Python logging configured** (`import logging` in `app.py:12`).
- `logging.basicConfig(level=INFO)` set at module level (`app.py:14`).
- Model loading success/failure logged via `log.info()` / `log.error()`.
- Anomaly params loading success/failure logged via `log.info()` / `log.error()` / `log.warning()`.
- The only user-facing feedback is through Streamlit's `st.error()`, `st.warning()`, and `st.info()`.
- Inference timing is done with `time.sleep(0.2)` (mock) — no real instrumentation.

---

# TESTS

**No tests exist.** No test files, test directory, or test framework configuration is present in the repository.

---

# DEPLOYMENT

- **Docker**: Claimed in README but no Dockerfile, docker-compose.yml, or .dockerignore exists.
- **CI/CD**: None configured (no GitHub Actions, GitLab CI, or other pipeline).
- **Model artifacts**: 42 MB `.pth` files tracked in Git (not using Git LFS or DVC properly).
- **Environment**: README mentions Kaggle for training; inference runs via `streamlit run`.

---

# KNOWN_ISSUES

## Critical

1. ~~**Hardcoded absolute paths** — Fixed: now uses `BASE_DIR` from `os.path.dirname(__file__)`.~~
2. ~~**Filename mismatch: anomaly_params** — Fixed: path uses `anomaly_params.pkl`.~~
3. ~~**Filename mismatches: all insight images** — Fixed: all 6 insight paths corrected.~~
4. ~~**Bare `except: pass`** — Replaced with specific exception logging.~~
5. **No Dockerfile** — README claims Docker support but no container files exist.

## Moderate

6. **No input validation** — Uploaded images not validated for type/size/content.
7. **No type hints** — Zero type annotations on functions.
8. **Magic numbers** — Image size (300), transforms count (8), epsilon (0.1) hardcoded.
9. ~~**Session state reset bug** — `result` was unconditionally wiped on rerun, causing "Analyze Leaf" to appear broken~~ — Fixed: `result` only cleared when a genuinely new file is uploaded.
10. **Extra spaces in filenames** — `goad_distribution .png` and `overfitting_analysis .png` have a trailing space before the extension on disk.
12. ~~**Hardcoded absolute paths + filename mismatch + bare `except:pass`** — Regressed in a file overwrite, then re-fixed: paths now use `BASE_DIR` dynamically, filenames corrected to `plant-disease-model.pth` and `anomaly_params.pkl`, bare excepts replaced with `log.error()`.~~

---

# ORPHANS

| File | Size | Status | Notes |
|------|------|--------|-------|
| `plant-disease-model-complete.pth` | 42 MB | Unused | Never referenced in `app.py`; likely full training checkpoint |
| `.dvc/` | — | Unused | DVC initialized but no config or remotes configured |
| `.idea/` | — | Tracked in Git | IDE config should be gitignored |

---

# PENDING

- [x] Generate and commit `requirements.txt` or `pyproject.toml`
- [x] Fix hardcoded absolute paths → use `os.path.dirname(__file__)` relative paths (regressed, re-fixed 2026-06-27)
- [x] Fix all filename mismatches in `app.py` to match actual disk filenames (regressed, re-fixed 2026-06-27)
- [x] Add Python logging (`import logging`) (regressed, re-fixed 2026-06-27)
- [x] Replace bare `except: pass` with specific exception handling (regressed, re-fixed 2026-06-27)
- [x] Complete UI/UX redesign (superseded — file overwritten with dark theme version)
- [x] Refactor from monolithic to component-based architecture (superseded — components/ removed)
- [x] Fix image preview sizing, session state reset on rerun, processing guard + prediction logging (superseded — overwritten)
- [x] Fix regressed model/anomaly paths, logging, and bare except clauses (2026-06-27)
- [ ] Create Dockerfile + docker-compose.yml
- [ ] Set up CI/CD pipeline (GitHub Actions recommended)
- [ ] Add input validation for uploaded images
- [ ] Add `.idea/` to `.gitignore` and remove from Git tracking
- [ ] Add type hints to all function signatures
- [ ] Add unit/integration tests
- [ ] Configure DVC properly or remove `.dvc/` directory
- [ ] Either configure Git LFS for model files or move to external storage
- [ ] Add setup/installation instructions to README

---

*Last updated: 2026-06-27 — Fixed regressed model/anomaly paths, re-added logging, replaced bare excepts.*
