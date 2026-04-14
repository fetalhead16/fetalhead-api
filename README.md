# Fetal Head Biometrics Web App

This project turns the paper in this folder into a working web application. It accepts an ultrasound image upload, detects the fetal head region, fits an ellipse, and returns the core biometric values from the paper:

- `HC` - Head circumference
- `BPD` - Biparietal diameter
- `OFD` - Occipitofrontal diameter
- `CI` - Cephalic index
- `HA` - Head area

The current build is a research-demo MVP. The paper describes a trained lightweight U-Net and a Random Forest classifier, but no trained weight files were provided in the project folder. Because of that, this app ships with:

- A working classical image-processing fallback for segmentation and ellipse fitting
- A pluggable classifier hook that automatically uses `models/random_forest.joblib` if you add it later
- A heuristic screening summary when trained weights are not present

## Stack

- Backend: FastAPI
- Frontend: HTML, CSS, vanilla JavaScript
- Image pipeline: OpenCV, NumPy, Pillow

## Run Locally

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the app:

```powershell
uvicorn app.main:app --reload
```

5. Open `http://127.0.0.1:8000`.

## Notes About Measurements

- If you upload a DICOM file and it contains `PixelSpacing`, the app uses it automatically.
- If you upload a JPG or PNG, enter `Pixel spacing (mm/pixel)` in the UI if you want values in millimeters.
- If no spacing is available, the app still calculates the values, but they will be shown in pixels instead of millimeters.

## Add The Trained Classifier Later

If you later export the paper's Random Forest model, place the files here:

```text
models/random_forest.joblib
models/feature_scaler.joblib
```

The backend will detect them automatically and switch from heuristic screening to the trained classifier.

## Hostinger Deployment

There are two realistic deployment paths depending on your Hostinger plan:

### Option 1: Hostinger VPS

1. Push this project to GitHub or upload it to the VPS.
2. Install Docker on the VPS.
3. Build and run:

```bash
docker build -t fetal-head-app .
docker run -d --name fetal-head-app -p 8000:8000 fetal-head-app
```

Or with Compose:

```bash
docker compose up -d --build
```

For a real domain deployment with HTTPS on `fetalhead.in`, use the production files in this repo:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

This starts:

- the FastAPI app
- Caddy as a reverse proxy on ports `80` and `443`

Caddy will request and renew HTTPS certificates automatically once DNS is pointed to the VPS and ports `80` and `443` are reachable.

4. Put Nginx in front of it and point your Hostinger domain A record to the VPS IP.
5. Add SSL with Let's Encrypt.

### Option 2: Hostinger Domain Only or Shared Hosting

Shared hosting is often not a good fit for a Python API like this.

1. Deploy this FastAPI app to a Python-friendly host such as Railway, Render, or a VPS.
2. Keep the domain on Hostinger.
3. Point the domain or subdomain DNS record to the deployed backend.

Example:

- `app.yourdomain.com` -> FastAPI app
- `yourdomain.com` -> same app or a static landing page

## Test

```powershell
python -m unittest discover -s tests
```

## Render Deployment

If you deploy on Render as a native Python web service:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`

The repo also includes:

- `.python-version` to pin Python
- `render.yaml` if you later prefer Blueprint deployment
