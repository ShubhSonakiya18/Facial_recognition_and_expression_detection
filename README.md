# VisionX — Multi-Tenant Facial Recognition & Emotion Detection Platform

A production-ready, multi-tenant AI platform that performs real-time facial
recognition and emotion detection via image upload, live webcam feed, and
snapshots — with full role-based access control, organisation management,
and a React-based admin dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Face Recognition | FaceNet (via DeepFace) + Calibrated SVM + cosine similarity gate |
| Emotion Detection | DeepFace with custom correction rules |
| Backend API | FastAPI (async) |
| Database | PostgreSQL (pgvector) |
| Caching / Token Blacklist | Redis |
| Cloud Storage | Google Cloud Storage |
| Frontend | React + Vite + Recharts |
| Auth | JWT (access + refresh tokens) |
| Email | SMTP (Gmail) for invite emails |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
Facial_recognition_and_expression_detection/
│
├── config/
│   └── settings.py            ← All config values (env-driven)
│
├── pipelines/                  ← AI pipeline stages
│   ├── data_ingestion.py       ← Download dataset from GCS
│   ├── preprocessing.py        ← Clean + validate images
│   ├── feature_extraction.py   ← FaceNet embeddings + cache
│   ├── training.py             ← SVM train + evaluate + save
│   ├── inference.py            ← Static image prediction
│   ├── realtime.py             ← Live webcam / snapshot prediction
│   └── deployment.py           ← Upload trained model to GCS
│
├── models/
│   ├── emotion_model.py        ← infer_emotion(), correction rules
│   └── face_model.py           ← FaceRecognizer (SVM + cosine gate)
│
├── storage/
│   └── gcs_storage.py          ← GCS upload/download abstraction
│
├── api/
│   ├── app.py                  ← FastAPI app, /predict endpoints, system routes
│   ├── models.py               ← SQLAlchemy ORM models
│   ├── email_service.py        ← SMTP invite emails
│   ├── dependencies.py         ← Auth guards (get_current_user, require_role)
│   ├── auth/
│   │   ├── router.py           ← /auth/signup, login, invite, refresh, logout
│   │   ├── schemas.py          ← Pydantic request/response models
│   │   └── service.py          ← JWT + password hashing + Redis blacklist
│   └── routers/
│       ├── users.py            ← User management
│       ├── organisations.py    ← Organisation management
│       ├── persons.py          ← Enrolled persons management
│       ├── sessions.py         ← Detection session history
│       └── audit.py            ← Audit log endpoints
│
├── db/
│   ├── base.py                 ← SQLAlchemy async engine/session
│   └── schema.sql              ← Full multi-tenant database schema
│
├── frontend/
│   └── src/
│       ├── App.jsx             ← Router + role-based redirects
│       ├── AuthContext.jsx     ← Global auth state
│       ├── api.js              ← All backend API calls
│       ├── components/
│       │   ├── AppShell.jsx    ← Navbar/header
│       │   └── ProtectedRoute.jsx
│       └── pages/
│           ├── HomePage.jsx
│           ├── LoginPage.jsx           ← Login + first-time Super Admin setup
│           ├── SignupInvitePage.jsx    ← Invite-based signup
│           ├── UploadPage.jsx          ← Image upload detection
│           ├── LivePage.jsx            ← Live webcam + snapshot detection
│           ├── UserSessionsPage.jsx    ← Member's own session history
│           ├── SuperAdminDashboard.jsx ← System-wide KPI dashboard
│           └── OrgAdminDashboard.jsx   ← Organisation-level dashboard
│
├── saved_models/                ← Trained model artefacts (synced with GCS)
├── secrets/                     ← gcs_key.json (gitignored)
├── docker-compose.yml
├── Dockerfile                   ← Backend image
├── frontend/Dockerfile          ← Frontend image
├── main.py                      ← CLI orchestrator (train / infer / api)
└── requirements.txt
```

---

## Multi-Tenant Architecture

The platform supports multiple organisations ("tenants") on a single
deployment, each fully isolated:

- **super_admin** — manages all organisations, all users, system-wide KPIs
- **org_admin** — manages their own organisation's team members, enrolled
  persons, and sessions
- **member** — runs detections (upload/live/snapshot) and views their own
  session history

Roles, permissions, organisations, users, enrolled persons, sessions, and
audit logs are all stored in PostgreSQL with full audit columns
(`created_at/by/ip`, `updated_at/by/ip`, `deleted_at/by/ip`, `is_active`,
`is_deleted`).

---

## Detection Methods

Every detection run is recorded as a **session** tagged with its `source`:

| Source | Description |
|--------|-------------|
| `upload` | Static image uploaded via the Upload page |
| `live` | Real-time webcam frame from the Live page |
| `snapshot` | Manual snapshot captured from the live feed |

### How Detection Works

1. **Face Detection** — multi-backend cascade (OpenCV, RetinaFace, MTCNN),
   with a quality gate that rejects faces that are too small, too dark, too
   blurry, or too large.
2. **Face Recognition** — FaceNet extracts a 512-d embedding per face. A
   dual-gate strategy determines identity:
   - SVM probability ≥ `SVM_CONFIDENCE_THRESHOLD` (0.70)
   - Cosine similarity to stored mean embedding ≥ `COSINE_SIMILARITY_THRESHOLD` (0.50)
   - Both must pass, otherwise labelled `UNKNOWN`.
3. **Emotion Detection** — DeepFace produces raw scores for 7 emotions
   (happy, sad, angry, neutral, surprise, fear, disgust), refined by custom
   correction rules to fix known DeepFace misclassifications (e.g.
   fear→surprise confusion, disgust→angry confusion).
4. **Real-time mode** — adds a vote buffer over recent frames to prevent
   emotion labels flickering between frames.

---

## Setup — Docker (Recommended)

### 1. Clone the repository

```bash
git clone <repo-url>
cd Facial_recognition_and_expression_detection
```

### 2. Configure environment files

Two `.env` files are required — `frontend/.env` and a root `.env`. These
are gitignored and never committed. Ask a team member with existing access
for the current values, or refer to `.env.example` if present.

Required variables (root `.env`):
- Database and Redis connection strings
- JWT secret and token expiry settings
- SMTP credentials for invite emails
- GCS project, bucket name, and key path
- Frontend URL and logging configuration

Required variables (`frontend/.env`):
- API key
- Max upload size
- Frontend URL

### 3. Add credentials

Place your GCS service-account JSON at `secrets/gcs_key.json`.

> `secrets/` and `.env` files are gitignored — never commit credentials.

### 4. Build and start all services

```bash
docker compose up --build -d
```

This starts:

| Service | Container | Port |
|---------|-----------|------|
| Backend API | face-emotion-api | 8000 |
| Frontend | face-emotion-ui | 3000 |
| PostgreSQL | face-emotion-db | 5432 |
| Redis | face-emotion-redis | 6379 |
| pgAdmin | face-emotion-pgadmin | 5050 |

### 5. Load the database schema

Wait ~30 seconds for PostgreSQL to become healthy, then:

```bash
Get-Content db\schema.sql | docker exec -i face-emotion-db psql -U faceapp -d face_emotion_db
```

### 6. First-time setup

Open `http://localhost:3000` — you'll see a first-time setup screen to
create your organisation and Super Admin account. This can only be done
once per fresh database.

---

## Running on a Network (Inviting Teammates)

The frontend dynamically detects the host it's accessed from
(`window.location.hostname`), so:

- Open `http://localhost:3000` for yourself — works on any WiFi.
- To invite a teammate, open the dashboard using your current WiFi IP
  (e.g. `http://192.168.x.x:3000`) — the invite link automatically uses
  that IP. The teammate must be on the same network to use it.

Invite emails are sent via SMTP (Gmail) and work for any email address.

---

## Common Docker Commands

| Change made | Command |
|-------------|---------|
| Frontend code (`frontend/src/...`) | `docker compose build --no-cache frontend && docker compose up -d` |
| Backend Python code (`api/...`) | `docker compose restart backend` |
| New database table added to `schema.sql` | `Get-Content db\schema.sql \| docker exec -i face-emotion-db psql -U faceapp -d face_emotion_db` |
| New column added to existing table | `docker exec -i face-emotion-db psql -U faceapp -d face_emotion_db -c "ALTER TABLE <table> ADD COLUMN IF NOT EXISTS <col> <type>;"` |
| Full reset (wipes all data) | `docker compose down -v && docker compose up -d` then reload schema |

---

## Setup — Local Python (Training / CLI only)

For running the AI pipelines directly without Docker (e.g. retraining the
model):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Train the model

```bash
python main.py train
```

Runs all pipeline stages: data ingestion → preprocessing → feature
extraction → SVM training → deployment to GCS.

```bash
python main.py train --force-retrain --skip-deploy
```

### Run inference on a photo

```bash
python main.py infer --image path/to/photo.jpg
```

### Run real-time webcam (local)

```bash
python main.py realtime
```

Controls: `q` to quit, `s` to save snapshot.

---

## API Reference

The backend exposes a full REST API covering authentication, user and
organisation management, enrolled persons, detection sessions, and audit
logs, in addition to the core `/predict/image` and `/predict/base64`
detection endpoints.

Interactive documentation with all available endpoints, request/response
schemas, and the ability to test calls directly is available at:

```
http://localhost:8000/docs
```

---

## Database Schema Overview

| Table | Purpose |
|-------|---------|
| `master_roles` | super_admin / org_admin / member role definitions |
| `master_actions` | Available permission actions |
| `permissions` | Role → action mapping |
| `organisations` | Tenant/organisation records |
| `users` | All users, linked to an organisation and role |
| `password_reset_logs` | Password reset history |
| `persons` | Enrolled face dataset entries per organisation |
| `sessions` | Every detection run (upload/live/snapshot) with results |
| `audit_logs` | Action audit trail (invites, role changes, deletions, etc.) |

All core tables include audit columns: `is_active`, `is_deleted`,
`created_at/by/ip`, `updated_at/by/ip`, `deleted_at/by/ip`.

---

## Security Notes

- Cloud storage credentials, JWT secrets, and SMTP credentials are kept
  only in gitignored local files — never hardcoded or committed.
- Passwords are hashed before storage.
- Invite tokens expire after 48 hours.
- Refresh tokens and logout are backed by Redis blacklisting.

---

## Logging

| Output | Format | Purpose |
|--------|--------|---------|
| Console | Coloured human-readable | Development |
| `logs/app.log` | Plain-text rotating | Ops / debugging |
| `logs/app.jsonl` | One JSON object per line | Machine ingestion (ELK, Datadog, Cloud Logging) |

Log level controlled by `LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).