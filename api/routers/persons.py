"""
api/routers/persons.py
───────────────────────
Person (face enrolment) management endpoints.

A "Person" is someone whose face is enrolled in the recognition dataset.
This is separate from a "User" (someone who logs into the app).

For example in Netsmartz:
  - User = employee who logs in to scan faces
  - Person = any team member whose face is in the dataset (may or may not have a login)

Who can do what
---------------
super_admin → full access across all orgs
org_admin   → create / view / delete persons within their own org only
user        → no access (403)

GCS Note
--------
Photo upload to GCS is NOT implemented yet (Phase 3).
The `gcs_path` and `photo_count` fields are tracked in the DB now,
and will be populated once GCS upload is wired up later.
The endpoint POST /persons/{id}/mark-dataset is a placeholder that lets
you manually mark that a dataset exists (for testing), until GCS is live.

Endpoints
---------
GET    /persons                      → list persons (scoped by role)
GET    /persons/{id}                 → get one person
POST   /persons                      → create a person record
PATCH  /persons/{id}                 → update person details
PATCH  /persons/{id}/mark-dataset    → manually set gcs_path + photo_count (placeholder until GCS)
PATCH  /persons/{id}/mark-enrolled   → mark person as enrolled in the model
DELETE /persons/{id}                 → soft-delete a person (is_active = False)
"""

from uuid import UUID
from typing import Optional, List
import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_role
from api.models import Person, User
import shutil
import logging
from db.base import get_db
from config.settings import TEAM_FACES_DIR, GCS_BUCKET_NAME, AUTO_RETRAIN_ON_DELETE
from utils.utils import sanitize_name


router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PersonOut(BaseModel):
    id          : str
    org_id      : Optional[str]
    full_name   : str
    employee_id : Optional[str]
    department  : Optional[str]
    gcs_path    : Optional[str]
    photo_count : int
    is_enrolled : bool
    is_active   : bool
    added_by    : Optional[str]
    created_at  : str

    model_config = {"from_attributes": True}


class CreatePersonRequest(BaseModel):
    full_name   : str
    employee_id : Optional[str] = None
    department  : Optional[str] = None


class UpdatePersonRequest(BaseModel):
    full_name   : Optional[str] = None
    employee_id : Optional[str] = None
    department  : Optional[str] = None


class MarkDatasetRequest(BaseModel):
    """
    Placeholder until GCS is wired up.
    Lets org_admin manually record that a dataset was uploaded.
    """
    gcs_path    : str
    photo_count : int


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _person_out(p: Person) -> dict:
    return {
        "id"         : str(p.id),
        "org_id"     : str(p.org_id) if p.org_id else None,
        "full_name"  : p.full_name,
        "employee_id": p.employee_id,
        "department" : p.department,
        "gcs_path"   : p.gcs_path,
        "photo_count": p.photo_count,
        "is_enrolled": p.is_enrolled,
        "is_active"  : p.is_active,
        "added_by"   : str(p.created_by) if p.created_by else None,
        "created_at" : p.created_at.isoformat(),
    }


async def _get_person_or_404(person_id: str, db: AsyncSession) -> Person:
    try:
        pid = UUID(person_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid person ID format.")
    person = await db.get(Person, pid)
    if not person:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Person not found.")
    return person


def _check_person_access(current_user: User, person: Person) -> None:
    """org_admin can only access persons within their own org."""
    if (
        current_user.role == "org_admin"
        and person.org_id != current_user.org_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You can only access persons within your own organisation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

log = logging.getLogger(__name__)


async def _sync_local_folders_to_db(db: AsyncSession, current_user: User) -> None:
    """
    Scan TEAM_FACES_DIR for person folders and ensure each one has a
    corresponding Person record in the database.  This keeps the
    Enrolled Persons page in sync with data downloaded from GCS.
    """
    if not TEAM_FACES_DIR.exists():
        return

    # Fetch ALL persons (including inactive) so we don't re-create deleted ones
    result = await db.execute(select(Person))
    db_persons = result.scalars().all()

    # Build lookup: sanitized name → Person
    existing = {sanitize_name(p.full_name): p for p in db_persons}

    changed = False
    for folder in TEAM_FACES_DIR.iterdir():
        if not folder.is_dir():
            continue
        # Skip nested "team_faces" artifact from zip extraction
        if folder.name.lower() == "team_faces":
            continue

        photo_count = len([f for f in folder.iterdir()
                          if f.is_file() and f.suffix.lower() in _IMAGE_EXTS])
        if photo_count == 0:
            continue

        key = sanitize_name(folder.name)
        gcs_folder = f"team_faces/{folder.name}"
        gcs_path   = f"gs://{GCS_BUCKET_NAME}/{gcs_folder}/"

        if key in existing:
            # Update photo count / gcs_path if they drifted
            person = existing[key]
            if person.photo_count != photo_count or person.gcs_path != gcs_path:
                person.photo_count = photo_count
                person.gcs_path    = gcs_path
                changed = True
        else:
            # Auto-create a new Person record
            display_name = folder.name  # preserve original capitalisation
            new_person = Person(
                org_id      = current_user.org_id,
                full_name   = display_name,
                gcs_path    = gcs_path,
                photo_count = photo_count,
                is_enrolled = photo_count >= 5,
                is_active   = True,
            )
            db.add(new_person)
            log.info("Auto-created Person record for local folder: %s (%d photos)",
                     display_name, photo_count)
            changed = True

    if changed:
        await db.commit()


@router.get("", response_model=list[PersonOut])
async def list_persons(
    include_inactive : bool = False,
    current_user     : User = Depends(require_role(["super_admin", "org_admin"])),
    db               : AsyncSession = Depends(get_db),
):
    """
    List persons.
    - super_admin → sees all persons across all orgs
    - org_admin   → sees only persons in their own org
    - By default only active persons are returned. Pass ?include_inactive=true to see all.
    """
    # Sync local TeamFaces folders into the database first
    await _sync_local_folders_to_db(db, current_user)

    stmt = select(Person)

    if current_user.role == "org_admin":
        stmt = stmt.where(Person.org_id == current_user.org_id)

    if not include_inactive:
        stmt = stmt.where(Person.is_active == True)

    result = await db.execute(stmt)
    persons = result.scalars().all()
    return [_person_out(p) for p in persons]


@router.get("/{person_id}", response_model=PersonOut)
async def get_person(
    person_id    : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """Get a single person's details."""
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)
    return _person_out(person)


@router.post("", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def create_person(
    payload      : CreatePersonRequest,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Create a new person record in the database.
    - The person's face dataset in GCS is tracked via gcs_path (set later).
    - org_admin creates persons within their own org automatically.
    - super_admin creates persons without an org (org_id will be null unless specified).
    """
    person = Person(
        org_id      = current_user.org_id,   # org_admin → their org; super_admin → null
        created_by  = current_user.id,
        full_name   = payload.full_name,
        employee_id = payload.employee_id,
        department  = payload.department,
    )
    db.add(person)
    await db.commit()
    await db.refresh(person)
    return _person_out(person)


@router.patch("/{person_id}", response_model=PersonOut)
async def update_person(
    person_id    : str,
    payload      : UpdatePersonRequest,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """Update a person's name, employee ID, or department."""
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    if payload.full_name   is not None: person.full_name   = payload.full_name
    if payload.employee_id is not None: person.employee_id = payload.employee_id
    if payload.department  is not None: person.department  = payload.department

    await db.commit()
    await db.refresh(person)
    return _person_out(person)


@router.patch("/{person_id}/mark-dataset", response_model=PersonOut)
async def mark_dataset(
    person_id    : str,
    payload      : MarkDatasetRequest,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Placeholder until GCS upload is wired up.
    Manually records that a dataset exists for this person in GCS.

    Once GCS is implemented, this endpoint will be replaced by
    POST /persons/{id}/photos which auto-uploads and sets these values.
    """
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    person.gcs_path    = payload.gcs_path
    person.photo_count = payload.photo_count

    await db.commit()
    await db.refresh(person)
    return _person_out(person)


@router.patch("/{person_id}/mark-enrolled", response_model=PersonOut)
async def mark_enrolled(
    person_id    : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Mark a person as enrolled — meaning the face recognition model
    has been trained on their photos and they can now be identified.
    """
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    if person.is_enrolled:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Person is already enrolled.")

    person.is_enrolled = True
    await db.commit()
    await db.refresh(person)
    return _person_out(person)


@router.post("/{person_id}/photos", response_model=PersonOut)
async def upload_photos(
    person_id    : str,
    files        : List[UploadFile] = File(...),
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Upload one or more face photos for a person.

    - Saves photos to data/raw/TeamFaces/{person.full_name}/
    - Uploads the same files to GCS at TeamFaces/{person.full_name}/
    - Updates photo_count and gcs_path in the DB
    """
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    # ── Validate files ────────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    for f in files:
        if f.content_type not in allowed_types:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"File '{f.filename}' is not a supported image type (jpeg/png/webp).",
            )

    # ── Local folder: data/raw/TeamFaces/<full_name>/ ─────────────────────
    safe_name  = sanitize_name(person.full_name)
    person_dir = TEAM_FACES_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for upload in files:
        dest = person_dir / upload.filename
        contents = await upload.read()
        dest.write_bytes(contents)
        saved_files.append(dest)

    # ── Upload to GCS ─────────────────────────────────────────────────────
    gcs_folder = f"team_faces/{safe_name}"
    try:
        from storage.gcs_storage import GCSStorage
        gcs = GCSStorage()
        for local_path in saved_files:
            remote_path = f"{gcs_folder}/{local_path.name}"
            gcs.upload_file(local_path, remote_path)
    except Exception as e:
        # Non-fatal: files are saved locally; log and continue
        import logging
        logging.getLogger(__name__).warning("GCS upload failed: %s", e)

    # ── Update DB ─────────────────────────────────────────────────────────
    person.gcs_path    = f"gs://{GCS_BUCKET_NAME}/{gcs_folder}/"
    person.photo_count = len(list(person_dir.glob("*")))

    await db.commit()
    await db.refresh(person)
    return _person_out(person)


@router.post("/retrain_all")
async def retrain_all_persons(
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
):
    """
    Trigger global model retraining for all active enrolled persons.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Global pipeline retraining failed: {e}",
        )

    # Hot-reload the model into the running API state
    try:
        from api.app import hot_reload_model
        hot_reload_model()
    except ImportError:
        pass

    return {"status": "success", "retrain_result": result}


class RetrainResponse(PersonOut):
    retrain_result: Optional[dict] = None


@router.post("/{person_id}/retrain", response_model=RetrainResponse)
async def retrain_person(
    person_id    : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Trigger the ML pipeline (preprocessing → feature_extraction → training)
    for the given person's photos, then mark them as enrolled.

    Requires at least 5 photos to be uploaded first.
    """
    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    if person.photo_count < 5:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Need at least 5 photos to retrain. Currently {person.photo_count} uploaded.",
        )

    # Run the pipeline in a thread so we don't block the event loop
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Pipeline failed: {e}",
        )

    # Mark enrolled
    person.is_enrolled = True
    await db.commit()
    await db.refresh(person)

    # Hot-reload the model into the running API state
    try:
        from api.app import hot_reload_model
        hot_reload_model()
    except ImportError:
        pass

    out = _person_out(person)
    out["retrain_result"] = result
    return out


def _run_pipeline() -> dict:
    """Blocking helper — runs preprocessing → feature_extraction → training."""
    from pipelines import preprocessing, feature_extraction, training

    pre_result  = preprocessing.run()
    feat_result = feature_extraction.run(valid_members=pre_result["valid_members"])
    train_result = training.run(
        X            = feat_result["X"],
        y            = feat_result["y"],
        X_raw        = feat_result["X_raw"],
        y_raw        = feat_result["y_raw"],
        valid_members= pre_result["valid_members"],
    )
    return {
        "preprocessing"      : {"valid_members": pre_result["valid_members"], "elapsed_s": pre_result["elapsed_s"]},
        "feature_extraction" : {"n_embeddings": feat_result["n_embeddings"],  "elapsed_s": feat_result["elapsed_s"]},
        "training"           : {k: v for k, v in train_result.items() if not hasattr(v, "__len__") or isinstance(v, (str, float, int, bool))},
    }


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id    : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Soft-delete a person (sets is_active = False).
    They remain in the database for audit purposes but won't appear in listings.
    Hard delete is not allowed to preserve historical integrity.
    """
    # imports already at module level

    person = await _get_person_or_404(person_id, db)
    _check_person_access(current_user, person)

    if not person.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Person is already deactivated.")

    person.is_active = False
    await db.commit()

    # Cleanup local directories (checking sanitized name, original name, and case insensitivity)
    safe_name = sanitize_name(person.full_name)
    local_dirs = {TEAM_FACES_DIR / safe_name, TEAM_FACES_DIR / person.full_name}
    for folder in TEAM_FACES_DIR.iterdir():
        if folder.is_dir() and sanitize_name(folder.name) == safe_name:
            local_dirs.add(folder)
            
    for local_dir in local_dirs:
        if local_dir.is_dir():
            try:
                shutil.rmtree(local_dir, ignore_errors=True)
                logging.getLogger(__name__).info("Deleted local person directory %s", local_dir)
            except Exception as e:
                logging.getLogger(__name__).warning("Failed to delete local directory %s: %s", local_dir, e)

    # Delete GCS folders
    try:
        from storage.gcs_storage import GCSStorage
        gcs = GCSStorage()
        
        prefixes = []
        if person.gcs_path and person.gcs_path.startswith(f"gs://{GCS_BUCKET_NAME}/"):
            prefix = person.gcs_path.replace(f"gs://{GCS_BUCKET_NAME}/", "")
            prefixes.append(prefix)
            
        for name in (safe_name, person.full_name):
            p1 = f"team_faces/{name}/"
            p2 = f"team_faces/{name}"
            for p in (p1, p2):
                if p not in prefixes:
                    prefixes.append(p)
                    
        for gcs_prefix in prefixes:
            logging.getLogger(__name__).debug("Cleaning up GCS folder: %s", gcs_prefix)
            gcs.delete_folder(gcs_prefix)
    except Exception as e:
        logging.getLogger(__name__).warning("GCS cleanup failed for %s: %s", person_id, e)

    # Optional auto‑retrain after deletion
    if AUTO_RETRAIN_ON_DELETE:
        asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, _run_pipeline)
        )

    return None