# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""``/apps``: register an app, get its key once, revoke it (ADR-006).

Managing apps needs what any other route needs: on a rig with
``access.token`` set, the shared token or an app key. An app key admits
exactly what the shared token does, so it can manage apps too; apps are
labels for counting, not roles.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from giq.apps import AppError, get_apps

router = APIRouter(tags=["apps"])


class AppCreate(BaseModel):
    name: str = Field(description="Slug: lowercase letters, digits, '-' and '_'")
    description: str | None = None


@router.get("/apps")
async def list_apps() -> dict:
    """Every app, without keys."""
    return {"apps": [a.public() for a in get_apps().entries()]}


@router.post("/apps", status_code=201)
async def create_app(body: AppCreate) -> dict:
    """A new app and its key. The key is in this response and nowhere else."""
    try:
        app, key = get_apps().create(body.name, body.description)
    except AppError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return {**app.public(), "key": key}


@router.post("/apps/{name}/revoke")
async def revoke_app(name: str) -> dict:
    """Stop the key working. The app's history stays, and its name stays taken."""
    try:
        return get_apps().revoke(name).public()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No app named {name!r}.") from None


@router.delete("/apps/{name}", status_code=204)
async def delete_app(name: str) -> None:
    """Remove the entry. Its jobs stay in the stats under its name, and a new
    app created under the same name adds to that history."""
    try:
        get_apps().delete(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No app named {name!r}.") from None
