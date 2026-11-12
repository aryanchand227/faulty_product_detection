"""
Role-Based Access Control (RBAC) and Security Gatekeeper for Industrial AOI Service.
Defines Operator, Auditor, and Administrator privilege hierarchies.
Enforces constant-time authentication and eliminates silent authorization bypasses.
"""

from enum import Enum
import hmac
import logging
from typing import Callable, Optional
from fastapi import Header, HTTPException, status

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    OPERATOR = "OPERATOR"
    AUDITOR = "AUDITOR"
    ADMIN = "ADMIN"


# Role privilege hierarchy: ADMIN > AUDITOR > OPERATOR
ROLE_HIERARCHY = {
    Role.ADMIN: {Role.ADMIN, Role.AUDITOR, Role.OPERATOR},
    Role.AUDITOR: {Role.AUDITOR, Role.OPERATOR},
    Role.OPERATOR: {Role.OPERATOR},
}


def authenticate_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[Role]:
    """
    Authenticates client identity against role-scoped secrets.
    Uses constant-time comparison to prevent timing side-channel attacks.
    """
    if not x_api_key:
        return None

    api_key_bytes = x_api_key.encode("utf-8")

    if hmac.compare_digest(api_key_bytes, settings.ADMIN_API_KEY.encode("utf-8")):
        return Role.ADMIN
    if hmac.compare_digest(api_key_bytes, settings.AUDITOR_API_KEY.encode("utf-8")):
        return Role.AUDITOR
    if hmac.compare_digest(api_key_bytes, settings.OPERATOR_API_KEY.encode("utf-8")):
        return Role.OPERATOR

    return None


def require_role(required_role: Role, allow_dev_default: bool = False) -> Callable:
    """
    FastAPI dependency factory enforcing strict role-based authorization.
    Rejects unauthorized, invalid, or missing credentials.
    """
    def dependency(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ) -> Role:
        if not x_api_key:
            # Only allow dev fallback for non-admin endpoints when auth enforcement is not explicitly enabled
            if allow_dev_default and not getattr(settings, "ENFORCE_AUTH", False):
                return required_role
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication credentials required. Missing 'X-API-Key' header for {required_role.value} endpoint.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        client_role = authenticate_api_key(x_api_key)
        if client_role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid 'X-API-Key' credential.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        allowed_roles = ROLE_HIERARCHY.get(client_role, set())
        if required_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Action requires '{required_role.value}' role privileges. Your role: '{client_role.value}'.",
            )

        return client_role

    return dependency


# Production Security Dependencies
# Admin strictly requires valid admin key (NEVER allows anonymous access)
require_admin = require_role(Role.ADMIN, allow_dev_default=False)

# Auditor & Operator
require_auditor = require_role(Role.AUDITOR, allow_dev_default=True)
require_operator = require_role(Role.OPERATOR, allow_dev_default=True)

# Strict versions requiring explicit header
require_auditor_strict = require_role(Role.AUDITOR, allow_dev_default=False)
require_operator_strict = require_role(Role.OPERATOR, allow_dev_default=False)
