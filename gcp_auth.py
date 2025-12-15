import os
import logging
from typing import Optional, Sequence

from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# 서비스 계정 JSON 파일 경로를 담은 환경변수
SERVICE_ACCOUNT_FILE: Optional[str] = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
DEFAULT_SCOPES: Sequence[str] = (
    "https://www.googleapis.com/auth/cloud-platform",
)


class GCPAuthError(Exception):
    """GCP 인증 관련 에러 (어느 단계에서 터졌는지 구분용)."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage          # 예: "env", "file_missing", "load", "credentials"
        self.detail = detail
        super().__init__(f"[{stage}] {detail}")


def get_credentials(scopes: Optional[Sequence[str]] = None) -> service_account.Credentials:
    """
    서비스 계정 JSON 파일로부터 Credentials 객체 생성.
    문제가 생기면 GCPAuthError(stage, detail)를 던진다.
    """
    if not SERVICE_ACCOUNT_FILE:
        raise GCPAuthError("env", "GCP_SERVICE_ACCOUNT_JSON 환경변수가 비어 있습니다.")

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise GCPAuthError("file_missing", f"파일을 찾을 수 없습니다: {SERVICE_ACCOUNT_FILE}")

    real_scopes = scopes or DEFAULT_SCOPES

    logger.info("Using GCP service account file: %s", SERVICE_ACCOUNT_FILE)

    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=list(real_scopes),
        )
    except Exception as e:
        raise GCPAuthError("credentials", f"서비스 계정 로딩 실패: {e}")

    return creds
