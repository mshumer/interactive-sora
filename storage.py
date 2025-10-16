from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3


@dataclass
class StoredAsset:
    video_url: str
    poster_url: str
    bytes_written: int


class StorageClient:
    def upload(self, video_path: Path, poster_path: Path, *, key_prefix: str) -> StoredAsset:
        raise NotImplementedError

    def delete(self, key_prefix: str) -> None:
        raise NotImplementedError


class R2StorageClient(StorageClient):
    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        public_base_url: Optional[str] = None,
    ) -> None:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        self._bucket = bucket_name
        self._base_url = public_base_url.rstrip("/") if public_base_url else None
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def upload(self, video_path: Path, poster_path: Path, *, key_prefix: str) -> StoredAsset:
        video_key = f"{key_prefix}.mp4"
        poster_key = f"{key_prefix}.jpg"
        total_bytes = 0

        total_bytes += self._upload_file(video_path, video_key, "video/mp4")
        total_bytes += self._upload_file(poster_path, poster_key, "image/jpeg")

        video_url = self._asset_url(video_key)
        poster_url = self._asset_url(poster_key)

        return StoredAsset(video_url=video_url, poster_url=poster_url, bytes_written=total_bytes)

    def _upload_file(self, path: Path, key: str, content_type: str) -> int:
        with path.open("rb") as fh:
            data = fh.read()
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return len(data)

    def _asset_url(self, key: str) -> str:
        if self._base_url:
            return f"{self._base_url}/{key}"
        return f"https://{self._bucket}.r2.cloudflarestorage.com/{key}"

    def delete(self, key_prefix: str) -> None:
        for suffix in (".mp4", ".jpg"):
            key = f"{key_prefix}{suffix}"
            try:
                self._s3.delete_object(Bucket=self._bucket, Key=key)
            except Exception:
                pass


class LocalStorageClient(StorageClient):
    def __init__(self, *, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or Path("storage")
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def upload(self, video_path: Path, poster_path: Path, *, key_prefix: str) -> StoredAsset:
        key_path = Path(key_prefix)
        video_target = (self._base_dir / key_path).with_suffix(".mp4")
        poster_target = (self._base_dir / key_path).with_suffix(".jpg")
        video_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, video_target)
        shutil.copy2(poster_path, poster_target)

        relative = key_path.as_posix()
        return StoredAsset(
            video_url=f"/storage/{relative}.mp4",
            poster_url=f"/storage/{relative}.jpg",
            bytes_written=video_target.stat().st_size + poster_target.stat().st_size,
        )

    def delete(self, key_prefix: str) -> None:
        for suffix in (".mp4", ".jpg"):
            candidate = (self._base_dir / Path(key_prefix)).with_suffix(suffix)
            if candidate.exists():
                candidate.unlink(missing_ok=True)


def build_storage_client() -> StorageClient:
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")

    if account_id and access_key_id and secret_access_key and bucket:
        return R2StorageClient(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket,
            public_base_url=os.environ.get("R2_PUBLIC_BASE_URL"),
        )

    return LocalStorageClient()


def random_key() -> str:
    return uuid.uuid4().hex
