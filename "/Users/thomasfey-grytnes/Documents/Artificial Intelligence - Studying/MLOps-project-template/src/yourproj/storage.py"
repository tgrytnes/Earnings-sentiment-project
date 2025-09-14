from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContainerClient


def get_blob_service_client() -> BlobServiceClient:
    """Create an Azure BlobServiceClient from env.

    Env vars supported:
      - AZURE_BLOB_ENDPOINT (e.g., https://account.blob.core.windows.net)
      - AZURE_STORAGE_ACCOUNT (optional if endpoint present)
      - AZURE_STORAGE_KEY or AZURE_STORAGE_CONNECTION_STRING
    """
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    endpoint = os.environ.get("AZURE_BLOB_ENDPOINT")
    if not endpoint:
        acct = os.environ.get("AZURE_STORAGE_ACCOUNT")
        if not acct:
            raise RuntimeError("Set AZURE_BLOB_ENDPOINT or AZURE_STORAGE_ACCOUNT")
        endpoint = f"https://{acct}.blob.core.windows.net"
    key = os.environ.get("AZURE_STORAGE_KEY")
    if not key:
        raise RuntimeError("Set AZURE_STORAGE_KEY or AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient(account_url=endpoint, credential=key)


def get_container_client(container: Optional[str] = None) -> ContainerClient:
    container = container or os.environ.get("AZURE_BLOB_CONTAINER")
    if not container:
        raise RuntimeError("Provide container or set AZURE_BLOB_CONTAINER")
    svc = get_blob_service_client()
    return svc.get_container_client(container)


def upload_file(local_path: str, container: str, blob_name: str, overwrite: bool = True) -> None:
    cc = get_container_client(container)
    with open(local_path, "rb") as fh:
        cc.upload_blob(name=blob_name, data=fh, overwrite=overwrite)


def download_file(container: str, blob_name: str, local_path: str) -> None:
    cc = get_container_client(container)
    stream = cc.download_blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        fh.write(stream.readall())

