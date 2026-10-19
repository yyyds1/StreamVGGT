#!/usr/bin/env python3
import os
from pathlib import Path
from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parent / "ActionVGGT"
REPO = os.environ.get("MODELSCOPE_MODEL_REPO", "yyyyds139/ActionVGGT")

def main():
    token = os.environ.get("MODELSCOPE_TOKEN", "")
    if os.environ.get("UPLOAD", "0") != "1":
        print(f"DRY-RUN: {ROOT} -> {REPO}; set UPLOAD=1 to upload")
        return 0
    if not token:
        raise SystemExit("UPLOAD=1 requires MODELSCOPE_TOKEN")
    api = HubApi(); api.login(token); api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=str(ROOT), repo_type="model", commit_message="Upload ActionVGGT checkpoints")
    print("Upload complete")

if __name__ == "__main__": raise SystemExit(main())
