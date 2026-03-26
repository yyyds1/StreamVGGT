import os
from huggingface_hub import HfApi, hf_hub_download

# Must be set before API usage
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def main():
    dataset_repo = "robbyant/robotwin-clean-and-aug-lerobot"
    local_dir = "/home/yds/code/StreamVGGT/dataset"
    subdir = "lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50"
    endpoint = os.environ["HF_ENDPOINT"]

    print(f"Running file: {__file__}")
    print(f"HF_ENDPOINT={endpoint}")

    api = HfApi(endpoint=endpoint)

    files = []
    for item in api.list_repo_tree(
        repo_id=dataset_repo,
        repo_type="dataset",
        path_in_repo=subdir,
        recursive=True,
        expand=False,
    ):
        if getattr(item, "type", None) == "file":
            files.append(item.path)

    print(f"Found {len(files)} files under {subdir}")
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path}")
        hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=path,
            local_dir=local_dir,
            endpoint=endpoint,
            etag_timeout=10,
        )

if __name__ == "__main__":
    main()