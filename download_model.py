"""
Downloads the chess model from Google Drive if it doesn't already exist.
Runs automatically at container startup before the API loads.
"""
import os
import sys
import requests

MODEL_PATH  = os.getenv("MODEL_PATH", "my_chess_model.v2.keras")
FILE_ID     = os.getenv("GDRIVE_FILE_ID", "")   # set this on Render


def download_from_gdrive(file_id: str, dest_path: str):
    print(f"[model-download] Downloading model from Google Drive...")

    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    response = session.get(url, stream=True)

    # Google Drive adds a virus-scan warning for large files
    # We need to confirm and re-request
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(
            url,
            params={"confirm": token},
            stream=True
        )

    # Stream the file to disk in chunks (avoids loading it all into RAM)
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"[model-download] Done. Saved to '{dest_path}' ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # Only download if the model file doesn't already exist
    if os.path.exists(MODEL_PATH):
        print(f"[model-download] Model already exists at '{MODEL_PATH}', skipping download.")
        sys.exit(0)

    if not FILE_ID:
        print("[model-download] ERROR: GDRIVE_FILE_ID environment variable is not set.")
        sys.exit(1)

    download_from_gdrive(FILE_ID, MODEL_PATH)
