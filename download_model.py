"""
Downloads the chess model from Google Drive if it doesn't already exist.
Handles Google Drive's virus-scan confirmation page for large files.
"""
import os
import sys
import requests

MODEL_PATH = os.getenv("MODEL_PATH", "my_chess_model.v2.keras")
FILE_ID    = os.getenv("GDRIVE_FILE_ID", "")


def download_from_gdrive(file_id: str, dest_path: str):
    print(f"[model-download] Starting download from Google Drive...")

    session = requests.Session()

    # Step 1 — Hit the initial download URL
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(url, params=params, stream=True)

    print(f"[model-download] Initial response status: {response.status_code}")

    # Step 2 — Google Drive returns a confirmation page for files > 40MB
    # We need to find the confirmation token and re-request with it
    confirm_token = None

    # Check cookies for download_warning token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            print(f"[model-download] Got confirmation token from cookie.")
            break

    # If not in cookies, search the response HTML for the confirm parameter
    if not confirm_token:
        content = response.text
        if "confirm=" in content:
            start = content.find("confirm=") + len("confirm=")
            end   = content.find("&", start)
            confirm_token = content[start:end] if end != -1 else content[start:start+10]
            print(f"[model-download] Got confirmation token from HTML.")

    # Step 3 — Re-request with the confirmation token
    if confirm_token:
        print(f"[model-download] Confirming download...")
        params["confirm"] = confirm_token
        params["uuid"]    = _extract_uuid(response.text)
        response = session.get(url, params=params, stream=True)
    else:
        # Try the newer Google Drive direct download URL format
        print(f"[model-download] No token found, trying direct URL format...")
        alt_url  = f"https://drive.usercontent.google.com/download"
        params2  = {"id": file_id, "export": "download", "confirm": "t"}
        response = session.get(alt_url, params=params2, stream=True)

    print(f"[model-download] Download response status: {response.status_code}")

    # Step 4 — Stream file to disk in chunks
    total_bytes = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    size_mb = total_bytes / (1024 * 1024)
    print(f"[model-download] Downloaded {size_mb:.1f} MB → '{dest_path}'")

    # Step 5 — Sanity check: file should be more than 1MB
    if size_mb < 1:
        os.remove(dest_path)
        raise RuntimeError(
            f"Downloaded file is only {size_mb:.2f} MB — "
            "Google Drive likely returned an error page instead of the model. "
            "Please check that the file is shared as 'Anyone with the link'."
        )

    print(f"[model-download] Model download complete.")


def _extract_uuid(html: str) -> str:
    """Extract uuid parameter from Google Drive confirmation page if present."""
    if "uuid=" in html:
        start = html.find("uuid=") + len("uuid=")
        end   = html.find("&", start)
        return html[start:end] if end != -1 else html[start:start+36]
    return ""


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb > 1:
            print(f"[model-download] Model already exists ({size_mb:.1f} MB), skipping download.")
            sys.exit(0)
        else:
            print(f"[model-download] Existing file is too small ({size_mb:.2f} MB), re-downloading...")
            os.remove(MODEL_PATH)

    if not FILE_ID:
        print("[model-download] ERROR: GDRIVE_FILE_ID environment variable is not set.")
        sys.exit(1)

    download_from_gdrive(FILE_ID, MODEL_PATH)
