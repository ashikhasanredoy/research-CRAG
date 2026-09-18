"""
Downloads real AI/ML and Adverse Weather/CRAG research papers from arXiv.
"""
import urllib.request
from pathlib import Path
from app.config import settings

REAL_PAPERS = [
    {
        "url": "https://arxiv.org/pdf/2401.15884.pdf",
        "filename": "2401.15884_crag_corrective_rag.pdf",
        "title": "Corrective Retrieval Augmented Generation"
    },
    {
        "url": "https://arxiv.org/pdf/2112.08088.pdf",
        "filename": "2112.08088_ia_yolo_adverse_weather.pdf",
        "title": "IA-YOLO: Image-Adaptive YOLO for Object Detection in Adverse Weather"
    },
    {
        "url": "https://arxiv.org/pdf/1707.06543.pdf",
        "filename": "1707.06543_aod_net_image_dehazing.pdf",
        "title": "AOD-Net: All-in-One Dehazing Network"
    },
    {
        "url": "https://arxiv.org/pdf/2308.01633.pdf",
        "filename": "2308.01633_pe_yolo_low_light_detection.pdf",
        "title": "PE-YOLO: Pyramid Enhancement Network for Object Detection in Extremely Low-Light Conditions"
    }
]

def download_real_papers():
    settings.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

    for item in REAL_PAPERS:
        dest = settings.PAPERS_DIR / item["filename"]
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"Downloading {item['title']} -> {item['filename']}...")
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            print(f"Saved {item['filename']} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"Paper already exists: {item['filename']} ({dest.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_real_papers()
