"""Download deep-sea fish images + bounding boxes from FathomNet (MBARI) as a YOLO 'fish' dataset.

FathomNet (https://fathomnet.org) is a public, expert-annotated deep-sea image database; no account
is needed for downloads. Images are listed by the fish groups below; then EVERY box whose concept is a
fish taxon (any species under Actinopterygii, Chondrichthyes, Myxini, Petromyzonti) is labelled 'fish',
so species-level boxes and other fish in the same frame are not left unlabelled. Images are resized to
max 640 px on download.

Run:  python -m src.data.fetch_fathomnet_fish
"""
from __future__ import annotations

import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "raw" / "fathomnet_fish"
# Deep-sea fish concepts with measured box counts (2026-10-13); per-concept image cap keeps classes balanced
FISH_CONCEPTS = {
    "Sebastolobus": 1800, "Sebastes": 900, "Macrouridae": 700, "Actinopterygii": 400,
    "Ophidiidae": 80, "Zoarcidae": 80, "Liparidae": 60, "Myctophidae": 50,
}
MAX_SIDE = 640
TAXA_CACHE = OUT / "fish_taxa.json"
FISH_TAXA: set[str] = set()


def load_fish_taxa() -> set[str]:
    import json as _json

    if TAXA_CACHE.exists():
        return set(_json.loads(TAXA_CACHE.read_text()))
    from fathomnet.api import taxa

    names = {"Actinopterygii", "Chondrichthyes", "Myxini", "Teleostei", "Pisces"}
    for group in ("Actinopterygii", "Chondrichthyes", "Myxini", "Petromyzonti"):
        names |= {t.name for t in taxa.find_taxa("fathomnet", group)}  # ~2 min for Actinopterygii
    TAXA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TAXA_CACHE.write_text(_json.dumps(sorted(names)))
    return names


def _download(img) -> dict | None:
    name = f"fn_{img.uuid}.jpg"
    out_img = OUT / "images" / name
    boxes = [b for b in (img.boundingBoxes or []) if b.concept in FISH_TAXA]
    if not boxes or not img.width or not img.height:
        return None
    if not out_img.exists():
        try:
            with urllib.request.urlopen(img.url, timeout=60) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            return None
        scale = MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)))
        im.save(out_img, quality=90)
    lines = []
    for b in boxes:
        cx = (b.x + b.width / 2) / img.width
        cy = (b.y + b.height / 2) / img.height
        w, h = b.width / img.width, b.height / img.height
        if w > 0 and h > 0:
            lines.append(f"0 {min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} {min(w, 1):.6f} {min(h, 1):.6f}")
    (OUT / "labels" / f"fn_{img.uuid}.txt").write_text("\n".join(lines))
    return {"image": name, "boxes": len(lines), "concepts": sorted({b.concept for b in boxes})}


def main() -> None:
    from fathomnet.api import images

    global FISH_TAXA
    FISH_TAXA = load_fish_taxa()
    print(f"  {len(FISH_TAXA)} fish taxa loaded", flush=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    seen, todo = set(), []
    for concept, cap in FISH_CONCEPTS.items():
        found = images.find_by_concept(concept)
        picked = [im for im in found if im.uuid not in seen][:cap]
        seen |= {im.uuid for im in picked}
        todo += picked
        print(f"  {concept}: {len(found)} images available, taking {len(picked)}", flush=True)
    print(f"Downloading {len(todo)} images...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, row in enumerate(ex.map(_download, todo), 1):
            if row:
                rows.append(row)
            if i % 250 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"Done: {len(rows)} images, {sum(r['boxes'] for r in rows)} fish boxes -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
