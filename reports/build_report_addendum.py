"""Insert two new subsections (7.5 fish detection details, 7.6 Gulf of Mannar environmental
risk module) into the faculty-reviewed report without touching any existing page.

The reviewed PDF is copied page for page; the new pages are laid out to match it (Letter,
1-inch margins, Times New Roman 12 pt, bordered tables with a light-blue header row, bold
centred captions) and placed after page 18, i.e. after Table 7.6 and before
"8. Results Discussion". Every number comes from outputs/metrics/*.json or
models/environmental_risk/metrics.json.

Run: .venv/Scripts/python.exe reports/build_report_addendum.py "<reviewed.pdf>"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "reports" / "assets" / "env_risk"
OUT_DIR = ROOT / "reports"
INSERT_AFTER_PAGE = 18  # 1-based: Table 7.6 page; chapter 8 starts on page 19
PAGE = pymupdf.paper_rect("letter")
BODY = PAGE + (72, 72, -72, -72)

CSS = """
@font-face {font-family: tnr; src: url(times.ttf);}
@font-face {font-family: tnr; src: url(timesbd.ttf); font-weight: bold;}
@font-face {font-family: tnr; src: url(timesi.ttf); font-style: italic;}
* {font-family: tnr; font-size: 12pt; color: #000000;}
p {text-align: justify; line-height: 1.32; margin: 0 0 11pt 0;}
h3 {font-weight: bold; font-size: 12pt; margin: 0 0 11pt 0;}
.cap {font-weight: bold; text-align: center; margin: 4pt 0 6pt 0;}
.figcap {font-weight: bold; text-align: center; margin: 4pt 0 14pt 0;}
table {border-collapse: collapse; width: 100%; margin: 0 0 14pt 0;}
td, th {border: 1px solid #000000; padding: 3pt 5pt; text-align: center; vertical-align: middle; line-height: 1.2;}
th {font-weight: bold;}
td.l {text-align: left;}
table.small td, table.small th {font-size: 11pt; padding: 3pt 3pt;}
table.figs td {border: none; padding: 0 4pt;}
.eq {text-align: center; margin: 2pt 0 11pt 0;}
.fig {text-align: center; margin: 6pt 0 0 0;}
"""

HTML = """
<h3>7.5 Fish Detection: Test Details</h3>
<p>Table 7.7 gives the full test figures behind Table 7.6. All fish detectors were scored the same
way: a predicted box counts as correct when it overlaps a labelled fish with IoU &#8805; 0.5 (greedy
one-to-one matching), the confidence threshold is chosen on a validation split only, and the test
footage comes from videos, camera drops or clips that the threshold choice never saw. The 95%
confidence intervals are bootstrapped over those test videos, drops or clips.</p>

<p class="cap">Table 7.7. Fish detection test details (IoU &#8805; 0.5)</p>
<table class="small">
<tr><th width="25%">Model</th><th width="19%">Test data (frames / fish)</th><th width="7%">Thr.</th><th width="17%">Precision [95% CI]</th><th width="17%">Recall [95% CI]</th><th width="7%">F1</th><th width="8%">AP@50</th></tr>
<tr><td class="l">16-class detector (ours)</td><td>TrashCan, 1,313 / 151</td><td>0.18</td><td>40.7% [7.1, 62.3]</td><td>15.9% [4.2, 27.2]</td><td>22.9%</td><td>-</td></tr>
<tr><td class="l">Fish detector + FathomNet (ours)</td><td>TrashCan, 1,313 / 151</td><td>0.18</td><td>17.8% [5.9, 29.2]</td><td>38.4% [25.3, 52.9]</td><td>24.4%</td><td>-</td></tr>
<tr><td class="l">Fish detector + FathomNet (ours)</td><td>TrashCan, 1,313 / 151</td><td>0.35*</td><td>26.5% [8.0, 46.4]</td><td>20.5% [8.8, 38.7]</td><td>23.1%</td><td>-</td></tr>
<tr><td class="l">Community Fish Detector (RF-DETR M)</td><td>TrashCan, 1,313 / 151</td><td>0.85*</td><td>12.8% [0.0, 32.8]</td><td>7.9% [0.0, 25.4]</td><td>9.8%</td><td>8.1%</td></tr>
<tr><td class="l">Community Fish Detector (RF-DETR M)</td><td>OzFish, 1,144 / 27,802</td><td>0.20*</td><td>76.7% [75.5, 77.9]</td><td>59.3% [56.3, 62.1]</td><td>66.9%</td><td>65.9%</td></tr>
<tr><td class="l">Brackish YOLOv8s</td><td>DeepFish, 758 / 2,767</td><td>0.35*</td><td>4.7% [1.0, 14.5]</td><td>2.3% [0.5, 6.3]</td><td>3.1%</td><td>0.5%</td></tr>
</table>
<p>* threshold chosen on the validation split.</p>

<p>The FathomNet-trained fish detector finds more fish at the default threshold (recall 38.4%)
but at the cost of many false alarms, and once its threshold is set on validation data its F1
(23.1%) is no better than the original detector's. Both results lie inside each other's confidence
intervals, so the original 16-class detector was kept and is the model that supplies fish
detections to tracking and the dashboard.</p>

<p>The pretrained Community Fish Detector works well on shallow, well-lit OzFish footage (66.9% F1
on 27,802 fish) but drops to 9.8% F1 on the dark deep-sea TrashCan frames. At image level it
judged whether a fish was present correctly in 86.9% of TrashCan test images, which is below the
92.2% obtained by always answering 'no fish'. The Brackish YOLOv8s model, which reports 99.3%
mAP@50 on its own data, found only 65 of 2,767 fish on DeepFish. Inspection of its output showed
its 1,312 false alarms falling on coral and along the bottom edge of the frame, while large, clearly
visible fish were missed.</p>

<p>Together these tests show that fish detectors do not carry over between water types: each model
was strongest on footage like its training data and weak elsewhere. Fish labelled in the target
environment are therefore needed. A fine-tuning run on DeepFish, with every training clip that
appears in the test set removed, has been prepared but not yet run, so no result is reported for
it.</p>

<h3>7.6 Regional Environmental Risk Assessment (Gulf of Mannar)</h3>
<p>The image pipeline cannot place its survey images on a map, because TrashCan frames carry no GPS
position. This module, developed by Anannya Jadhav, works the other way round: it starts from a real
location and scores ecological risk across it from measured ocean, species and habitat data. The
study area is the Gulf of Mannar and Palk Bay coral region between India and Sri Lanka, bounded by
an Allen Coral Atlas polygon. Every ocean grid cell of the Copernicus model inside the polygon is a
study site, which gives 245 sites about 9.2 km apart (Table 7.8).</p>

<p class="cap">Table 7.8. Data used by the environmental risk module</p>
<table>
<tr><th>Source</th><th>Variables</th><th>Coverage used</th></tr>
<tr><td class="l">Copernicus Marine global analysis (0.083&#176;)</td><td class="l">Current speed and direction, sea temperature (daily); significant wave height (3-hourly, averaged per day)</td><td class="l">60 days, 4 Aug - 2 Oct 2026; 245 sites</td></tr>
<tr><td class="l">OBIS occurrence records [18]</td><td class="l">Species name, class, position, year, depth</td><td class="l">Since 2000; 6,208 records, 5,254 after cleaning, 3,536 species</td></tr>
<tr><td class="l">Allen Coral Atlas benthic map</td><td class="l">Coral/algae, seagrass, rubble, rock and sand polygons</td><td class="l">Whole study polygon</td></tr>
</table>

<p>For each site the module computes, within a 10 km radius: species richness, number of taxonomic
classes and record density from OBIS; coral cover, seagrass cover, distance to the nearest coral and
a habitat sensitivity score from the Coral Atlas (coral 1.0, seagrass 0.8, rubble 0.4, rock 0.3,
sand 0.1); and the 60-day mean current speed, wave height and temperature from Copernicus. Across the
area the mean current was 0.122 m/s, the mean sea temperature 29.02 &#176;C and the mean wave height
1.28 m. OBIS records were cleaned by keeping species-level names inside the polygon at depths of
200 m or less and removing duplicates.</p>

<p>An XGBoost regressor (200 trees, depth 3, learning rate 0.05) predicts log(1 + species richness)
from the seven habitat and ocean features, and SHAP measures how much each feature drives the
prediction (Table 7.9, Figure 7.4). The SHAP importances are then used as weights. Habitat features
form an ecological value score V and physical features form an exposure score E, each a weighted sum
of features scaled to 0-1. Sites with fewer species than the model predicts receive a biodiversity
deficit D (0-1), which raises their score by up to 30%:</p>
<p class="eq">Impact = 100 &#215; min(1, &#8730;(V &#183; E) &#215; (1 + 0.3 &#183; D))</p>
<p>The geometric mean means a site scores high only when it is both valuable and exposed.</p>

<p class="cap">Table 7.9. Species richness model and learned weights</p>
<table>
<tr><th>Result</th><th>Value</th></tr>
<tr><td class="l">XGBoost, 5-fold cross-validation R&#178;</td><td>0.385 (folds 0.18 - 0.60)</td></tr>
<tr><td class="l">XGBoost, nested cross-validation with tuning, R&#178;</td><td>0.325 (folds 0.11 - 0.49)</td></tr>
<tr><td class="l">XGBoost, held-out 20% test R&#178;</td><td>0.157</td></tr>
<tr><td class="l">Random Forest, 5-fold cross-validation R&#178;</td><td>0.427</td></tr>
<tr><td class="l">Value weights (from SHAP)</td><td>habitat sensitivity 0.357, seagrass 0.304, coral proximity 0.287, coral cover 0.052</td></tr>
<tr><td class="l">Exposure weights (from SHAP)</td><td>wave height 0.420, current speed 0.355, temperature 0.225</td></tr>
<tr><td class="l">Rank stability, 500 random weightings (Spearman)</td><td>mean 0.991, 5th percentile 0.968, minimum 0.877</td></tr>
</table>

<p>Habitat sensitivity, wave height and seagrass cover had the largest influence on predicted
richness. The ranking of sites barely changed when the weights were drawn at random, so the result
does not depend on the exact weights chosen. The highest-scoring sites were S005 (8.00&#176; N,
79.75&#176; E, score 67.1), S231 (9.67&#176; N, 79.92&#176; E, 61.5) and S230 (9.67&#176; N,
79.83&#176; E, 57.1): S231 and S230 combine high habitat value with moderate exposure, while
S005 is strongly exposed and has far fewer recorded species than its conditions predict (Figure 7.3).</p>

<table class="figs"><tr><td><img src="cell03_0.png" width="225"/></td><td><img src="cell20_0.png" width="225"/></td></tr></table>
<p class="figcap">Figure 7.3: Left: the 245 study sites inside the Coral Atlas polygon, coloured by mean
current speed. Right: Marine Pollution Impact Score per site.</p>

<p class="fig"><img src="cell18_0.png" width="310"/></p>
<p class="figcap">Figure 7.4: SHAP summary of the species richness model.</p>

<p>The module has clear limits. Plastic is not yet part of the score, because no measured plastic data
exists for these sites; the score is built so that a plastic term can be added later. The OBIS
download carried no IUCN Red List categories, so threatened species could not be weighted. 115 of the
245 sites have no OBIS record within 10 km, so species richness partly reflects where surveys were
made. The habitat sensitivity weights are assumptions, and the held-out R&#178; of 0.157 shows that
the richness model explains only part of the variation; random cross-validation on neighbouring grid
cells is also likely to be optimistic. The site scores therefore rank relative risk within the study
area and are not measured biodiversity loss. The model results are shown on the Biodiversity Loss
Prediction page of the dashboard and the site map on the Marine Pollution Impact Score page.</p>
"""


def build_new_pages() -> pymupdf.Document:
    arch = pymupdf.Archive("C:/Windows/Fonts")
    arch.add(str(ASSETS))
    story = pymupdf.Story(html=HTML, user_css=CSS, archive=arch)
    buf = pymupdf.DocumentWriter(OUT_DIR / "_addendum_pages.pdf")
    more = True
    while more:
        dev = buf.begin_page(PAGE)
        more, _ = story.place(BODY)
        story.draw(dev)
        buf.end_page()
    buf.close()
    doc = pymupdf.open(OUT_DIR / "_addendum_pages.pdf")
    shade_header_rows(doc)
    return pymupdf.open("pdf", doc.tobytes())


def shade_header_rows(doc: pymupdf.Document) -> None:
    """Light-blue fill behind each table's first row, like the reviewed report's tables.

    Story draws each border as a thin filled rectangle, so the header row is the band
    between the first two horizontal borders below a "Table 7.x." caption.
    """
    for page in doc:
        hlines = [d["rect"] for d in page.get_drawings()
                  if d.get("fill") is not None and d["rect"].height < 1.5 and d["rect"].width > 5]
        ys = sorted({round(r.y0, 1) for r in hlines})
        for cap in page.search_for("Table 7."):
            below = [y for y in ys if y > cap.y1]
            if len(below) < 2 or below[0] - cap.y1 > 15:  # a mention in body text, not a caption
                continue
            top, bottom = below[0], next(y for y in below if y > below[0] + 2)
            row = [r for r in hlines if abs(r.y0 - top) < 0.5]
            page.draw_rect(pymupdf.Rect(min(r.x0 for r in row), top, max(r.x1 for r in row), bottom + 1),
                           color=None, fill=(0.851, 0.886, 0.953), overlay=False)


def main(reviewed_pdf: str) -> None:
    src = pymupdf.open(reviewed_pdf)
    new = build_new_pages()
    out = pymupdf.open()
    out.insert_pdf(src, from_page=0, to_page=INSERT_AFTER_PAGE - 1)
    out.insert_pdf(new)
    out.insert_pdf(src, from_page=INSERT_AFTER_PAGE)
    out.set_metadata(src.metadata)
    target = OUT_DIR / "Group_18_v3.pdf"
    out.save(target, garbage=3, deflate=True)
    print(f"{src.page_count} original + {new.page_count} new pages -> {target} ({out.page_count} pages)")


if __name__ == "__main__":
    main(sys.argv[1])
