"""
scripts/inspect_dataset.py
--------------------------
Report what an unfamiliar detection dataset actually contains, before you
merge it into anything. Standard library only -- runs on any Python 3, no
install needed, so it works on a machine that has none of this project set up.

    python inspect_dataset.py <folder>

Handles COCO (*.json), Pascal VOC (*.xml) and YOLO (labels/*.txt + data.yaml).

The number that matters most for UroLens is the "per img" column: our bacteria
are annotated as individual organisms at ~23 boxes per image. A dataset that
boxes bacterial CLUSTERS will show ~1-3, and merging it would teach the model
two contradictory definitions of the same class.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


def report(title, inst, imgs, note=""):
    if not inst:
        return
    total = sum(inst.values())
    print(f"\n=== {title} — {total} annotations ===")
    if note:
        print(f"    {note}")
    print(f"{'class':<30}{'inst':>9}{'images':>9}{'per img':>10}")
    for name, n in inst.most_common():
        m = len(imgs[name])
        flag = ""
        low = name.lower()
        if "bacteri" in low:
            flag = "   <-- BACTERIA (ours: 22.9/img)"
        elif "mucus" in low or "mucous" in low:
            flag = "   <-- MUCUS (ours: 3.7/img)"
        print(f"{name:<30}{n:>9}{m:>9}{n/m if m else 0:>10.1f}{flag}")


def do_coco(path):
    d = json.load(open(path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    inst, imgs = collections.Counter(), collections.defaultdict(set)
    for a in d.get("annotations", []):
        n = cats.get(a["category_id"], str(a["category_id"]))
        inst[n] += 1
        imgs[n].add(a["image_id"])
    lic = d.get("licenses")
    info = d.get("info", {})
    note = f"images={len(d.get('images', []))}  licence={lic}  info={str(info)[:120]}"
    report(f"COCO {os.path.basename(path)}", inst, imgs, note)


def do_voc(files):
    inst, imgs = collections.Counter(), collections.defaultdict(set)
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except Exception:
            continue
        for o in root.findall(".//object"):
            n = (o.findtext("name") or "?").strip()
            inst[n] += 1
            imgs[n].add(f)
    report(f"Pascal VOC ({len(files)} xml files)", inst, imgs)


def do_yolo(root, label_files):
    names = None
    for dirpath, _d, fs in os.walk(root):
        for f in fs:
            if f.endswith((".yaml", ".yml")):
                txt = open(os.path.join(dirpath, f), encoding="utf-8", errors="replace").read()
                found = re.findall(r"^\s*-\s*(.+)$", txt, re.M)
                if found:
                    names = [x.strip().strip("'\"") for x in found]
                    break
        if names:
            break
    inst, imgs = collections.Counter(), collections.defaultdict(set)
    for f in label_files:
        for line in open(f, encoding="utf-8", errors="replace"):
            p = line.split()
            if not p:
                continue
            i = int(p[0])
            n = names[i] if names and i < len(names) else f"class_{i}"
            inst[n] += 1
            imgs[n].add(f)
    report(f"YOLO ({len(label_files)} label files)", inst, imgs,
           f"names from data.yaml: {names}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    coco, voc, yolo = [], [], []
    exts = collections.Counter()
    for dirpath, _d, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            e = os.path.splitext(f)[1].lower()
            exts[e] += 1
            if e == ".json" and ("annot" in f.lower() or "instance" in f.lower()
                                 or "train" in f.lower() or "coco" in f.lower()):
                coco.append(p)
            elif e == ".xml":
                voc.append(p)
            elif e == ".txt" and os.sep + "labels" + os.sep in p:
                yolo.append(p)

    print(f"root: {os.path.abspath(root)}")
    print("file types:", dict(exts.most_common(12)))

    for c in coco[:6]:
        try:
            do_coco(c)
        except Exception as ex:
            print(f"  ! {c}: {ex}")
    if voc:
        do_voc(voc)
    if yolo:
        do_yolo(root, yolo)
    if not (coco or voc or yolo):
        print("\nNo annotations found. Top-level entries:")
        for x in sorted(os.listdir(root))[:30]:
            print("   ", x)


if __name__ == "__main__":
    main()
