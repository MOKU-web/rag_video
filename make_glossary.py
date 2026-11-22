import re, csv
from collections import Counter

text = open(r"D:\專題\final.txt", encoding="utf-8").read()
slide_text = "\n".join(re.findall(r"【投影片文字】\n(.*?)【逐字稿】", text, flags=re.S))

# 1. 中文(English) 對照
names, aliases = {}, {}
for zh, en in re.findall(r"([\u4e00-\u9fff]{2,10})\s*[\(（]\s*([A-Za-z][A-Za-z0-9 /\-]{1,40}?)\s*[\)）]", slide_text):
    key = en.strip().lower()
    names.setdefault(key, en.strip())
    aliases.setdefault(key, set()).add(zh)

# 2. 投影片上出現的英文詞（含大寫字母的才算，濾掉一般小寫單字）
counter, orig = Counter(), {}
for p in re.findall(r"[A-Za-z][A-Za-z0-9/\-]*(?: [A-Za-z][A-Za-z0-9/\-]*){0,3}", slide_text):
    if len(p) >= 2 and any(c.isupper() for c in p):
        counter[p.lower()] += 1
        orig.setdefault(p.lower(), p)

rows = []
for key in set(counter) | set(names):
    rows.append({
        "term": names.get(key, orig.get(key, key)),
        "aliases": " ".join(sorted(aliases.get(key, []))),
        "chapter": "",
        "count": counter.get(key, 0),
    })
rows.sort(key=lambda r: -r["count"])

with open(r"D:\專題\glossary_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["term", "aliases", "chapter", "count"])
    w.writeheader()
    w.writerows(rows[:200])
print("候選術語", len(rows), "個，已輸出前 200 個")