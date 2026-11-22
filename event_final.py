import re, json, os, csv
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

FINAL = r"D:\專題\final.txt"
GLOSSARY = r"D:\專題\glossary.csv"      # 沒有這個檔案也能跑，只是知識點沒有學術名稱

ALPHA = 0.6                 # 逐字稿相似度權重；投影片內容權重 = 1 - ALPHA
SMOOTH = 2                  # 新穎度曲線的高斯平滑
LOW = 0.2                   # 換頁處：新穎度高於此值才保留為知識點邊界（越高併越多頁）
HIGH = 1.5                  # 同一頁內：新穎度高於此值才額外切開（越低切越細）
MIN_SPLIT_SLIDE_SEC = 120   # 投影片停留超過幾秒，才允許頁內切分
MIN_GAP_SEC = 60            # 頁內切分點離其他邊界至少幾秒

def mmss(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)

# ---------- 1. 讀取 final.txt ----------
text = open(FINAL, encoding="utf-8").read()
parts = re.split(r"(?m)^===== 投影片 (\d+)\s+\[(\d+:\d+) - ([^\]]+)\] =====\n", text)
TS = re.compile(r"^\[(\d+(?:\.\d+)?)\s*->\s*(\d+(?:\.\d+)?)\]\s*(.*)$")

slides = []
for i in range(1, len(parts), 4):
    idx, st, en, body = int(parts[i]), mmss(parts[i + 1]), parts[i + 2], parts[i + 3]
    slide_text, _, trans = body.partition("【逐字稿】")
    slide_text = slide_text.replace("【投影片文字】", "").strip()
    lines = [l.strip() for l in trans.split("\n") if l.strip()]
    ts_lines = []
    for l in lines:
        m = TS.match(l)
        if m:
            ts_lines.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    slides.append(dict(slide=idx, start=float(st),
                       end=float(mmss(en)) if ":" in en else None,
                       slide_text=slide_text, lines=lines, ts_lines=ts_lines))
if not slides:
    raise SystemExit("final.txt 格式不符，請把前 30 行貼給我")

for s in slides:                       # 補上「結尾」那頁的結束時間
    if s["end"] is None:
        s["end"] = s["ts_lines"][-1][1] if s["ts_lines"] else s["start"] + 60.0

n_lines = sum(len(s["lines"]) for s in slides)
n_ts = sum(len(s["ts_lines"]) for s in slides)
line_mode = n_lines > 0 and n_ts >= 0.8 * n_lines
print("模式：", "逐行（有時間戳）" if line_mode else "以投影片為單位（沒有逐行時間戳）")

# ---------- 2. 建立單位 ----------
units = []      # (開始秒, 結束秒, 文字, 所屬投影片在 slides 裡的位置)
for pos, s in enumerate(slides):
    if line_mode:
        for a, b, t in s["ts_lines"]:
            units.append((a, b, t, pos))
    else:
        units.append((s["start"], s["end"], " ".join(s["lines"]), pos))
n = len(units)
if n < 2:
    raise SystemExit("資料太少，無法偵測")

HALF, WIN = (8, 3) if line_mode else (2, 1)
SIGMA = HALF / 2

# ---------- 3. 向量化 ----------
model = SentenceTransformer("BAAI/bge-m3")
texts = [" ".join(u[2] for u in units[i:i + WIN]) for i in range(n)]
Et = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
Es = model.encode([(s["slide_text"][:1000] or " ") for s in slides],
                  normalize_embeddings=True, batch_size=16)
Es_u = Es[[u[3] for u in units]]

# ---------- 4. 融合相似度 + 高斯棋盤格核新穎度 ----------
S = ALPHA * (Et @ Et.T) + (1 - ALPHA) * (Es_u @ Es_u.T)
kidx = np.arange(-HALF, HALF) + 0.5
w = np.sign(kidx) * np.exp(-kidx ** 2 / (2 * SIGMA ** 2))
K = np.outer(w, w)
Sp = np.pad(S, HALF, mode="edge")
nov = np.array([(K * Sp[i:i + 2 * HALF, i:i + 2 * HALF]).sum() for i in range(n)])
nov = gaussian_filter1d(nov, sigma=SMOOTH)
nov = (nov - nov.mean()) / (nov.std() + 1e-9)

# ---------- 5. 以投影片換頁為骨架決定邊界（用單位編號，不用時間，避免誤差）----------
first_idx = {}
for i, u in enumerate(units):
    first_idx.setdefault(u[3], i)

bound_idx = [0]

# 5-1 換頁處：新穎度夠高才保留，否則併入前一個知識點
for pos in range(1, len(slides)):
    f = first_idx.get(pos)
    if f is None or f == 0:
        continue
    score = float(nov[max(0, f - 2): f + 3].max())
    keep = score >= LOW
    print(f"換頁 → 投影片{slides[pos]['slide']}　新穎度 {score:.2f}　{'保留' if keep else '併入前一頁'}")
    if keep:
        bound_idx.append(f)

# 5-2 同一頁內的額外切分（只在逐行模式、停留很久的投影片）
if line_mode:
    peaks, _ = find_peaks(nov, height=HIGH)
    for p in peaks:
        pos = units[p][3]
        t = units[p][0]
        if slides[pos]["end"] - slides[pos]["start"] < MIN_SPLIT_SLIDE_SEC:
            continue
        if min(abs(t - units[b][0]) for b in bound_idx) < MIN_GAP_SEC:
            continue
        bound_idx.append(int(p))
        print(f"頁內切分：投影片{slides[pos]['slide']}　{int(t)//60:02d}:{int(t)%60:02d}")

bound_idx = sorted(set(bound_idx))

# ---------- 6. 術語表（選用）----------
terms = []
if os.path.exists(GLOSSARY):
    terms = list(csv.DictReader(open(GLOSSARY, encoding="utf-8-sig")))
    T = model.encode([f"{t['term']} {t.get('aliases', '')}" for t in terms],
                     normalize_embeddings=True)

# ---------- 7. 組出知識點 ----------
slide_nums = [s["slide"] for s in slides]
result = []
for k, b in enumerate(bound_idx):
    e = bound_idx[k + 1] if k + 1 < len(bound_idx) else n
    members = list(range(b, e))
    if not members:
        continue
    nums = sorted(set(slides[units[i][3]]["slide"] for i in members))
    nxt = slides[units[bound_idx[k + 1]][3]]["slide"] if k + 1 < len(bound_idx) else slide_nums[-1] + 1
    nums += list(range(nums[-1] + 1, nxt))                 # 沒有逐字稿的投影片也歸進來
    if k == 0:
        nums = list(range(slide_nums[0], nums[0])) + nums
    by_num = {s["slide"]: s for s in slides}
    item = {
        "kp_id": len(result) + 1,
        "start": units[b][0],
        "end": units[e - 1][1],
        "slides": nums,
        "slide_content": "\n".join(f"[投影片{x}] {by_num[x]['slide_text']}" for x in nums if x in by_num),
        "transcript": " ".join(units[i][2] for i in members),
    }
    if terms:
        c = Et[members].mean(axis=0) + Es_u[members].mean(axis=0)
        c /= np.linalg.norm(c)
        top = (T @ c).argsort()[::-1][:3]
        item["glossary_terms"] = [terms[j]["term"] for j in top]
        item["chapter"] = terms[top[0]].get("chapter", "")
    result.append(item)

# ---------- 8. 輸出 ----------
json.dump(result, open(r"D:\專題\knowledge_segments.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
with open(r"D:\專題\knowledge_segments.txt", "w", encoding="utf-8") as f:
    for r in result:
        a, b2 = int(r["start"]), int(r["end"])
        f.write(f"===== 知識點 {r['kp_id']}　[{a//60:02d}:{a%60:02d} - {b2//60:02d}:{b2%60:02d}]"
                f"　投影片 {r['slides'][0]}~{r['slides'][-1]}")
        if "glossary_terms" in r:
            f.write("　" + "、".join(r["glossary_terms"]))
        f.write(" =====\n【投影片內容】\n" + r["slide_content"] + "\n【逐字稿】\n" + r["transcript"] + "\n\n")
print("找到", len(result), "個知識點片段")