import cv2, os
import numpy as np

VIDEO = r"D:\專題\os.mp4"
OUT = r"D:\專題\slides"
SAMPLE_SEC = 0.5      # 每幾秒取樣一次
THRESH_NEW = 8.0      # 換頁判斷：差異超過才算新的一頁
THRESH_STABLE = 1.5   # 穩定判斷（只影響換頁時間）
SKIP_START = 3.0      # 換頁後幾秒內的畫面不選（避開進場動畫）
SKIP_END = 0.5        # 下一頁開始前幾秒內的畫面不選（避開轉場）
MIN_PAGE_SEC = 6.0    # 短於這個秒數的頁，併回上一頁
MERGE_DIFF = 8.0      # 和前一頁的畫面差異低於這個值，就合併成同一頁

os.makedirs(OUT, exist_ok=True)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("打不開影片，請確認路徑")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, int(fps * SAMPLE_SEC))

def small_of(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(g, (320, 180)), (5, 5), 0).astype(np.float32)

# ---- 第一階段：取樣、找換頁點 ----
times, samples = [], []   # samples: (影格編號, 秒數, 內容分數)
ref, prev, prev_t, pending, idx = None, None, 0.0, None, 0

while cap.grab():
    if idx % step == 0:
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur = small_of(frame)
        score = int(np.count_nonzero(cv2.Canny(cv2.resize(gray, (640, 360)), 80, 160)))
        samples.append((idx, t, score))
        if prev is not None:
            if float(np.abs(cur - prev).mean()) >= THRESH_STABLE:
                if pending is None:
                    pending = prev_t
            else:
                if ref is None or float(np.abs(cur - ref).mean()) > THRESH_NEW:
                    times.append(0.0 if ref is None else (pending if pending is not None else t))
                ref = cur
                pending = None
        prev, prev_t = cur, t
    idx += 1

# ---- 第二階段：太短的頁併回上一頁 ----
filtered = []
for t in times:
    if filtered and t - filtered[-1] < MIN_PAGE_SEC:
        continue
    filtered.append(t)
times = filtered

# ---- 第三階段：每頁挑最完整的畫面，太像的就合併 ----
kept = []
for i, start in enumerate(times):
    end = times[i + 1] if i + 1 < len(times) else 1e9
    cand = [s for s in samples if start + SKIP_START <= s[1] < end - SKIP_END]
    if not cand:
        cand = [s for s in samples if start <= s[1] < end] or samples[:1]
    top = max(s[2] for s in cand)
    best = [s for s in cand if s[2] >= top * 0.7][-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, best[0])
    ok, frame = cap.read()
    if not ok:
        continue
    sm = small_of(frame)
    if kept and float(np.abs(sm - kept[-1]["small"]).mean()) < MERGE_DIFF:
        if best[2] > kept[-1]["score"]:
            kept[-1].update(idx=best[0], score=best[2], small=sm)
        print("原第", i + 1, "頁（", round(start, 1), "秒）和前一頁太像，已合併")
        continue
    kept.append(dict(start=start, idx=best[0], score=best[2], small=sm))

# ---- 存圖與時間 ----
for n, k in enumerate(kept, 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k["idx"])
    ok, frame = cap.read()
    if ok:
        cv2.imencode(".png", frame)[1].tofile(os.path.join(OUT, f"{n:05d}.png"))
    print("第", n, "頁 換頁", round(k["start"], 1), "秒，選用影格", k["idx"])

with open(r"D:\專題\scene_log.txt", "w", encoding="utf-8") as f:
    for k in kept:
        f.write(f"pts_time:{k['start']:.3f}\n")
print("完成，共", len(kept), "頁")