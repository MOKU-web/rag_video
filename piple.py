import os
import re
import traceback
import cv2
import numpy as np
from paddleocr import PPStructureV3

BASE = r"D:\專題"
SLIDES_DIR = os.path.join(BASE, "slides")
IMGS_DIR = os.path.join(BASE, "imgs")

PADDLE_DEVICE = "cpu"    # 版面分析用 CPU
TEST_SLIDES = []         # 只測試某幾頁，例如 [3, 5]；空清單 = 全部處理

# 測試時另存新檔，避免覆蓋原本的結果
OUTPUT = os.path.join(BASE, "slides_rag_results_test.txt" if TEST_SLIDES else "slides_rag_results.txt")

print(f"正在初始化 PP-StructureV3（{PADDLE_DEVICE}）...")
try:
    pipeline = PPStructureV3(
        device=PADDLE_DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
    )
except Exception as e:
    print("部分參數不支援，改用預設設定：", e)
    pipeline = PPStructureV3(device=PADDLE_DEVICE)

print("✅ PP-StructureV3 初始化完成")


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)   # 支援中文路徑讀取


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 30)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def clean_markdown(raw):
    raw = re.sub(r"<div[^>]*>\s*<img[^>]*>\s*</div>", "", raw, flags=re.S)  # 刪除圖片區塊
    raw = re.sub(r"<img[^>]*>", "", raw)
    raw = re.sub(r"</?div[^>]*>", "", raw)                                  # 移除 div 標籤保留內容
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def parse_slide(img_bgr):
    texts, images, fallback = [], {}, []
    for res in pipeline.predict(img_bgr):
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            t = md.get("markdown_texts", "")
            if isinstance(t, list):
                t = "\n".join(t)
            if t:
                texts.append(t)
            images.update(md.get("markdown_images", {}) or {})
        try:
            fallback.extend(res["overall_ocr_res"]["rec_texts"])
        except Exception:
            pass
    text = clean_markdown("\n".join(texts))
    if not text and fallback:
        print("  ⚠ 版面分析沒有文字，改用整頁 OCR 的文字")
        text = "\n".join(fallback)
    return text, images


def main():
    os.makedirs(IMGS_DIR, exist_ok=True)
    if not os.path.isdir(SLIDES_DIR):
        raise SystemExit(f"找不到資料夾：{SLIDES_DIR}")
    files = sorted(f for f in os.listdir(SLIDES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print("找到投影片截圖", len(files), "張")
    if not files:
        raise SystemExit("資料夾裡沒有圖片")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for idx, name in enumerate(files, 1):
            if TEST_SLIDES and idx not in TEST_SLIDES:
                continue
            print("\n" + "-" * 40)
            print(f"處理投影片 {idx}/{len(files)}：{name}")
            parts = [f"=== [Slide {idx}] 投影片文字與圖表提取 ==="]
            try:
                img = read_image(os.path.join(SLIDES_DIR, name))
                if img is None:
                    raise ValueError("讀不到圖片")
                
                text, images = parse_slide(crop_black(img))
                print(f"  文字 {len(text)} 字，偵測到圖形區塊 {len(images)} 個")
                if text:
                    parts.append(f"--- 【文字與表格內容】 ---\n{text}")

                saved_images = []
                for key, pil_img in images.items():
                    w, h = pil_img.size
                    base = os.path.basename(key)
                    if w < 130 or h < 130 or (w > 800 and h < 450):
                        print(f"  略過 {base}（{w}x{h}，太小或像橫幅）")
                        continue
                    
                    saved_name = f"slide_{idx:05d}_{base}"
                    save_path = os.path.join(IMGS_DIR, saved_name)
                    pil_img.save(save_path)
                    print(f"  🎯 已儲存有效圖形：{saved_name}（{w}x{h}）")
                    saved_images.append(saved_name)

                if saved_images:
                    img_list_str = "\n".join([f"- {img_name}" for img_name in saved_images])
                    parts.append(f"--- 【萃取出的圖片檔案】 ---\n{img_list_str}")

            except Exception:
                traceback.print_exc()
                parts.append("(這一頁處理失敗，錯誤訊息請看終端機)")

            out.write("\n\n".join(parts) + "\n" + "=" * 50 + "\n\n")
            out.flush()
            print(f"✅ 投影片 {idx} 已寫入")

    print("\n🎉 全部完成，輸出：", OUTPUT)


if __name__ == "__main__":
    main()