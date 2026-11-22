import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

BASE = r"D:\專題"
IMGS_DIR = os.path.join(BASE, "imgs")
OUTPUT = os.path.join(BASE, "vlm_graphic_results.txt")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "請用繁體中文描述圖形的結構、各元件的含義，以及它想表達的核心概念，200 字以內。"
    "圖上看得清楚的文字請照抄；看不清楚的文字請寫「看不清楚」，不要猜測。"
)

print("CUDA 可用：", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

print("正在載入 Qwen2.5-VL 模型（請稍候）...")
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
vlm_processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=640 * 28 * 28  # 限制圖片大小，節省顯存
)
print("✅ Qwen2.5-VL 模型載入完成")


def describe_graphic(pil_img):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img.convert("RGB")},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(vlm_model.device)
    
    with torch.inference_mode():
        ids = vlm_model.generate(**inputs, max_new_tokens=512)
    
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    out = vlm_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out.strip()


def main():
    if not os.path.isdir(IMGS_DIR):
        raise SystemExit(f"找不到圖片資料夾：{IMGS_DIR}，請先執行 OCR 切圖腳本！")
    
    files = sorted(f for f in os.listdir(IMGS_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"在 imgs/ 資料夾中找到 {len(files)} 張切片圖片")
    if not files:
        raise SystemExit("資料夾裡沒有圖片可供解析")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for idx, name in enumerate(files, 1):
            img_path = os.path.join(IMGS_DIR, name)
            print("\n" + "-" * 40)
            print(f"正在解析第 {idx}/{len(files)} 張圖片：{name}")
            
            try:
                pil_img = Image.open(img_path)
                w, h = pil_img.size
                print(f"  圖片尺寸：{w}x{h}")
                
                # 呼叫 Qwen 解析
                desc = describe_graphic(pil_img)
                print(f"  ✨ 解析完成")
                
                # 寫入結果檔案
                out.write(f"=== [Image: {name}] ===\n")
                out.write(f"{desc}\n")
                out.write("=" * 50 + "\n\n")
                out.flush()
                
            except Exception as e:
                print(f"  ❌ 解析失敗：{e}")
                out.write(f"=== [Image: {name}] ===\n(解析失敗)\n" + "=" * 50 + "\n\n")

    print(f"\n🎉 全部圖形解析完成，結果已輸出至：{OUTPUT}")


if __name__ == "__main__":
    main()