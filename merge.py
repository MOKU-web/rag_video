import re

def time_to_seconds(time_str):
    """將 SRT 時間格式 (00:01:23,456) 轉為總秒數 (float)"""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def parse_srt(srt_path):
    """解析 os.srt，回傳包含起始時間、結束時間與文字的串列"""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(srt_path, "r", encoding="big5") as f:
            content = f.read()

    # 匹配 SRT 格式
    pattern = r"\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    subtitles = []
    for start, end, text in matches:
        subtitles.append({
            "start_str": start,
            "end_str": end,
            "start": time_to_seconds(start),
            "end": time_to_seconds(end),
            "text": text.replace("\n", " ").strip()
        })
    return subtitles

def parse_scene_log(log_path):
    """解析 scene_log.txt 取得每頁換頁時間（支援 pts_time:秒數 格式）"""
    slides = []
    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # 抓取數值（支援 float 秒數）
            match = re.search(r"([\d.]+)", line)
            if match:
                slides.append({
                    "slide_index": idx + 1,
                    "start_time": float(match.group(1))
                })
    
    # 計算每一頁的結束時間
    for i in range(len(slides) - 1):
        slides[i]["end_time"] = slides[i + 1]["start_time"]
    if slides:
        slides[-1]["end_time"] = 999999.0  # 最後一頁直到影片結束
        
    return slides

def main():
    print("正在讀取 scene_log.txt 與 os.srt...")
    slides = parse_scene_log("scene_log.txt")
    subtitles = parse_srt("os.srt")

    print(f"成功載入 {len(slides)} 張投影片切換點，以及 {len(subtitles)} 句逐字稿。")

    # 將逐字稿分配到對應的投影片區間並寫入 final.txt
    with open("final.txt", "w", encoding="utf-8") as f_out:
        for slide in slides:
            slide_idx = slide["slide_index"]
            start_t = slide["start_time"]
            end_t = slide["end_time"] if slide["end_time"] != 999999.0 else "影片結束"

            # 寫入投影片大標題與時間範圍
            f_out.write(f"=== [Slide {slide_idx}] 區間: {start_t}秒 ~ {end_t}秒 ===\n")

            # 篩選屬於這張投影片時間範圍內的逐字稿句子
            slide_subs = [
                sub for sub in subtitles 
                if start_t <= sub["start"] < slide["end_time"]
            ]

            if not slide_subs:
                f_out.write("(此區間無對應逐字稿)\n\n")
                continue

            # 逐句寫入帶時間戳的逐字稿
            for sub in slide_subs:
                f_out.write(f"[{sub['start_str']} --> {sub['end_str']}] {sub['text']}\n")
            
            f_out.write("\n" + "="*50 + "\n\n")

    print("✓ 整合完成！已成功將對齊後的帶時間戳逐字稿寫入 final.txt")

if __name__ == "__main__":
    main()