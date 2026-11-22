# rag_video
transcribe.py 將逐字檔寫入檔案 (Whisper)
os.txt時間戳記逐字檔
os.srt 逐字稿字幕格式

grab_stable.py 偵測換頁並截圖的程式
slides（資料夾）	每頁投影片的截圖
scene_log.txt	每頁的換頁時間，格式是 pts_time:秒數，切逐字稿的依據

Piple.py 解讀投影片內容分割為文字、表格、圖型區塊 (PPStructureV3)
Imgs(資料夾) 投影片切分出來的圖形區塊儲存地 
vlm_infer.py圖形區塊解析 (Qwen 2.5-VLM)
vlm_graphic_results.txt 圖形區塊解析成果
slides_rag_results.txt 投影片經分割後整理的文字與表格

merge.py	只依換頁時間切逐字稿的程式（不含投影片文字）
final.txt	依投影片分段的逐字稿
