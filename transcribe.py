from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    r"D:\專題\os.wav",
    language="zh",
    vad_filter=True,
    initial_prompt="以下是繁體中文的作業系統課程。",
)

def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

with open(r"D:\專題\os.txt", "w", encoding="utf-8") as ftxt, \
     open(r"D:\專題\os.srt", "w", encoding="utf-8") as fsrt:
    for i, s in enumerate(segments, 1):
        text = s.text.strip()
        print(f"[{s.start:.1f} -> {s.end:.1f}] {text}")
        ftxt.write(f"[{s.start:.1f} -> {s.end:.1f}] {text}\n")
        ftxt.flush()
        fsrt.write(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{text}\n\n")
        fsrt.flush()