import os
import pandas as pd
import re

def srt_to_csv(srt_path, csv_path):
    with open(srt_path, 'r') as f:
        content = f.read()

    entries = re.split(r'\n\s*\n', content.strip())
    rows = []

    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue
        time_line = lines[1]
        label = lines[2].strip()

        start, end = time_line.split(' --> ')
        start = start.replace(",", ".")
        end = end.replace(",", ".")

        rows.append({
            "Start Time": start,
            "End Time": end,
            "Text": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Converted {os.path.basename(srt_path)} → {os.path.basename(csv_path)}")

# === Batch convert all ===
if __name__ == "__main__":
    srt_folder = "data/timetable/srt"
    csv_folder = "data/timetable/csv"
    os.makedirs(csv_folder, exist_ok=True)

    for filename in os.listdir(srt_folder):
        if filename.endswith(".srt") and "_vrew" in filename:
            uid = filename.split("_")[0]
            srt_file = os.path.join(srt_folder, filename)
            csv_file = os.path.join(csv_folder, f"user{uid}.csv")
            srt_to_csv(srt_file, csv_file)
