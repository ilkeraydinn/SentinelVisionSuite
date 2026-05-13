import json
import os
from PIL import Image

# AYARLAR
DATASET_PATH = "dataset.json"  # Mevcut JSON dosyan
IMAGE_DIR = "images/"  # Resimlerin olduğu klasör
OUTPUT_DIR = "power_images/"  # Yeni kesilmiş resimlerin gideceği yer
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []

for entry in data:
    img_name = entry['image'].split('/')[-1]
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path): continue

    # Koordinatları çek (JSON formatına göre: y1, x1, y2, x2)
    # Senin attığın formatta [598, 536, 707, 629] şeklindeydi
    val_str = entry['conversations'][1]['value']
    nums = [int(n) for n in
            [n for n in val_str.replace('[', '').replace(']', '').split() if n.isdigit()] or [0, 0, 0, 0]]
    if len(nums) < 4: continue

    y1, x1, y2, x2 = nums
    img = Image.open(img_path)
    w, h = img.size

    # 0-1000'den gerçek piksele çevir
    py1, px1 = int(y1 * h / 1000), int(x1 * w / 1000)
    py2, px2 = int(y2 * h / 1000), int(x2 * w / 1000)

    # Bıçağın etrafından %20 pay bırakarak kes (Zoom yap)
    pad = 100
    left = max(0, px1 - pad);
    top = max(0, py1 - pad)
    right = min(w, px2 + pad);
    bottom = min(h, py2 + pad)

    crop_img = img.crop((left, top, right, bottom))
    crop_name = f"zoom_{img_name}"
    crop_img.save(os.path.join(OUTPUT_DIR, crop_name))

    # Yeni koordinatları hesapla (0-1000 arası)
    new_w, new_h = crop_img.size
    ny1 = int((py1 - top) * 1000 / new_h)
    nx1 = int((px1 - left) * 1000 / new_w)
    ny2 = int((py2 - top) * 1000 / new_h)
    nx2 = int((px2 - left) * 1000 / new_w)

    # Yeni JSON girişini oluştur
    entry['image'] = f"power_images/{crop_name}"
    entry['conversations'][1]['value'] = f"Knife detected at [{ny1} {nx1} {ny2} {nx2}]"
    new_data.append(entry)

with open("dataset_power.json", "w", encoding='utf-8') as f:
    json.dump(new_data, f, indent=4)

print(f"BİTTİ! {len(new_data)} adet yakın çekim veri hazır.")