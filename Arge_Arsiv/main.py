import os
import re  # Metin içinden koordinatları çekmek için ekledik
from huggingface_hub import login
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from PIL import Image, ImageDraw  # Çizim yapmak için ImageDraw'u ekledik
import time

# 1. Klasör ve İndirme Ayarları
YENI_KLASOR = "C:/huggingface_cache"
os.environ["HF_HOME"] = YENI_KLASOR
os.environ["HUGGINGFACE_HUB_CACHE"] = YENI_KLASOR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Token girişin (GÜVENLİK NOTU: Proje bitince Hugging Face'ten bu token'ı silmeyi unutma)
login("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")

# 2. Donanım ve Model Yükleme (RTX 4050 için)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Başlatılıyor... Kullanılan donanım: {device.upper()}")

baslangic_zamani = time.time()
model_id = "google/paligemma-3b-pt-224"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

print("Model ekran kartına (RTX 4050) yükleniyor...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    cache_dir=YENI_KLASOR
)
model.eval()

processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=YENI_KLASOR)

yukleme_suresi = time.time() - baslangic_zamani
print(f"Model {yukleme_suresi:.2f} saniyede belleğe yüklendi.\n")

# 3. Resim İnceleme ve Çizim Aşaması
resim_yolu = "images/knifes_Knife_scenario_70.png"

if not os.path.exists(resim_yolu):
    print(f"HATA: '{resim_yolu}' bulunamadı.")
else:
    print(f"'{resim_yolu}' inceleniyor...")
    resim = Image.open(resim_yolu).convert("RGB")

    # Uyarı vermemesi için <image> etiketini ekledik ve koordinat istedik
    # Sadece bu satırı değiştir:
    prompt = "<image>detect knife"
    inputs = processor(text=prompt, images=resim, return_tensors="pt").to(device)

    print("Yapay Zeka düşünüyor...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    # Sorunun papağan gibi tekrarlanmasını engelliyoruz
    ham_sonuc = processor.decode(output[0], skip_special_tokens=True)
    sonuc = ham_sonuc.replace(prompt.replace("<image>", ""), "").strip()

    print("-" * 50)
    print("📝 MODELİN ANALİZİ:")
    print(sonuc)
    print("-" * 50)

    # 4. Kırmızı Kutu Çizme (Bounding Box) İşlemi
    koordinatlar = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', sonuc)

    if koordinatlar:
        # 0-1000 formatındaki veriyi resmin kendi piksellerine çeviriyoruz
        x1, y1, x2, y2 = [int(deger) for deger in koordinatlar[0]]
        genislik, yukseklik = resim.size

        gercek_x1 = int(x1 * genislik / 1000)
        gercek_y1 = int(y1 * yukseklik / 1000)
        gercek_x2 = int(x2 * genislik / 1000)
        gercek_y2 = int(y2 * yukseklik / 1000)

        # Kırmızı ve kalın (5px) bir çizgi çekiyoruz
        cizim = ImageDraw.Draw(resim)
        cizim.rectangle([gercek_x1, gercek_y1, gercek_x2, gercek_y2], outline="red", width=5)

        print("Görsel işaretleme başarılı! Resim açılıyor...")
        resim.show()  # Windows fotoğraflar uygulamasında resmi açar
    else:
        print("Model tehdidi algıladı ancak kesin bir koordinat vermediği için çizim yapılamadı.")