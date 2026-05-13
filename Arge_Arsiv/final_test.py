# ==============================================================================
# BİL348 - MACHINE LEARNING TERM PROJECT: FINAL DEBUG & TEST SCRIPT
# BU KOD, MODELİN İÇİNDEN GEÇEN HER KELİMEYİ YAZDIRARAK HATAYI BULMANI SAĞLAR.
# ==============================================================================

import os  # Dosya yolları ve sistem işlemleri için
import re  # Düzenli ifadelerle koordinat ayıklamak için
import torch  # Tensör hesaplamaları ve GPU yönetimi için
from PIL import Image, ImageDraw  # Resim işleme ve görselleştirme için
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)  # Hugging Face model sınıfları
from peft import PeftModel  # Eğitilen LoRA katmanlarını yüklemek için

# 1. KONUM VE DOSYA AYARLARI
# Önceki hataları aşmak için yolu normalize ediyoruz
BASE_MODEL_YOLU = os.path.abspath("C:/paligemma_base")
ADAPTER_PATH = os.path.abspath("./final_colab_expert")
# Yanlış tespit yapılan resim:[cite: 1]
TEST_RESMI = os.path.abspath("Master_Dataset/images/knifes_Knife_scenario_41.png")

# Klasörlerin varlığını fiziksel olarak kontrol ediyoruz
if not os.path.exists(BASE_MODEL_YOLU):
    print(f"KRİTİK HATA: {BASE_MODEL_YOLU} klasörü bulunamadı! Dosyalar taşınmamış.")
    exit()

# 2. MODELİN YÜKLENMESİ (OFFLINE MOD)
print("-" * 50)
print("SİSTEM: Model ve Adaptör yükleniyor, lütfen bekleyin...")
print("-" * 50)

# Bellek kullanımı için 4-bit yapılandırması (GPU verimliliği için)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# İşlemciyi yerel klasörden (C:/paligemma_base) yüklüyoruz[cite: 1]
processor = PaliGemmaProcessor.from_pretrained(
    BASE_MODEL_YOLU,
    local_files_only=True
)

# Ana modeli GPU'ya yüklüyoruz
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL_YOLU,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# Eğittiğin adaptörü ana modelin üzerine entegre ediyoruz
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # Modeli tahmin moduna alıyoruz


# 3. GELİŞMİŞ ANALİZ FONKSİYONU (TEŞHİS MODU)
def analiz_yap(girdi_resmi):
    """
    Modelin ham çıktısını terminale basarak 0 0 0 hatasını inceler.
    """
    # ÖNEMLİ: Eğitimde kullandığın komutun aynısı olmalı.
    # Eğer 'detect knife' kullandıysan burayı 'detect knife' yap.
    prompt = "detect knife\n"

    inputs = processor(text=prompt, images=girdi_resmi, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    # Modelin ürettiği her şeyi (özel tokenlar dahil) decode ediyoruz
    sonuc = processor.decode(output[0], skip_special_tokens=False)

    # Terminale modelin ağzından çıkan her şeyi yazdırıyoruz
    print(f"--- MODEL HAM ÇIKTISI (Bölge Analizi) ---")
    print(f"ÇIKTI: {sonuc}")
    print(f"----------------------------------------")

    # Koordinatları (locXXX) veya rakamları ayıklıyoruz
    return re.findall(r'(\d+)', sonuc)


# 4. İŞLEME VE GÖRSELLEŞTİRME
if os.path.exists(TEST_RESMI):
    # Ana resmi yükle
    img = Image.open(TEST_RESMI).convert("RGB")
    w, h = img.size

    # Odaklanma: Resmin sadece bıçağın olabileceği alt kısmını alıyoruz[cite: 1]
    # (image_d202fd.jpg'deki gibi tüm resme bakınca model yanılıyor)
    odak_yolu = (0, h // 2, w, h)  # Alt yarıya odaklan
    odak_resim = img.crop(odak_yolu)
    ow, oh = odak_resim.size

    print("SİSTEM: Alt bölge taranıyor...")
    rakamlar = analiz_yap(odak_resim)

    # Eğer model en az 4 rakam döndürdüyse
    if len(rakamlar) >= 4:
        # Genelde son 4 rakam koordinatları verir
        y1, x1, y2, x2 = [int(r) for r in rakamlar[-4:]]

        # Koordinatları ana resme geri hesapla
        fx1 = (x1 * ow / 1024)
        fy1 = (y1 * oh / 1024) + (h // 2)
        fx2 = (x2 * ow / 1024)
        fy2 = (y2 * oh / 1024) + (h // 2)

        # Kutu çizimi
        draw = ImageDraw.Draw(img)
        draw.rectangle([fx1, fy1, fx2, fy2], outline="red", width=12)
        draw.text((fx1, fy1 - 40), "DETECTED WEAPON", fill="red")

        print(f"BAŞARI: Koordinatlar bulundu: {y1, x1, y2, x2}")
        img.show()
        img.save("debug_sonuc.png")
    else:
        print("UYARI: Model yine geçerli bir koordinat üretmedi.")
else:
    print(f"HATA: {TEST_RESMI} dosyası bulunamadı.")