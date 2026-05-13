import json
import re
import os

# Ayarlar
JSON_DOSYASI = "dataset.json"  # Eğer json dosyanın adı farklıysa burayı düzelt
ETIKET_KLASORU = "labels"  # .txt dosyalarının kaydedileceği klasör
RESIM_GENISLIK = 1000  # Senin json formatında koordinatlar 0-1000 arası verilmiş.
RESIM_YUKSEKLIK = 1000  # Eğer resimlerin gerçek boyutu farklıysa buraları ona göre güncelle.

# Etiket klasörünü oluştur
if not os.path.exists(ETIKET_KLASORU):
    os.makedirs(ETIKET_KLASORU)

print("SİSTEM: Dönüştürme işlemi başlıyor...")

try:
    with open(JSON_DOSYASI, 'r', encoding='utf-8') as f:
        veri = json.load(f)

    donusturulen_sayisi = 0

    for oge in veri:
        resim_adi = oge.get("image")
        if not resim_adi:
            continue

        # .png veya .jpg uzantısını silip .txt yapıyoruz
        txt_adi = os.path.splitext(resim_adi)[0] + ".txt"
        txt_yolu = os.path.join(ETIKET_KLASORU, txt_adi)

        # "vlm" (modelin cevabı) kısımlarını tarayarak koordinat arıyoruz
        for konusma in oge.get("conversations", []):
            if konusma.get("from") == "vlm":
                metin = konusma.get("value")
                # "[x1, y1, x2, y2]" formatını arayan regex (Örn: [290, 708, 354, 743])
                eslesme = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', metin)

                if eslesme:
                    # x1, y1, x2, y2 değerlerini çek
                    x1, y1, x2, y2 = map(int, eslesme.groups())

                    # Sınıfı belirle (Basitçe: metinde 'knife' geçiyorsa 0, 'gun' geçiyorsa 1)
                    # Sen kendi sınıflarına göre burayı ayarlayabilirsin.
                    sinif_id = 0 if "knife" in metin.lower() else (1 if "gun" in metin.lower() else 0)

                    # --- YOLO MATEMATİĞİ (Merkez X, Merkez Y, Genişlik, Yükseklik) ---
                    # 1. Kutunun merkez noktasını bul (0-1000 arası değerler)
                    merkez_x = (x1 + x2) / 2.0
                    merkez_y = (y1 + y2) / 2.0

                    # 2. Kutunun genişliğini ve yüksekliğini bul
                    genislik = x2 - x1
                    yukseklik = y2 - y1

                    # 3. Değerleri 0 ile 1 arasına normalize et (YOLO formatı bunu ister)
                    norm_merkez_x = merkez_x / RESIM_GENISLIK
                    norm_merkez_y = merkez_y / RESIM_YUKSEKLIK
                    norm_genislik = genislik / RESIM_GENISLIK
                    norm_yukseklik = yukseklik / RESIM_YUKSEKLIK

                    # .txt dosyasına yaz (Örn: 0 0.322 0.7255 0.064 0.035)
                    with open(txt_yolu, 'a', encoding='utf-8') as tf:
                        tf.write(
                            f"{sinif_id} {norm_merkez_x:.6f} {norm_merkez_y:.6f} {norm_genislik:.6f} {norm_yukseklik:.6f}\n")

                    donusturulen_sayisi += 1
                    break  # Bir resimde ilk koordinatı bulduktan sonra diğer konuşmalara bakmasına gerek yok.

    print(f"BAŞARI: Toplam {donusturulen_sayisi} adet resmin etiketi başarıyla YOLO formatına çevrildi!")
    print(f"Lütfen resimlerin bulunduğu klasörün içindeki '{ETIKET_KLASORU}' klasörünü kontrol et.")

except Exception as e:
    print(f"HATA OLUŞTU: {e}")