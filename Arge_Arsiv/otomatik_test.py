from ultralytics import YOLO

# Yeni eğittiğimiz 10 bin resimlik canavarı yüklüyoruz
model = YOLO('best_model.pt')

# YOLO'nun kendi tahmin ve kaydetme motoru
sonuclar = model.predict(
    source='Master_Dataset/images/video1.mp4', # İstersen buraya tek bir resim veya içi resim/video dolu bir klasör yolu da verebilirsin
    conf=0.15,      # Güven skoru (Çok iyi değil dediğin için biraz daha düşürebilir veya artırabilirsin)
    save=True,      # İŞTE KRİTİK NOKTA: runs/detect/predict içine otomatik kaydeder
    save_txt=False, # True yaparsan tespit koordinatlarını .txt olarak da verir
    show=True       # İşlem sırasında ekranda canlı gösterir
)

print("İşlem tamam! Sonuçlar runs/detect/predict klasörüne kaydedildi.")