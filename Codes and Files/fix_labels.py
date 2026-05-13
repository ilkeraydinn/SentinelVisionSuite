import os

# labels klasörünün yolunu buraya yaz
label_yolu = "Master_Dataset/labels"

for dosya in os.listdir(label_yolu):
    if dosya.endswith(".txt"):
        yol = os.path.join(label_yolu, dosya)
        with open(yol, "r") as f:
            satirlar = f.readlines()

        yeni_satirlar = []
        for satir in satirlar:
            parcalar = satir.split()
            if parcalar:
                # Sınıf ID'si 1 (Silah) olsa bile hepsini 0 (Weapon) yapıyoruz
                parcalar[0] = "0"
                yeni_satirlar.append(" ".join(parcalar) + "\n")

        with open(yol, "w") as f:
            f.writelines(yeni_satirlar)

print("İŞLEM TAMAM: Artık 20 bin küsur resmin hepsinde tek bir 'Weapon' sınıfı var!")