import os
import cv2
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading

# ==========================================
# CONTEXTUAL SAFETY ASSESSOR
# ==========================================

BASE_RUNS_DIR = "runs/safety_assessor"
DATASET_DIR = "Master_Dataset/images"

if not os.path.exists(BASE_RUNS_DIR):
    os.makedirs(BASE_RUNS_DIR)

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

print("SİSTEM: Bağlamsal analiz motorları yükleniyor...")
yolo_model = YOLO('best_model.pt')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("SİSTEM: Motorlar hazır! Arayüz başlatılıyor...")


class SafetyAssessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Kurumsal İsimlendirme
        self.title("Contextual Safety Assessor")
        self.geometry("1100x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tum_resimler = []
        if os.path.exists(DATASET_DIR):
            self.tum_resimler = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # --- SOL PANEL ---
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SAFETY ASSESSOR",
                                       font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.search_label = ctk.CTkLabel(self.sidebar_frame, text="Veri Setinde Ara:", text_color="gray")
        self.search_label.grid(row=1, column=0, padx=20, pady=(20, 0), sticky="w")

        self.search_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.search_frame.grid(row=2, column=0, padx=20, pady=(5, 5), sticky="ew")

        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text="Dosya adı yazın...", width=190)
        self.search_entry.pack(side="left", padx=(0, 10))
        self.search_entry.bind("<Return>", lambda event: self.dosya_ara())

        self.search_button = ctk.CTkButton(self.search_frame, text="Ara", width=50, command=self.dosya_ara)
        self.search_button.pack(side="left")

        ilk_liste = self.tum_resimler[:50] if self.tum_resimler else ["Dosya bulunamadı"]
        self.file_dropdown = ctk.CTkComboBox(self.sidebar_frame, values=ilk_liste, command=self.onizleme_yukle,
                                             width=250)
        self.file_dropdown.grid(row=3, column=0, padx=20, pady=(5, 30))

        # Analiz Butonu (Artık Threading kullanıyor)
        self.analyze_button = ctk.CTkButton(
            self.sidebar_frame, text="GÜVENLİK ANALİZİ", command=self.thread_baslat,
            font=ctk.CTkFont(weight="bold"), height=45, fg_color="#C0392B", hover_color="#922B21"
        )
        self.analyze_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Durum: Hazır", text_color="green",
                                         font=ctk.CTkFont(size=14))
        self.status_label.grid(row=5, column=0, padx=20, pady=10)

        self.description_label = ctk.CTkLabel(self.sidebar_frame, text="Bağlamsal Sahne Özeti (VLM):",
                                              text_color="gray")
        self.description_label.grid(row=6, column=0, padx=20, pady=(20, 0), sticky="w")

        self.description_box = ctk.CTkTextbox(self.sidebar_frame, height=120, width=250, wrap="word")
        self.description_box.grid(row=7, column=0, padx=20, pady=10, sticky="n")
        self.description_box.insert("0.0", "Sistem analizi bekleniyor...")
        self.description_box.configure(state="disabled")

        # --- SAĞ PANEL ---
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Resim Seçilmedi", font=ctk.CTkFont(size=20))
        self.image_label.grid(row=0, column=0)

        self.current_image_path = None

        if self.tum_resimler:
            self.file_dropdown.set(self.tum_resimler[0])
            self.onizleme_yukle(self.tum_resimler[0])

    def dosya_ara(self):
        aranan_kelime = self.search_entry.get().lower()

        if not aranan_kelime:
            self.file_dropdown.configure(values=self.tum_resimler[:50])
            self.status_label.configure(text="Durum: Sıfırlandı", text_color="white")
            return

        sonuclar = [f for f in self.tum_resimler if aranan_kelime in f.lower()]

        if sonuclar:
            self.file_dropdown.configure(values=sonuclar[:100])
            self.file_dropdown.set(sonuclar[0])
            self.onizleme_yukle(sonuclar[0])
            self.status_label.configure(text=f"Bulundu: {len(sonuclar)} dosya", text_color="green")
        else:
            self.file_dropdown.configure(values=["Sonuç bulunamadı"])
            self.file_dropdown.set("Sonuç bulunamadı")
            self.status_label.configure(text="Bulunamadı!", text_color="red")

    def onizleme_yukle(self, filename):
        if filename == "Sonuç bulunamadı": return
        self.current_image_path = os.path.join(DATASET_DIR, filename)

        pil_img = Image.open(self.current_image_path)
        pil_img.thumbnail((750, 600), Image.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

        self.status_label.configure(text="Durum: Resim Yüklendi", text_color="white")

        self.description_box.configure(state="normal")
        self.description_box.delete("0.0", "end")
        self.description_box.insert("0.0", "Analiz için hazır...")
        self.description_box.configure(state="disabled")

    def thread_baslat(self):
        if not self.current_image_path: return

        # Butonu kilitle ve donmayı önlemek için thread başlat
        self.analyze_button.configure(state="disabled", text="ANALİZ EDİLİYOR...")
        self.status_label.configure(text="Durum: İşleniyor...", text_color="orange")
        threading.Thread(target=self.analizi_baslat, daemon=True).start()

    def analizi_baslat(self):
        frame = cv2.imread(self.current_image_path)
        frame = cv2.resize(frame, (1024, 768))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 1. YOLO Tarama
        yolo_results = yolo_model.predict(source=frame, conf=0.50, verbose=False)
        threat_detected = False

        for box in yolo_results[0].boxes:
            threat_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"WEAPON {float(box.conf[0]):.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        # 2. VLM Tarama
        inputs = processor(pil_image, return_tensors="pt")
        out = vlm_model.generate(**inputs, max_new_tokens=50)
        vlm_caption = processor.decode(out[0], skip_special_tokens=True).capitalize()

        # 3. Sonuç Çizimi
        if threat_detected:
            status_text = "STATUS: DANGER!"
            color = (0, 0, 255)
        else:
            status_text = "STATUS: SAFE"
            color = (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Scene: {vlm_caption}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Otomatik Kayıt (Yeni Klasöre)
        folder_idx = 1
        while os.path.exists(f"{BASE_RUNS_DIR}/deneme_{folder_idx}"):
            folder_idx += 1
        save_dir = f"{BASE_RUNS_DIR}/deneme_{folder_idx}"
        os.makedirs(save_dir)

        filename = os.path.basename(self.current_image_path)
        save_path = f"{save_dir}/assessed_{filename}"
        cv2.imwrite(save_path, frame)

        # 5. Arayüzü Güncelleme
        processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)

        processed_pil.thumbnail((750, 600), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=processed_pil, dark_image=processed_pil, size=processed_pil.size)

        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img

        self.status_label.configure(text=status_text, text_color="red" if threat_detected else "green")
        self.description_box.configure(state="normal")
        self.description_box.delete("0.0", "end")
        self.description_box.insert("0.0", vlm_caption)
        self.description_box.configure(state="disabled")

        # İşlem bitince butonu aç
        self.analyze_button.configure(state="normal", text="GÜVENLİK ANALİZİ")


if __name__ == "__main__":
    app = SafetyAssessorApp()
    app.mainloop()