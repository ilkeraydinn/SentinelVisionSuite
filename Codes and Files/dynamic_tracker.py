import os
import cv2
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO
import threading

# ==========================================
# DYNAMIC THREAT & SUBJECT TRACKER
# ==========================================

RUNS_DIR = "runs/dynamic_tracker"
DATASET_DIR = "Master_Dataset/images"

if not os.path.exists(RUNS_DIR):
    os.makedirs(RUNS_DIR)

print("SİSTEM: Tespit motorları yükleniyor...")
person_model = YOLO('yolov8n.pt')
weapon_model = YOLO('best_model.pt')

class DynamicTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Kurumsal İsimlendirme (AI kelimesi çıkarıldı)
        self.title("Dynamic Threat & Subject Tracker")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.file_list = []
        if os.path.exists(DATASET_DIR):
            self.file_list = [f for f in os.listdir(DATASET_DIR) if
                              f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]

        # --- SOL PANEL ---
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.sidebar, text="DYNAMIC TRACKER", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=30)

        self.search_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Dosya adı ara...", width=250)
        self.search_entry.pack(pady=10)
        self.search_entry.bind("<Return>", lambda e: self.search_files())

        self.search_btn = ctk.CTkButton(self.sidebar, text="Dosyaları Filtrele", command=self.search_files)
        self.search_btn.pack(pady=5)

        self.file_dropdown = ctk.CTkComboBox(self.sidebar, values=self.file_list[:50] if self.file_list else ["Boş"],
                                             width=250, command=self.preview_file)
        self.file_dropdown.pack(pady=20)

        self.run_btn = ctk.CTkButton(self.sidebar, text="ANALİZİ BAŞLAT", command=self.thread_baslat,
                                     font=ctk.CTkFont(weight="bold"), height=50, fg_color="#C0392B",
                                     hover_color="#922B21")
        self.run_btn.pack(pady=20, padx=20, fill="x")

        # Sadeleşmiş Durum Paneli
        self.stats_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E1E1E")
        self.stats_frame.pack(pady=20, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(self.stats_frame, text="Durum: Hazır", text_color="green",
                                         font=ctk.CTkFont(size=15, weight="bold"))
        self.status_label.pack(pady=25)

        # --- SAĞ PANEL ---
        self.viewer_frame = ctk.CTkFrame(self)
        self.viewer_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.viewer_frame.grid_rowconfigure(0, weight=1)
        self.viewer_frame.grid_columnconfigure(0, weight=1)

        self.display_label = ctk.CTkLabel(self.viewer_frame, text="Medya Yükleyin", font=ctk.CTkFont(size=20))
        self.display_label.grid(row=0, column=0)

        self.selected_path = None

    def search_files(self):
        term = self.search_entry.get().lower()
        results = [f for f in self.file_list if term in f.lower()]
        if results:
            self.file_dropdown.configure(values=results[:100])
            self.file_dropdown.set(results[0])
            self.preview_file(results[0])

    def preview_file(self, filename):
        self.selected_path = os.path.join(DATASET_DIR, filename)
        ext = filename.lower()

        if ext.endswith(('.jpg', '.jpeg', '.png')):
            pil_img = Image.open(self.selected_path)
            pil_img.thumbnail((800, 600), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.display_label.configure(image=ctk_img, text="")
            self.display_label.image = ctk_img
        elif ext.endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(self.selected_path)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                pil_img.thumbnail((800, 600), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                self.display_label.configure(image=ctk_img, text="")
                self.display_label.image = ctk_img
            cap.release()

    def thread_baslat(self):
        if not self.selected_path: return

        self.run_btn.configure(state="disabled", text="İŞLENİYOR...")
        self.status_label.configure(text="Durum: İşleniyor...", text_color="orange")

        islem_thread = threading.Thread(target=self.start_process, daemon=True)
        islem_thread.start()

    def apply_detections(self, frame):
        p_res = person_model.predict(frame, classes=[0], conf=0.4, verbose=False)
        w_res = weapon_model.predict(frame, conf=0.50, verbose=False)

        for b in p_res[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "PERSON", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for b in w_res[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "THREAT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def start_process(self):
        idx = 1
        while os.path.exists(f"{RUNS_DIR}/deneme_{idx}"): idx += 1
        save_dir = f"{RUNS_DIR}/deneme_{idx}"
        os.makedirs(save_dir)

        ext = os.path.splitext(self.selected_path)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            self.process_image(self.selected_path, save_dir)
        else:
            self.process_video(self.selected_path, save_dir)

        self.run_btn.configure(state="normal", text="ANALİZİ BAŞLAT")

    def process_image(self, path, save_dir):
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (1024, 768))

        processed_frame = self.apply_detections(frame)

        save_path = os.path.join(save_dir, f"sonuc_{os.path.basename(path)}")
        cv2.imwrite(save_path, processed_frame)

        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        final_pil = Image.fromarray(rgb)
        final_pil.thumbnail((800, 600), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=final_pil, dark_image=final_pil, size=final_pil.size)
        self.display_label.configure(image=ctk_img)
        self.display_label.image = ctk_img
        self.status_label.configure(text="Durum: Analiz Bitti", text_color="green")

    def process_video(self, path, save_dir):
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_path = os.path.join(save_dir, f"islenmis_{os.path.basename(path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed_frame = self.apply_detections(frame)
            out.write(processed_frame)

            preview = cv2.resize(processed_frame, (1024, 768))
            cv2.imshow("Video Analizi (Kapatmak icin 'q' basiniz)", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.status_label.configure(text="Durum: Video Kaydedildi", text_color="green")


if __name__ == "__main__":
    app = DynamicTrackerApp()
    app.mainloop()