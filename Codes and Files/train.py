import os
import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, LoraConfig, get_peft_model # PeftModel eklendi
from datasets import Dataset
import json
from PIL import Image

# 1. Klasör Ayarları
YENI_KLASOR = "C:/huggingface_cache"
os.environ["HF_HOME"] = YENI_KLASOR
# Daha önce eğitilen ağırlıkların klasörü
ESKI_AGIRLIKLAR = "./paligemma_weapon_project/checkpoint-57"

# 2. Model ve İşlemci Yükleme (4-bit Optimize)
model_id = "google/paligemma-3b-pt-224"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("SİSTEM: Ana model yükleniyor...")
processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=YENI_KLASOR)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=YENI_KLASOR
)

# 3. ÖNEMLİ: ESKİ EĞİTİMİ ÜZERİNE GİYDİRME
# TrainerState dosyası olmadığı için adaptörü manuel olarak yüklüyoruz
print(f"SİSTEM: {ESKI_AGIRLIKLAR} klasöründeki zeka yükleniyor...")
model = PeftModel.from_pretrained(base_model, ESKI_AGIRLIKLAR, is_trainable=True)

# 4. Veri Hazırlama (Aynı işlemler)
def veri_hazirla(json_yolu):
    with open(json_yolu, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset_list = []
    for item in data:
        dataset_list.append({
            "image": os.path.join("images", item['image']),
            "prefix": f"<image>{item['conversations'][0]['value']}",
            "suffix": item['conversations'][1]['value']
        })
    return Dataset.from_list(dataset_list)

dataset = veri_hazirla("dataset.json")

def preprocess(examples):
    images = [Image.open(p).convert("RGB") for p in examples["image"]]
    return processor(text=examples["prefix"], images=images, suffix=examples["suffix"],
                     return_tensors="pt", padding="max_length", max_length=512, truncation=True)

print("SİSTEM: Veriler işleniyor...")
train_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 5. Eğitim Ayarları (10 Epoch - Koordinatları öğrenmesi için şart)
training_args = TrainingArguments(
    output_dir="./paligemma_weapon_project_v2", # Karışmaması için yeni klasör
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=10, # Hedef: Kareyi düzgün çizmesi
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    report_to="none"
)

# 6. Eğitimi Başlat (resume_from_checkpoint KALDIRILDI)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("-" * 50)
print("BİL348: EK EĞİTİM BAŞLIYOR (Model artık daha tecrübeli)...")
print("-" * 50)

# Artık resume_from_checkpoint kullanmıyoruz, çünkü modeli yukarıda yükledik
trainer.train()

# 7. Final Uzmanı Kaydet
model.save_pretrained("./paligemma_weapon_project/final_expert_v2")
print("TEBRİKLER! Gelişmiş uzman model 'final_expert_v2' klasörüne kaydedildi.")