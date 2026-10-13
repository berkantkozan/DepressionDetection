import pandas as pd
import numpy as np
import torch
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- 1. GENEL AYARLAR ---
# Eğitilecek modellerin listesi
MODELS_TO_TRAIN = ["bert-base-uncased", "roberta-base"]

# RTX 4050 6GB VRAM için optimize edilmiş ayarlar
BATCH_SIZE = 8       
GRAD_ACCUMULATION = 4 
EPOCHS = 3           
MAX_LEN = 128        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Eğitim Cihazı: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- 2. VERİ YÜKLEME VE HAZIRLIK ---
try:
    df = pd.read_csv('dataset.csv')
    print(f"✅ Veri seti yüklendi: {len(df)} satır.")
except FileNotFoundError:
    print("❌ dataset.csv bulunamadı! Lütfen dosya yolunu kontrol edin.")
    exit()

# Etiketleri Sayısala Çevir
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])
label_map = {index: label for index, label in enumerate(le.classes_)}
num_labels = len(le.classes_)
print(f"🏷️ Sınıflar: {le.classes_}")

# Veriyi Train/Test/Validation olarak ayır
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['statement'].astype(str).tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, 
    temp_labels, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_labels
)

# Metrik Hesaplama Fonksiyonu
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 3. EĞİTİM DÖNGÜSÜ (BERT ve RoBERTa) ---
results_summary = {}

for model_name in MODELS_TO_TRAIN:
    print("\n" + "="*60)
    print(f"🤖 MODEL EĞİTİLİYOR: {model_name.upper()}")
    print("="*60)
    
    # 3.1. Tokenizer (Her modelin kendi tokenizer'ı vardır)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    # Dataset oluşturma
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 3.2. Model Yükleme
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=label_map,
        label2id={v: k for k, v in label_map.items()}
    ).to(device)

    # 3.3. Eğitim Argümanları (Her model ayrı klasöre kaydedilir)
    safe_name = model_name.split('-')[0] # bert veya roberta
    output_dir = f"./results_{safe_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True, # GPU Hızlandırma
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # 3.4. Eğitimi Başlat
    trainer.train()

    # 3.5. Test ve Kayıt
    print(f"\n📊 {model_name} Test Seti Değerlendiriliyor...")
    test_result = trainer.predict(tokenized_test)
    metrics = test_result.metrics
    results_summary[model_name] = metrics
    
    print(f"   -> Accuracy: %{metrics['test_accuracy']*100:.2f}")
    print(f"   -> F1 Score: %{metrics['test_f1']*100:.2f}")

    # Modeli Kalıcı Olarak Kaydet
    final_save_path = f"./finetuned_{safe_name}_final"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"✅ Model kaydedildi: {final_save_path}")

    # 3.6. GPU Belleğini Temizle (Diğer modele yer açmak için kritik!)
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

# --- 4. KARŞILAŞTIRMALI SONUÇ RAPORU ---
print("\n" + "="*60)
print("🏆 NİHAİ KARŞILAŞTIRMA RAPORU")
print("="*60)
print(f"{'MODEL':<20} {'ACCURACY':<15} {'F1 SCORE':<15} {'PRECISION':<15} {'RECALL':<15}")
print("-" * 80)

for model, m in results_summary.items():
    print(f"{model:<20} %{m['test_accuracy']*100:.2f}          %{m['test_f1']*100:.2f}          %{m['test_precision']*100:.2f}          %{m['test_recall']*100:.2f}")

print("-" * 80)