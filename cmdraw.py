import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

# --- AYARLAR ---
# Daha önce kaydettiğin model klasörlerinin isimleri (kodunda bu isimlerle kaydetmiştin)
SAVED_MODELS = ["./finetuned_bert_final", "./finetuned_roberta_final"]
BATCH_SIZE = 16
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. VERİYİ TEKRAR HAZIRLA (Aynı Test Setini Bulmak İçin) ---
# Not: random_state=42 olduğu için aynı veriler gelecektir.
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("❌ dataset.csv bulunamadı!")
    exit()

le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])
classes = le.classes_ # Sınıf isimlerini al (Anxiety, Bipolar vs.)

# Veriyi aynı şekilde bölüyoruz
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['statement'].astype(str).tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# Test setini elde et
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, 
    temp_labels, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_labels
)

# Test datasını Dataset formatına çevir
test_dataset_raw = Dataset.from_dict({"text": test_texts, "label": test_labels})

# --- 2. MODELLERİ YÜKLE VE MATRİS ÇİZ ---
for model_path in SAVED_MODELS:
    print(f"\n📂 Model Yükleniyor: {model_path}...")
    
    try:
        # Kayıtlı Tokenizer ve Modeli Yükle
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    except OSError:
        print(f"⚠️ {model_path} bulunamadı, bu model atlanıyor.")
        continue

    # Tokenize Fonksiyonu
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    # Test verisini tokenize et
    tokenized_test = test_dataset_raw.map(tokenize_function, batched=True)

    # Trainer kullanarak sadece tahmin (predict) yap
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_test)
    
    # En yüksek olasılıklı sınıfı seç
    y_preds = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # --- 3. CONFUSION MATRIX ÇİZİMİ ---
    cm = confusion_matrix(y_true, y_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    model_name = "RoBERTa" if "roberta" in model_path else "BERT"
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    
    # Resmi kaydet ve göster
    save_filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(save_filename)
    print(f"✅ Grafik kaydedildi: {save_filename}")
    plt.show()