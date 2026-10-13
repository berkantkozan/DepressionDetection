import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. AYARLAR VE CİHAZ SEÇİMİ ---
model_name = "bert-base-uncased" # İngilizce metinler için standart BERT
# Eğer Türkçe çalışıyorsanız "dbmdz/bert-base-turkish-cased" kullanabilirsiniz.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")
if device.type == 'cpu':
    print("UYARI: GPU bulunamadı. Eğitim işlemi CPU üzerinde çok yavaş olabilir!")

# --- 2. VERİ YÜKLEME VE HAZIRLIK ---
try:
    df = pd.read_csv('dataset.csv')
    print("Veri seti yüklendi.")
except FileNotFoundError:
    print("HATA: dataset.csv bulunamadı.")
    exit()

# Etiketleri Sayısallaştırma
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['status'])
class_names = encoder.classes_
num_labels = len(class_names)

# Metin ve Etiketleri Ayırma
X = df['statement'].astype(str).tolist()
y = df['label'].tolist()

# Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim Verisi: {len(X_train)}, Test Verisi: {len(X_test)}")

# --- 3. BERT TOKENIZATION ---
print("Tokenization işlemi yapılıyor (Bu işlem biraz sürebilir)...")
tokenizer = BertTokenizer.from_pretrained(model_name)

# Veriyi BERT formatına uygun hale getiren fonksiyon
class DepressionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Verileri Tokenize Et
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Dataset Objelerini Oluştur
train_dataset = DepressionDataset(train_encodings, y_train)
test_dataset = DepressionDataset(test_encodings, y_test)

# --- 4. SINIF DENGESİZLİĞİ İÇİN AĞIRLIK HESAPLAMA ---
# SMOTE yerine Class Weights kullanıyoruz.
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Sınıf Ağırlıkları Hesaplandı: {class_weights}")

# Özel Trainer Sınıfı (Ağırlıklı Loss Fonksiyonu İçin)
from torch import nn
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 5. MODEL KURULUMU ---
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# --- 6. EĞİTİM AYARLARI ---
training_args = TrainingArguments(
    output_dir='./results',          # Sonuçların kaydedileceği klasör
    num_train_epochs=3,              # Eğitim tur sayısı (Epoch)
    per_device_train_batch_size=8,   # Batch size (GPU belleğine göre düşürün: 8 veya 4)
    per_device_eval_batch_size=16,
    warmup_steps=500,                # Isınma adımları
    weight_decay=0.01,               # Overfitting önleme
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",     # Her epoch sonunda test et
    save_strategy="epoch",
    load_best_model_at_end=True,     # En iyi modeli yükle
)

# Metrik Hesaplama Fonksiyonu
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# --- 7. EĞİTİMİ BAŞLAT ---
print("\n--- BERT EĞİTİMİ BAŞLIYOR ---")
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- 8. SONUÇLAR VE RAPORLAMA ---
print("\n--- TEST SONUÇLARI ---")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# Rapor
print(classification_report(y_test, preds, target_names=class_names))

# Confusion Matrix Görselleştirme
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - BERT Model')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.savefig('BERT_Confusion_Matrix.png')
plt.show()

# Modeli Kaydetme (İsteğe bağlı)
model.save_pretrained("./final_bert_model")
tokenizer.save_pretrained("./final_bert_model")
print("Model './final_bert_model' klasörüne kaydedildi.")