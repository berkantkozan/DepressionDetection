import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from itertools import cycle

# --- AYARLAR ---
SAVED_MODELS = {
    "BERT": "./finetuned_bert_final",
    "RoBERTa": "./finetuned_roberta_final"
}
BATCH_SIZE = 16
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. VERİ HAZIRLIĞI ---
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("❌ dataset.csv bulunamadı!")
    exit()

le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])
class_names = le.classes_
n_classes = len(class_names)

# Train/Test ayrımı (Aynı random_state ile)
_, temp_texts, _, temp_labels = train_test_split(
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

# Test etiketlerini binarize et (One-vs-Rest formatı için)
y_test_bin = label_binarize(test_labels, classes=range(n_classes))
test_dataset_raw = Dataset.from_dict({"text": test_texts, "label": test_labels})

# --- 2. MODEL DÖNGÜSÜ VE ÇİZİM ---
plt.figure(figsize=(12, 10))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])

# Sadece RoBERTa'nın "Macro Average" eğrisini ve Sınıf bazlı eğrilerini çizelim
# (Karmaşıklık olmasın diye en iyi modeli detaylı çiziyoruz)
target_model_key = "BERT" 

print(f"📊 {target_model_key} için ROC Eğrileri hazırlanıyor...")

model_path = SAVED_MODELS[target_model_key]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized_test = test_dataset_raw.map(tokenize_function, batched=True)
trainer = Trainer(model=model)

# Tahminleri al (Logits)
preds_output = trainer.predict(tokenized_test)
logits = preds_output.predictions

# Logits -> Probabilities (Softmax)
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

# --- 3. ROC HESAPLAMA ---
fpr = dict()
tpr = dict()
roc_auc = dict()

# Her sınıf için ayrı ROC hesapla
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-Average ROC (Genel performans)
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# --- 4. ÇİZİM ---
lw = 2

# Micro-Average Çizgisi
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.2f})',
         color='deeppink', linestyle=':', linewidth=4)

# Her sınıfın çizgisi
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'ROC: {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=lw) # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm Oranı)')
plt.ylabel('True Positive Rate (Duyarlılık)')
plt.title(f'{target_model_key} Modeli İçin Çok Sınıflı ROC Analizi')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

save_name = f"roc_curve_{target_model_key}.png"
plt.savefig(save_name)
print(f"✅ Grafik kaydedildi: {save_name}")
plt.show()