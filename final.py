import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle

# --- KÜTÜPHANE KONTROLLERİ ---
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("HATA: 'imblearn' kütüphanesi eksik. 'pip install imbalanced-learn' çalıştırın.")
    exit()

# BERT/RoBERTa için gerekli kütüphaneler
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    print("Transformers ve Torch kütüphaneleri algılandı. Derin öğrenme modelleri aktif.")
    DL_AVAILABLE = True
except ImportError:
    print("UYARI: 'transformers' veya 'torch' eksik. BERT/RoBERTa devre dışı kalacak.")
    print("Kurulum için: pip install transformers torch")
    DL_AVAILABLE = False

# --- NLTK Verilerini İndirme ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- 1. Veri Setini Yükleme ---
try:
    df = pd.read_csv('dataset.csv')
    print("Veri seti 'dataset.csv' dosyasından yüklendi.")
except FileNotFoundError:
    print("UYARI: 'dataset.csv' bulunamadı, simülasyon verisi kullanılıyor.")
    from io import StringIO
    csv_data = """uniqe_id,statement,status
0,oh my gosh,Anxiety
1,"trouble sleeping, confused mind, restless heart.",Anxiety
2,"All wrong, back off dear, forward doubt.",Anxiety
3,I've shifted my focus to something else but I'm still worried,Anxiety
4,"Feeling great today, sunshine and rainbows",Normal
5,"I am so happy and relieved right now.",Normal
6,"Just a regular day, nothing special.",Normal
7,"This is fine.",Normal
8,"I want to end it all, no hope left.",Suicidal
9,"Goodbye world, I cannot take this anymore.",Suicidal
10,"I feel so stressed and overwhelmed with work.",Stress
11,"Too much pressure, I can't breathe.",Stress"""
    df = pd.read_csv(StringIO(csv_data))

# --- 2. Metin Ön İşleme ---
print("Metin ön işleme başlıyor...")
nltk_stops = set(stopwords.words('english'))

# Olumsuzluk eklerini listeden ÇIKAR (Bunları silme, model görsün)
exclude_words = {'not', 'no', 'nor', 'never', "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"}

stop_words = nltk_stops - exclude_words 
# Artık model "not happy" ifadesindeki "not"ı görecek.

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("Metin ön işleme yapılıyor...")
df['cleaned_statement'] = df['statement'].apply(preprocess_text)

# --- 3. Veri Hazırlama ---
encoder = LabelEncoder()
y = encoder.fit_transform(df['status'])
classes = encoder.classes_
n_classes = len(classes)

# Train/Test Ayrımı
X_text_train, X_text_test, y_train, y_test = train_test_split(
    df['cleaned_statement'], 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

print(f"\nEğitim Verisi Sayısı: {len(X_text_train)}")
print(f"Test Verisi Sayısı: {len(X_text_test)}")
print("-" * 30)

# --- DERİN ÖĞRENME ENTEGRASYONU (BERT/RoBERTa Class) ---
class TransformerEmbedder(BaseEstimator, TransformerMixin):
    """
    HuggingFace modellerini (BERT, RoBERTa) sklearn pipeline ile uyumlu hale getirir.
    Metinleri vektöre çevirir (Feature Extraction).
    """
    def __init__(self, model_name='bert-base-uncased', batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y=None):
        # Model ve tokenizer'ı yükle
        if DL_AVAILABLE:
            print(f"[{self.model_name}] Yükleniyor... (Cihaz: {self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        return self

    def transform(self, X):
        if not DL_AVAILABLE:
            return np.zeros((len(X), 1)) # Dummy output
            
        embeddings = []
        # Batch işlemi (Bellek taşmasını önlemek için)
        X_list = list(X)
        total = len(X_list)
        
        self.model.eval() # Değerlendirme modu
        
        with torch.no_grad():
            for i in range(0, total, self.batch_size):
                batch_texts = X_list[i : i + self.batch_size]
                
                # Tokenization
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=64, # Hız için limitli, gerekirse 128/256 yapın
                    return_tensors="pt"
                ).to(self.device)
                
                # Modelden geçirme
                outputs = self.model(**inputs)
                
                # [CLS] tokeni veya Mean Pooling (Burada CLS kullanıyoruz)
                # last_hidden_state: (batch, seq_len, hidden_dim) -> CLS: (batch, hidden_dim)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
                
        return np.vstack(embeddings)

# --- 4. MODELLER VE PARAMETRELER ---
tfidf_params = {
    'max_features': 3000,   # Kelime havuzunu 3000'e çıkardık
    'ngram_range': (1, 2)   # Hem tek kelimeleri hem ikilileri al ("not happy" gibi)
}
n_features_select = 500  

# Komşu sayısı ayarları
min_class_samples = pd.Series(y_train).value_counts().min()
smote_k = min(5, min_class_samples - 1)
if smote_k < 1: smote_k = 1
safe_k = 5
if safe_k < 1: safe_k = 1

# --- KONFIGÜRASYON LİSTESİ ---
# Her konfigürasyon: (Ad, Vectorizer, Seçici, Classifier)
model_configs = []

# 1. Klasik TF-IDF Modelleri
model_configs.append((
    'TF-IDF + Random Forest',
    TfidfVectorizer(**tfidf_params),
    SelectKBest(score_func=chi2, k=n_features_select),
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None,n_jobs=-1)
))

model_configs.append((
    'TF-IDF + KNN',
    TfidfVectorizer(**tfidf_params),
    SelectKBest(score_func=chi2, k=n_features_select),
    KNeighborsClassifier(n_neighbors=safe_k, metric='cosine',n_jobs=-1)
))

# 2. Derin Öğrenme Modelleri (Varsa ekle)
if DL_AVAILABLE:
    # 1. BERT + RF
    model_configs.append((
        'BERT + Random Forest',
        TransformerEmbedder(model_name='bert-base-uncased'),
        None, # BERT vektörleri zaten yoğundur, özellik seçimine gerek yok
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None,n_jobs=-1)
    ))
    
    # 2. RoBERTa + RF
    model_configs.append((
        'RoBERTa + Random Forest',
        TransformerEmbedder(model_name='roberta-base'),
        None,
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None,n_jobs=-1)
    ))

    # 3. BERT + KNN (Cosine Metriği ile)
    model_configs.append((
        'BERT + KNN',
        TransformerEmbedder(model_name='bert-base-uncased'),
        None, # Özellik seçimine gerek yok
        KNeighborsClassifier(n_neighbors=safe_k, metric='cosine',n_jobs=-1) # Cosine çok kritiktir!
    ))
    
    # 4. RoBERTa + KNN (Cosine Metriği ile)
    model_configs.append((
        'RoBERTa + KNN',
        TransformerEmbedder(model_name='roberta-base'),
        None,
        KNeighborsClassifier(n_neighbors=safe_k, metric='cosine',n_jobs=-1)
    ))

results = {}

print(f"\n--- EĞİTİM SÜRECİ BAŞLIYOR (Toplam {len(model_configs)} Model) ---")

for config in model_configs:
    name, vectorizer, selector, classifier = config
    print(f"\n>> Model Eğitiliyor: {name}")
    
    steps = [('vectorizer', vectorizer)]
    
    # SMOTE her zaman var
    steps.append(('smote', SMOTE(random_state=42, k_neighbors=smote_k)))
    
    # Seçici varsa ekle (TF-IDF için)
    if selector:
        steps.append(('selector', selector))
        
    steps.append(('clf', classifier))
    
    pipeline = ImbPipeline(steps)
    
    # CV Skor
    try:
        splits = min(3, min_class_samples) if min_class_samples > 1 else 2
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        # Not: BERT CV çok uzun sürebilir, demo için pas geçebilirsiniz
        if "BERT" not in name and "RoBERTa" not in name: 
            cv_scores = cross_val_score(pipeline, X_text_train, y_train, cv=cv, scoring='accuracy',n_jobs=1)
            print(f"   -> CV Doğruluk Ortalaması: %{cv_scores.mean()*100:.2f}")
        else:
            print("   -> Derin öğrenme modelleri için CV atlanıyor (hız optimizasyonu).")
    except Exception as e:
        print(f"   -> CV Uyarısı: {e}")
    
    # Final Eğitim
    pipeline.fit(X_text_train, y_train)
    
    # Tahmin
    y_pred = pipeline.predict(X_text_test)
    
    try:
        y_proba = pipeline.predict_proba(X_text_test)
    except:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    results[name] = {'pipeline': pipeline, 'y_pred': y_pred, 'y_proba': y_proba, 'accuracy': acc}
    print(f"   -> Test Seti Doğruluk: %{acc*100:.2f}")

# --- 5. DETAYLI RAPORLAMA ---
for model_name, data in results.items():
    print("\n" + "="*40)
    print(f"RAPOR: {model_name}")
    print("="*40)
    
    print(classification_report(y_test, data['y_pred'], target_names=classes, zero_division=0))
    
    # Confusion Matrix (Sadece Kaydet, Gösterme - kalabalık olmasın)
    cm = confusion_matrix(y_test, data['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'CM - {model_name}')
    plt.tight_layout()
    clean_name = model_name.replace(" ", "_").replace("+", "plus")
    plt.savefig(f'CM_{clean_name}.png')
    plt.close()

    # ROC Eğrileri
    if data['y_proba'] is not None and len(np.unique(y_test)) > 1:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(8, 6))
        colors = cycle(['blue', 'red', 'green', 'orange'])
        
        for i, color in zip(range(n_classes), colors):
            if np.sum(y_test_bin[:, i]) == 0: continue
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], data['y_proba'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC={roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC_{clean_name}.png')
        plt.close()

