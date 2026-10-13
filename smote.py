import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# --- Sklearn Modülleri ---
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- SMOTE İÇİN KRİTİK KÜTÜPHANELER ---
# Standart sklearn Pipeline'ı SMOTE ile çalışmaz, imblearn Pipeline'ı şarttır.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("HATA: 'imblearn' kütüphanesi eksik. Lütfen 'pip install imbalanced-learn' komutunu çalıştırın.")
    exit()

# --- NLTK Verilerini İndirme ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
stop_words = set(stopwords.words('english'))

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

# Train/Test Ayrımı (SMOTE öncesi)
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

# --- 4. MODELLER VE SMOTE PİPELINE KURULUMU ---

# Ortak Parametreler
tfidf_params = {'max_features': 1500}
n_features = 500 

# K Parametresi Ayarı (İsteğiniz üzerine 500)
# NOT: Eğer veri setiniz 500'den küçükse kodun çökmemesi için güvenlik kontrolü ekledim.
target_k = 30
safe_k = min(target_k, int(len(X_text_train) * 0.8)) # Veri boyutuna göre güvenli k
if safe_k < 1: safe_k = 1

print(f"KNN için Hedef k={target_k}. Kullanılacak (Güvenli) k={safe_k}")

# SMOTE k_neighbors ayarı (Az veride hata vermemesi için)
# Çok küçük sınıflarda (örn: 2 örnek) SMOTE varsayılan k=5 ile çalışamaz.
min_class_samples = pd.Series(y_train).value_counts().min()
smote_k = min(5, min_class_samples - 1)
if smote_k < 1: smote_k = 1

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'KNN (Cosine)': KNeighborsClassifier(n_neighbors=safe_k, metric='cosine') 
}

results = {}

print("\n--- SMOTE ENTEGRELİ K-FOLD CV VE EĞİTİM ---")

for model_name, classifier in models.items():
    print(f"\nModel: {model_name} işleniyor...")
    
    # --- SMOTE PİPELINE YAPISI ---
    # 1. Tfidf: Metni sayıya çevir
    # 2. SMOTE: Sayısal veriyi artır (Sadece Train setinde çalışır)
    # 3. SelectKBest: En iyi özellikleri seç
    # 4. Classifier: Sınıflandır
    
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('smote', SMOTE(random_state=42, k_neighbors=smote_k)), # ENTEGRASYON BURADA
        ('selector', SelectKBest(score_func=chi2, k='all')), 
        ('clf', classifier)
    ])
    
    # 1. K-Fold Cross Validation
    # İpucu: Veri setiniz çok küçükse n_splits=3 bile hata verebilir, 2'ye düşürdüm.
    # Gerçek büyük veride bunu 5 veya 10 yapabilirsiniz.
    splits = min(10, min_class_samples) if min_class_samples > 1 else 2
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    
    try:
        cv_scores = cross_val_score(pipeline, X_text_train, y_train, cv=cv, scoring='accuracy')
        print(f"   -> {splits}-Fold CV Doğruluk Ortalaması: %{cv_scores.mean()*100:.2f}")
    except ValueError as e:
        print(f"   -> CV Hatası (Veri çok az olabilir): {e}")
    
    # 2. Final Modeli Eğitme (SMOTE burada tüm X_train'e uygulanır)
    pipeline.fit(X_text_train, y_train)
    
    # 3. Test Verisiyle Tahmin
    y_pred = pipeline.predict(X_text_test)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = {'pipeline': pipeline, 'y_pred': y_pred, 'accuracy': acc}
    
    print(f"   -> Test Seti Doğruluk (Accuracy): %{acc*100:.2f}")

# --- 5. DETAYLI RAPOR VE GÖRSELLEŞTİRME ---

for model_name, data in results.items():
    print("\n" + "="*40)
    print(f"DETAYLI RAPOR: {model_name}")
    print("="*40)
    
    # Sınıflandırma Raporu (Hata bastırma: UndefinedMetricWarning için zero_division=0)
    print(classification_report(y_test, data['y_pred'], target_names=encoder.classes_, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, data['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, 
                yticklabels=encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}\n(Accuracy: %{data["accuracy"]*100:.2f})')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.show()