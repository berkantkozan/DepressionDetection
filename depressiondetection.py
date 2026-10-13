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
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- NLTK Verilerini İndirme ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# --- 1. Veri Setini Yükleme ---
try:
    df = pd.read_csv('dataset.csv')
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
# Etiketleri Sayısallaştır
encoder = LabelEncoder()
y = encoder.fit_transform(df['status'])

# ÖNEMLİ DEĞİŞİKLİK: 
# Train/Test ayrımını TF-IDF'ten ÖNCE yapıyoruz.
# Bu, K-Fold sırasında veri sızıntısını önlemek için kritik.
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

# --- 4. MODELLER VE PİPELINE KURULUMU ---

# Ortak Parametreler
tfidf_params = {'max_features': 2000}
kbest_params = {'score_func': chi2, 'k': 'all'} # Veri azsa 'all', çoksa 300 yapın. Örn: k=300

# Eğer verinizde yeterli özellik (kelime) yoksa k=300 hata verebilir. 
# Güvenlik için şimdilik 'all' veya veri setinize göre dinamik ayarlama:
n_features = 1000 

# Modelleri Tanımla
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'KNN (Cosine)': KNeighborsClassifier(n_neighbors=5, metric='cosine') # Metin için Cosine
}

# Sonuçları saklamak için
results = {}

print("\n--- K-FOLD CROSS VALIDATION VE EĞİTİM SÜRECİ ---")

for model_name, classifier in models.items():
    print(f"\nModel: {model_name} işleniyor...")
    
    # PIPELINE OLUŞTURMA: Vektörleştir -> Seç -> Sınıflandır
    # Pipeline sayesinde K-Fold her seferinde ham veriyi alıp işler (Sızıntı yok)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('selector', SelectKBest(score_func=chi2, k=min(n_features, len(X_text_train)))), # Hata almamak için min
        ('clf', classifier)
    ])
    
    # 1. K-Fold Cross Validation (Sadece Eğitim Verisi Üzerinde)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Veri azsa split'i düşürün (örn: 3)
    cv_scores = cross_val_score(pipeline, X_text_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"   -> {cv.get_n_splits()}-Fold CV Doğruluk Ortalaması: %{cv_scores.mean()*100:.2f}")
    
    # 2. Final Modeli Eğitme (Tüm Eğitim Verisiyle)
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
    
    # Sınıflandırma Raporu
    print(classification_report(y_test, data['y_pred'], target_names=encoder.classes_))
    
    # Confusion Matrix Görselleştirme
    cm = confusion_matrix(y_test, data['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, 
                yticklabels=encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}\n(Accuracy: %{data["accuracy"]*100:.2f})')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.show()