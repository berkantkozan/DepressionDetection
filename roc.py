import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle

# --- SMOTE İÇİN KRİTİK KÜTÜPHANELER ---
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("HATA: 'imblearn' kütüphanesi eksik. Lütfen 'pip install imbalanced-learn' komutunu çalıştırın.")
    exit()

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

# --- 4. MODELLER VE PARAMETRELER ---
tfidf_params = {'max_features': 1500} # Kelime havuzunu genişlettik
n_features_select = 500  # En iyi 500 özelliği seçeceğiz

# Güvenli KNN komşu sayısı
target_k = 30
safe_k = min(target_k, int(len(X_text_train) * 0.8))
if safe_k < 1: safe_k = 1

# SMOTE komşu sayısı
min_class_samples = pd.Series(y_train).value_counts().min()
smote_k = min(5, min_class_samples - 1)
if smote_k < 1: smote_k = 1

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'KNN (Cosine)': KNeighborsClassifier(n_neighbors=safe_k, metric='cosine') 
}

results = {}

print("\n--- SMOTE ENTEGRELİ EĞİTİM SÜRECİ ---")

for model_name, classifier in models.items():
    print(f"\nModel: {model_name} eğitiliyor...")
    
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('smote', SMOTE(random_state=42, k_neighbors=smote_k)),
        ('selector', SelectKBest(score_func=chi2, k='all')), # Hata almamak için 'all', aşağıda özellik öneminde filtereleyeceğiz
        ('clf', classifier)
    ])
    
    # K-Fold CV
    splits = min(10, min_class_samples) if min_class_samples > 1 else 2
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    
    try:
        cv_scores = cross_val_score(pipeline, X_text_train, y_train, cv=cv, scoring='accuracy')
        print(f"   -> {splits}-Fold CV Doğruluk Ortalaması: %{cv_scores.mean()*100:.2f}")
    except ValueError as e:
        print(f"   -> CV Hatası: {e}")
    
    # Final Eğitim
    pipeline.fit(X_text_train, y_train)
    
    # Tahminler
    y_pred = pipeline.predict(X_text_test)
    
    # ROC-AUC için olasılık değerleri (Probability)
    try:
        y_proba = pipeline.predict_proba(X_text_test)
    except AttributeError:
        y_proba = None # Bazı modellerde predict_proba olmayabilir

    acc = accuracy_score(y_test, y_pred)
    results[model_name] = {'pipeline': pipeline, 'y_pred': y_pred, 'y_proba': y_proba, 'accuracy': acc}
    
    print(f"   -> Test Seti Doğruluk: %{acc*100:.2f}")

# --- 5. DETAYLI RAPORLAMA VE GÖRSELLEŞTİRME ---

for model_name, data in results.items():
    print("\n" + "="*40)
    print(f"DETAYLI RAPOR: {model_name}")
    print("="*40)
    
    # 1. Classification Report
    print(classification_report(y_test, data['y_pred'], target_names=classes, zero_division=0))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, data['y_pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}\n(Accuracy: %{data["accuracy"]*100:.2f})')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig(f'{model_name}_confusion_matrix.png') # Resmi kaydet
    plt.show()

    # 3. ÖZNİTELİK ÖNEMİ (Sadece Random Forest için)
    if model_name == 'Random Forest':
        try:
            # Pipeline adımlarına erişim
            rf_model = data['pipeline'].named_steps['clf']
            tfidf = data['pipeline'].named_steps['tfidf']
            selector = data['pipeline'].named_steps['selector']
            
            # Özellik isimlerini al ve SelectKBest ile seçilenleri filtrele
            feature_names = np.array(tfidf.get_feature_names_out())
            # SelectKBest k='all' olduğu için maske tümü True olabilir, yine de logic bu şekildedir
            mask = selector.get_support() 
            selected_features = feature_names[mask]
            
            # Önem skorlarını al
            importances = rf_model.feature_importances_
            
            # Sıralama
            indices = np.argsort(importances)[::-1]
            top_n = 20 # En önemli 20 kelime
            
            plt.figure(figsize=(12, 6))
            plt.title("Depresyon Tespitinde En Önemli 20 Kelime (Feature Importance)")
            plt.bar(range(top_n), importances[indices[:top_n]], align="center")
            plt.xticks(range(top_n), selected_features[indices[:top_n]], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('RF_Feature_Importance.png')
            plt.show()
            print("   -> Feature Importance grafiği oluşturuldu.")
        except Exception as e:
            print(f"   -> Feature Importance hatası: {e}")

    # 4. ROC-AUC Eğrileri (Çok Sınıflı)
    if data['y_proba'] is not None and len(np.unique(y_test)) > 1:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # ROC hesapla
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
        
        for i, color in zip(range(n_classes), colors):
            # Sınıf test setinde yoksa atla
            if np.sum(y_test_bin[:, i]) == 0:
                continue
                
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], data['y_proba'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC ({0}) (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
        plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'{model_name}_ROC_Curve.png')
        plt.show()