import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Sklearn Modülleri ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- NLTK Verilerini İndirme (İlk kez çalıştırırken gereklidir) ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# --- 1. Veri Setini Yükleme ---
# CSV dosyanızın adını 'dataset.csv' olarak varsayıyoruz.
# Eğer dosya adı farklıysa, 'dataset.csv' kısmını güncelleyin.
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("HATA: 'dataset.csv' dosyası bulunamadı.")
    # Örnek verinizle devam etmek için simülasyon:
    from io import StringIO
    csv_data = """uniqe_id,statement,status
0,oh my gosh,Anxiety
1,"trouble sleeping, confused mind, restless heart. All out of tune",Anxiety
2,"All wrong, back off dear, forward doubt. Stay in a restless and restless place",Anxiety
3,I've shifted my focus to something else but I'm still worried,Anxiety
4,"Feeling great today, sunshine and rainbows",Normal
5,"I am so happy and relieved right now.",Normal
6,"Just a regular day, nothing special.",Normal
7,"This is fine.",Normal"""
    df = pd.read_csv(StringIO(csv_data))

print("Veri Seti İlk 5 Satır:")
print(df.head())
print("-" * 30)

# --- 2. Metin Ön İşleme (NLTK) ---
# Veri setiniz İngilizce görünüyor, bu yüzden İngilizce stopwords kullanacağız.
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Metni temizler: küçük harf, noktalama/sayı kaldırma, tokenizasyon ve stopwords"""
    text = str(text).lower()  # Küçük harfe çevir
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalamayı kaldır
    text = text.strip()  # Boşlukları kaldır
    
    tokens = word_tokenize(text)  # Tokenize et
    
    # Stopwords'leri kaldır
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

print("Metin ön işleme başlıyor...")
# 'statement' sütunundaki her bir metne ön işleme uygula
df['cleaned_statement'] = df['statement'].apply(preprocess_text)

print("Ön İşlenmiş Metin Örneği:")
print(df[['statement', 'cleaned_statement']].head())
print("-" * 30)


# --- 3. Veri Hazırlama (Öznitelik Çıkarımı ve Etiketleme) ---

# Öznitelikler (X): TF-IDF Vektörleri
# Metin verisini sayısal vektörlere dönüştür
vectorizer = TfidfVectorizer(max_features=2000) # En önemli 2000 kelimeyi al
X = vectorizer.fit_transform(df['cleaned_statement']).toarray()

# Hedef (y): Etiketler (Status)
# 'Anxiety', 'Normal' gibi metin etiketlerini 0, 1 gibi sayılara dönüştür
encoder = LabelEncoder()
y = encoder.fit_transform(df['status'])

print(f"Öznitelik matrisi (X) boyutu: {X.shape}")
print(f"Hedef vektör (y) boyutu: {y.shape}")
print(f"Etiketler ve kodları: {list(encoder.classes_)} -> {np.unique(y)}")
print("-" * 30)

# Veriyi Eğitim ve Test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# --- 4. MODEL PİPELINE (SVM-RFE + RF) ---
# -------------------- BU YENİ KODU EKLEYİN --------------------

# Gerekli kütüphaneleri import edin (kodun en başına da ekleyebilirsiniz)
from sklearn.feature_selection import SelectKBest, chi2

print("Model eğitimi başlıyor (SelectKBest + RF)...")

# 4.1. Öznitelik Seçimi: SelectKBest (chi2 testi ile)
# k=300, en iyi 300 özniteliği seçeceğimizi belirtir (RFE'deki n_features_to_select gibi)
print(f"SelectKBest (chi2) ile {X_train.shape[1]} öznitelikten en iyi 300 tanesi seçiliyor...")

kbest = SelectKBest(score_func=chi2, k=300)

# kbest'i EĞİTİM VERİSİNE uygula
X_train_kbest = kbest.fit_transform(X_train, y_train)

# TEST VERİSİNİ de aynı seçilen özelliklere göre dönüştür
X_test_kbest = kbest.transform(X_test)

print(f"Öznitelik seçimi sonrası yeni boyut: {X_train_kbest.shape}")
print("-" * 30)


# 4.2. Sınıflandırma: Random Forest (RF)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

print("Random Forest modeli seçilen öznitelikler üzerinde eğitiliyor...")
# !!! Değişken adlarına dikkat: X_train_kbest kullanıyoruz
rf_classifier.fit(X_train_kbest, y_train)


# --- 5. DEĞERLENDİRME ---
print("Model test verisi üzerinde değerlendiriliyor...")
# !!! Değişken adlarına dikkat: X_test_kbest kullanıyoruz
y_pred = rf_classifier.predict(X_test_kbest)

# -------------------- YENİ KOD BURADA BİTİYOR --------------------

# Metrikleri (Kesinlik, Duyarlılık, F1-Skoru) yazdır
report = classification_report(y_test, y_pred, target_names=encoder.classes_)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print("     MODEL DEĞERLENDİRME RAPORU")
print("="*30)
print(f"Genel Doğruluk (Accuracy): {accuracy:.4f}")
print("\nSınıflandırma Raporu:")
print(report)