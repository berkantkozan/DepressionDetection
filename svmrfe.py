print("Model eğitimi başlıyor (SVM-RFE + RF)...")

# 4.1. Öznitelik Seçimi: SVM-RFE
# RFE'nin temel alacağı tahminci (estimator) olarak lineer bir SVM kullanıyoruz
svc_estimator = SVC(kernel="linear")

# RFE'yi başlatıyoruz.
# n_features_to_select: 2000 özellikten en iyi kaç tanesini seçeceğiz?
# Bu, ayarlanması gereken önemli bir hiperparametredir. 300 ile başlayalım.
# rfe = RFE(estimator=svc_estimator, n_features_to_select=300, step=1)
rfe = RFE(estimator=svc_estimator, n_features_to_select=300, step=100)

print(f"SVM-RFE ile {X_train.shape[1]} öznitelikten en iyi 300 tanesi seçiliyor...")
# RFE'yi EĞİTİM VERİSİNE uygula
X_train_rfe = rfe.fit_transform(X_train, y_train)

# TEST VERİSİNİ de aynı seçilen özelliklere göre dönüştür
X_test_rfe = rfe.transform(X_test)

print(f"Öznitelik seçimi sonrası yeni boyut: {X_train_rfe.shape}")
print("-" * 30)


# 4.2. Sınıflandırma: Random Forest (RF)
# RF modelini, RFE tarafından seçilen öznitelikler üzerinde eğiteceğiz
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

print("Random Forest modeli seçilen öznitelikler üzerinde eğitiliyor...")
rf_classifier.fit(X_train_rfe, y_train)


# --- 5. DEĞERLENDİRME ---

print("Model test verisi üzerinde değerlendiriliyor...")
y_pred = rf_classifier.predict(X_test_rfe)
