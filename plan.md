# 🌍 Türkiye Deprem Risk Analizi — 4. Hafta LinkedIn Projesi

> **Mentor Modu:** Bu projeyi Cem ile birlikte, öğreterek yapacağız. Vibe-coding yok.

**Hedef:** Türkiye'deki 30 yıllık deprem verilerini analiz edip, illere göre risk haritası çıkarmak, depremleri kümelemek ve büyüklük tahmini yapmak.

**Neden Bu Proje?**
- Derste öğrendiğin TÜM teknikleri tek projede kullanırsın
- Türkiye konusu → LinkedIn'de çok ilgi çeker
- Gerçek dünya verisi (Kaggle + Kandilli)
- Hocanın kodlarındaki eksiklikleri (train/test split, cross-validation) düzelterek "ders ötesi" bir iş çıkarırsın

**Teknoloji:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 📋 AŞAMA 1: Veri Toplama ve Temizleme (EDA)
> Dersten kullanılan: Pandas, NumPy, istatistik, görselleştirme

### Görev 1.1 — Proje yapısı oluştur
- `deprem-risk-analizi/` klasörü
- `data/` → veri dosyaları
- `notebooks/` → Jupyter notebook'lar
- `README.md`

### Görev 1.2 — Veri setini indir
- **Kaynak:** Kaggle "Turkey Earthquake Data (1994-2023)" by Özge Çinko
  - URL: https://www.kaggle.com/datasets/ozgecinko/turkey-earthquake-data-1994-2023
  - ~15.000+ kayıt, 15 sütun, Kandilli verisi
- Manuel indirip `data/` klasörüne koy

### Görev 1.3 — Veriyi yükle ve ilk keşif
```python
import pandas as pd
df = pd.read_csv("data/veriler.csv")
print(df.shape)          # Kaç satır, kaç sütun?
print(df.columns)        # Sütun isimleri
print(df.head())         # İlk 5 satır
print(df.info())         # Veri tipleri, null sayıları
print(df.describe())     # İstatistikler
```

### Görev 1.4 — Eksik veri analizi
```python
print(df.isnull().sum())           # Her sütundaki null sayısı
print(df.isnull().sum() / len(df)) # Null yüzdeleri
```
- Kararlar: Hangi sütunlar drop edilecek? Hangileri fill edilecek?

### Görev 1.5 — Veri temizleme
- Gereksiz sütunları kaldır
- Tarih sütununu datetime'a çevir
- Büyüklük ve derinlik sütunlarını sayısal yap
- Negatif/mantıksız değerleri filtrele (derinlik < 0, büyüklük < 0)

### Görev 1.6 — Feature Engineering
```python
df['yil'] = df['tarih'].dt.year
df['ay'] = df['tarih'].dt.month
df['saat'] = df['tarih'].dt.hour
```
- Konum bilgisinden il çıkarma (varsa yer sütunundan parse etme)

---

## 📊 AŞAMA 2: Keşifsel Veri Analizi (EDA) ve Görselleştirme
> Dersten kullanılan: matplotlib, seaborn, istatistik, korelasyon

### Görev 2.1 — Deprem büyüklüğü dağılımı
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['buyukluk'], bins=50, kde=True)
plt.title('Deprem Büyüklüğü Dağılımı')
plt.xlabel('Büyüklük (ML)')
plt.ylabel('Frekans')
plt.show()
```

### Görev 2.2 — Yıllara göre deprem sayısı
```python
df.groupby('yil')['buyukluk'].count().plot(kind='bar', figsize=(14, 6))
plt.title('Yıllara Göre Deprem Sayısı')
plt.show()
```

### Görev 2.3 — Derinlik vs Büyüklük ilişkisi
```python
plt.scatter(df['derinlik'], df['buyukluk'], alpha=0.3)
plt.xlabel('Derinlik (km)')
plt.ylabel('Büyüklük')
plt.title('Derinlik - Büyüklük İlişkisi')
plt.show()

# Korelasyon (dersten: Pearson)
from scipy.stats import pearsonr
corr, p = pearsonr(df['derinlik'], df['buyukluk'])
print(f"Pearson korelasyon: {corr:.3f}, p-değeri: {p:.4f}")
```

### Görev 2.4 — Korelasyon matrisi (Heatmap)
```python
sns.heatmap(df[sayisal_sutunlar].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasyon Matrisi')
plt.show()
```

### Görev 2.5 — Coğrafi dağılım (Scatter plot harita)
```python
plt.figure(figsize=(12, 8))
plt.scatter(df['boylam'], df['enlem'], c=df['buyukluk'], cmap='Reds', alpha=0.5, s=df['buyukluk']**2)
plt.colorbar(label='Büyüklük')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.title('Türkiye Deprem Haritası')
plt.show()
```

### Görev 2.6 — İllere göre deprem risk sıralaması
- En çok deprem olan 20 il (bar chart)
- En yüksek ortalama büyüklük olan iller

---

## 🤖 AŞAMA 3: Gözetimsiz Öğrenme — Kümeleme
> Dersten kullanılan: K-Means, Elbow Method, Silhouette Score

### Görev 3.1 — Veriyi kümeleme için hazırla
```python
from sklearn.preprocessing import StandardScaler

# Özellikler: enlem, boylam, derinlik, büyüklük
X = df[['enlem', 'boylam', 'derinlik', 'buyukluk']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Görev 3.2 — Elbow Method ile optimal küme sayısı
```python
from sklearn.cluster import KMeans

wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(2, 11), wcss, 'bo-')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

### Görev 3.3 — K-Means kümeleme uygula
```python
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=4, random_state=42, n_init=10)  # Elbow'dan seç
df['kume'] = km.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df['kume'])
print(f"Silhouette Score: {sil:.3f}")
```

### Görev 3.4 — Kümeleri harita üzerinde görselleştir
```python
plt.figure(figsize=(12, 8))
plt.scatter(df['boylam'], df['enlem'], c=df['kume'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Küme')
plt.title('Deprem Kümeleri (K-Means)')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()
```

### Görev 3.5 — Küme yorumlama
- Her kümenin ortalama büyüklük, derinlik, konum bilgisi
- "Küme 0 = Batı Anadolu sığ depremler", "Küme 1 = Doğu Anadolu derin depremler" gibi

---

## 🎯 AŞAMA 4: Gözetimli Öğrenme — Sınıflandırma
> Dersten kullanılan: train/test split, LR, RF, SVM, confusion matrix, F1, ROC/AUC

### Görev 4.1 — Hedef değişken oluştur (Risk Seviyesi)
```python
def risk_seviyesi(buyukluk):
    if buyukluk < 3.0:
        return 'Düşük'
    elif buyukluk < 5.0:
        return 'Orta'
    elif buyukluk < 6.0:
        return 'Yüksek'
    else:
        return 'Çok Yüksek'

df['risk'] = df['buyukluk'].apply(risk_seviyesi)
```

### Görev 4.2 — Train/Test split (HOCANIN YAPMADIKLARI!)
```python
from sklearn.model_selection import train_test_split

X = df[['enlem', 'boylam', 'derinlik', 'yil', 'ay']]
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Eğitim: {X_train.shape[0]}, Test: {X_test.shape[0]}")
```

### Görev 4.3 — StandardScaler uygula
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ÖNEMLİ: test'e sadece transform!
```

### Görev 4.4 — Model 1: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print(classification_report(y_test, y_pred_lr))
```

### Görev 4.5 — Model 2: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print(classification_report(y_test, y_pred_rf))
```

### Görev 4.6 — Model 3: SVM
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print(classification_report(y_test, y_pred_svm))
```

### Görev 4.7 — Model karşılaştırma tablosu
```python
from sklearn.metrics import accuracy_score, f1_score

sonuclar = {
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm)
    ],
    'F1 (weighted)': [
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_svm, average='weighted')
    ]
}
pd.DataFrame(sonuclar)
```

### Görev 4.8 — Confusion Matrix görselleştirme (en iyi model)
```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Blues')
plt.title('Random Forest — Confusion Matrix')
plt.show()
```

### Görev 4.9 — Feature Importance (Random Forest)
```python
importances = rf.feature_importances_
feat_names = ['Enlem', 'Boylam', 'Derinlik', 'Yıl', 'Ay']

plt.barh(feat_names, importances)
plt.title('Özellik Önem Sıralaması')
plt.show()
```

---

## 📝 AŞAMA 5: Sonuçlandırma ve LinkedIn Paylaşımı

### Görev 5.1 — README.md yaz
- Proje açıklaması
- Kullanılan teknikler
- Sonuçlar ve bulgular
- Görseller (grafikleri kaydet)

### Görev 5.2 — GitHub'a yükle
```bash
git init
git add .
git commit -m "feat: Türkiye Deprem Risk Analizi projesi"
git remote add origin https://github.com/cemyildizcy/turkey-earthquake-risk-analysis.git
git push -u origin main
```

### Görev 5.3 — LinkedIn paylaşımı hazırla
- Proje özeti
- Öne çıkan bulgular (en riskli iller, kümeleme sonuçları)
- Görsel ekle (harita, confusion matrix)
- Hashtag'ler: #DataScience #MachineLearning #Python #DepremAnalizi

---

## 🎓 Derste Öğrendiğin Konularla Eşleştirme

| Proje Aşaması | Derste Gördüğün Konu | Hocanın Kodundaki Karşılığı |
|---|---|---|
| Veri temizleme | Hafta 1-3: Pandas, istatistik | u1_istatistikler.py |
| EDA & Görselleştirme | Hafta 2-3: matplotlib, seaborn, korelasyon | u1_istatistikler.py |
| K-Means kümeleme | Hafta 7: Gözetimsiz öğrenme | u3_model_km.py |
| Sınıflandırma (LR) | Hafta 4: Lojistik regresyon | u1_model_lr.py |
| Sınıflandırma (RF) | Hafta 5: Random Forest | u2_model_rf.py |
| Sınıflandırma (SVM) | Hafta 6: SVM | u4_model_svm.py |
| Performans metrikleri | Hafta 4: Confusion matrix, F1, ROC | Sunum materyalleri |
| Train/Test split | (Hocanın yapmadığı!) | SEN EKLİYORSUN ✅ |
| Feature scaling | Hafta 6: StandardScaler | u4_model_svm.py |

---

## ⏰ Tahmini Süre: 3-4 saat (öğrenerek, mentor modunda)
- Aşama 1: ~45 dk
- Aşama 2: ~60 dk
- Aşama 3: ~45 dk
- Aşama 4: ~60 dk
- Aşama 5: ~30 dk
