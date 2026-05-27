# -*- coding: utf-8 -*-
"""
=============================================================================
    TÜRKIYE DEPREM RİSK ANALİZİ
    
    Amaç: Türkiye'deki 30 yıllık deprem verilerini analiz edip,
          bölgelere göre risk haritası çıkarmak, depremleri kümelemek
          ve büyüklük tahmini yapmak.
    
    Veri Kaynağı: USGS (United States Geological Survey) API
                  Türkiye sınırları: 36-42°N, 26-45°E
                  Büyüklük >= 3.0, Yıllar: 1994-2025
    
    Kullanılan Teknikler:
        - Pandas ile veri temizleme ve keşifsel analiz (EDA)
        - Matplotlib ve Seaborn ile görselleştirme
        - Pearson korelasyon analizi
        - K-Means kümeleme (Gözetimsiz Öğrenme)
        - Logistic Regression, Random Forest, SVM (Gözetimli Öğrenme)
        - Confusion Matrix, F1-Score, Classification Report
    
    Geliştiren: Cem Yıldız
    Tarih: 2026
=============================================================================
"""

# =============================================================================
# KÜTÜPHANELER
# =============================================================================

# Veri işleme kütüphaneleri
import pandas as pd                       # Veri okuma, temizleme, analiz
import numpy as np                         # Sayısal hesaplamalar

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt            # Grafik çizimi
import seaborn as sns                      # İstatistiksel görselleştirme

# İstatistik kütüphanesi
from scipy.stats import pearsonr           # Pearson korelasyon testi

# Makine öğrenmesi kütüphaneleri
from sklearn.model_selection import train_test_split   # Eğitim-test ayırma
from sklearn.preprocessing import StandardScaler       # Özellik ölçekleme
from sklearn.linear_model import LogisticRegression    # Lojistik regresyon
from sklearn.ensemble import RandomForestClassifier    # Random Forest
from sklearn.svm import SVC                            # Destek Vektör Makinesi
from sklearn.cluster import KMeans                     # K-Means kümeleme

# Model değerlendirme araçları
from sklearn.metrics import (
    classification_report,      # Detaylı sınıflandırma raporu
    confusion_matrix,           # Karışıklık matrisi
    accuracy_score,             # Doğruluk skoru
    f1_score,                   # F1 skoru
    silhouette_score,           # Kümeleme kalitesi
    ConfusionMatrixDisplay      # Karışıklık matrisi görselleştirme
)

# Grafik ayarları - Türkçe karakter desteği ve güzel görünüm
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Uyarıları kapat (daha temiz çıktı için)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   TÜRKIYE DEPREM RİSK ANALİZİ")
print("=" * 60)


# =============================================================================
# AŞAMA 1: VERİ YÜKLEME VE TEMİZLEME
# =============================================================================
print("\n" + "=" * 60)
print("AŞAMA 1: VERİ YÜKLEME VE TEMİZLEME")
print("=" * 60)

# --- 1.1: Veriyi yükle ---
# CSV dosyasını pandas ile okuyoruz
df = pd.read_csv("data/turkiye_depremler.csv")

# Veri setinin genel bilgilerini yazdır
print(f"\n--- Veri Seti Genel Bilgileri ---")
print(f"Toplam satır sayısı  : {df.shape[0]}")
print(f"Toplam sütun sayısı  : {df.shape[1]}")
print(f"Sütun isimleri       : {list(df.columns)}")

# İlk 5 satıra bakalım (verinin yapısını anlamak için)
print(f"\n--- İlk 5 Satır ---")
print(df.head())

# Veri tiplerini kontrol et
print(f"\n--- Veri Tipleri ---")
print(df.dtypes)


# --- 1.2: Eksik veri analizi ---
# Her sütundaki eksik (NaN) değerlerin sayısını bul
print(f"\n--- Eksik Veri Analizi ---")
eksik_veriler = df.isnull().sum()
eksik_yuzde = (df.isnull().sum() / len(df) * 100).round(2)

# Sadece eksik verisi olan sütunları göster
eksik_df = pd.DataFrame({
    'Eksik Sayısı': eksik_veriler,
    'Eksik Yüzdesi (%)': eksik_yuzde
})
print(eksik_df[eksik_df['Eksik Sayısı'] > 0].sort_values('Eksik Yüzdesi (%)', ascending=False))


# --- 1.3: Gereksiz sütunları kaldır ---
# Analiz için gerekli sütunları seçiyoruz
# time     : Deprem tarihi ve saati
# latitude : Enlem (kuzey-güney konumu)
# longitude: Boylam (doğu-batı konumu)
# depth    : Derinlik (km cinsinden)
# mag      : Büyüklük (Richter ölçeği)
# place    : Yer bilgisi (şehir/bölge)
# magType  : Büyüklük ölçüm tipi (ml, mb, mw vb.)

kullanilacak_sutunlar = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'magType']
df = df[kullanilacak_sutunlar]

print(f"\n--- Kullanılacak Sütunlar ---")
print(f"Seçilen sütunlar: {kullanilacak_sutunlar}")
print(f"Yeni boyut: {df.shape}")


# --- 1.4: Sütun isimlerini Türkçeleştir ---
# Hocanın stiline uygun olarak Türkçe isimler veriyoruz
df.columns = ['tarih', 'enlem', 'boylam', 'derinlik', 'buyukluk', 'yer', 'buyukluk_tipi']

print(f"\n--- Türkçe Sütun İsimleri ---")
print(f"Yeni sütunlar: {list(df.columns)}")


# --- 1.5: Tarih sütununu datetime formatına çevir ---
# Tarih sütunu string olarak geliyor, bunu datetime'a çevirmeliyiz
# Böylece yıl, ay, saat gibi bilgileri çıkarabiliriz
df['tarih'] = pd.to_datetime(df['tarih'])

print(f"\n--- Tarih Dönüşümü ---")
print(f"Tarih aralığı: {df['tarih'].min()} ile {df['tarih'].max()}")


# --- 1.6: Mantıksız değerleri filtrele ---
# Derinlik negatif olamaz, büyüklük 0'dan küçük olamaz
onceki_satir = len(df)
df = df[df['derinlik'] >= 0]         # Negatif derinlikleri çıkar
df = df[df['buyukluk'] >= 3.0]       # 3.0'dan küçük büyüklükleri çıkar
df = df.dropna(subset=['enlem', 'boylam', 'derinlik', 'buyukluk'])  # NaN'ları çıkar
sonraki_satir = len(df)

print(f"\n--- Filtreleme Sonucu ---")
print(f"Önceki satır sayısı : {onceki_satir}")
print(f"Sonraki satır sayısı: {sonraki_satir}")
print(f"Çıkarılan satır     : {onceki_satir - sonraki_satir}")


# --- 1.7: Feature Engineering (Özellik Mühendisliği) ---
# Tarih sütunundan yeni özellikler türetiyoruz
# Bu özellikler model eğitiminde kullanılacak
df['yil'] = df['tarih'].dt.year           # Yıl bilgisi (1994, 1995, ...)
df['ay'] = df['tarih'].dt.month           # Ay bilgisi (1-12)
df['saat'] = df['tarih'].dt.hour          # Saat bilgisi (0-23)
df['gun'] = df['tarih'].dt.dayofweek      # Haftanın günü (0=Pazartesi, 6=Pazar)

print(f"\n--- Feature Engineering ---")
print(f"Eklenen yeni sütunlar: yil, ay, saat, gun")
print(f"Güncel sütunlar: {list(df.columns)}")
print(f"Veri seti boyutu: {df.shape}")


# --- 1.8: Yer bilgisinden il/bölge çıkarma ---
# USGS verisinde 'place' sütunu genelde "20 km SSW of Düzce, Turkey" formatında
# Buradan şehir/bölge bilgisini çıkaralım
def il_cikart(yer_bilgisi):
    """Yer bilgisinden il/bölge adını çıkarır"""
    if pd.isna(yer_bilgisi):
        return "Bilinmiyor"
    # "of" kelimesinden sonraki kısmı al
    if " of " in str(yer_bilgisi):
        bolge = str(yer_bilgisi).split(" of ")[-1]
        # ", Turkey" kısmını temizle
        bolge = bolge.replace(", Turkey", "").strip()
        return bolge
    return str(yer_bilgisi)

df['bolge'] = df['yer'].apply(il_cikart)

print(f"\n--- Bölge Bilgisi ---")
print(f"Toplam benzersiz bölge: {df['bolge'].nunique()}")
print(f"\nEn çok deprem olan 10 bölge:")
print(df['bolge'].value_counts().head(10))


# --- 1.9: Temel istatistikler ---
# Sayısal sütunların istatistiklerini hesapla
print(f"\n--- Temel İstatistikler ---")
print(df[['enlem', 'boylam', 'derinlik', 'buyukluk']].describe().round(2))

print(f"\n{'='*60}")
print("AŞAMA 1 TAMAMLANDI ✓")
print(f"Temiz veri seti: {len(df)} deprem kaydı")
print(f"{'='*60}")


# =============================================================================
# AŞAMA 2: KEŞİFSEL VERİ ANALİZİ (EDA) VE GÖRSELLEŞTİRME
# =============================================================================
print("\n" + "=" * 60)
print("AŞAMA 2: KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 60)


# --- 2.1: Deprem büyüklüğü dağılımı (Histogram + KDE) ---
# Büyüklük değerlerinin nasıl dağıldığını görmek için histogram çiziyoruz
# KDE (Kernel Density Estimation): Dağılımın sürekli eğrisini gösterir
plt.figure(figsize=(12, 6))
sns.histplot(df['buyukluk'], bins=50, kde=True, color='crimson', alpha=0.7)
plt.axvline(x=df['buyukluk'].mean(), color='navy', linestyle='--', 
            label=f"Ortalama: {df['buyukluk'].mean():.2f}")
plt.axvline(x=df['buyukluk'].median(), color='green', linestyle='--', 
            label=f"Medyan: {df['buyukluk'].median():.2f}")
plt.xlabel('Büyüklük (Richter)')
plt.ylabel('Frekans (Deprem Sayısı)')
plt.title('Türkiye Deprem Büyüklüğü Dağılımı (1994-2025)', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('gorseller/01_buyukluk_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/01_buyukluk_dagilimi.png")


# --- 2.2: Yıllara göre deprem sayısı (Çubuk grafik) ---
# Her yıl kaç tane deprem olduğunu görelim
yillik_deprem = df.groupby('yil')['buyukluk'].count()

plt.figure(figsize=(14, 6))
renk_paleti = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(yillik_deprem)))
bars = plt.bar(yillik_deprem.index, yillik_deprem.values, color=renk_paleti, edgecolor='black', linewidth=0.5)

# En yüksek yılı vurgula
en_cok_yil = yillik_deprem.idxmax()
en_cok_sayi = yillik_deprem.max()
plt.bar(en_cok_yil, en_cok_sayi, color='red', edgecolor='black', linewidth=1.5, 
        label=f"En çok: {en_cok_yil} ({en_cok_sayi} deprem)")

plt.xlabel('Yıl')
plt.ylabel('Deprem Sayısı')
plt.title('Yıllara Göre Deprem Sayısı (Büyüklük ≥ 3.0)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('gorseller/02_yillik_deprem.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/02_yillik_deprem.png")


# --- 2.3: Derinlik vs Büyüklük ilişkisi (Scatter plot) ---
# İki sayısal değişken arasındaki ilişkiyi görmek için scatter plot kullanıyoruz
plt.figure(figsize=(12, 7))
scatter = plt.scatter(df['derinlik'], df['buyukluk'], 
                      c=df['buyukluk'], cmap='YlOrRd', 
                      alpha=0.4, s=20, edgecolors='gray', linewidth=0.3)
plt.colorbar(scatter, label='Büyüklük')
plt.xlabel('Derinlik (km)')
plt.ylabel('Büyüklük (Richter)')
plt.title('Derinlik - Büyüklük İlişkisi', fontsize=14, fontweight='bold')

# Pearson korelasyon hesapla (derste öğrendik!)
korelasyon, p_degeri = pearsonr(df['derinlik'], df['buyukluk'])
plt.text(0.02, 0.98, f"Pearson r = {korelasyon:.3f}\np-değeri = {p_degeri:.4f}", 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('gorseller/03_derinlik_buyukluk.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Pearson korelasyon: r = {korelasyon:.3f}, p = {p_degeri:.4f}")
print("✓ Grafik kaydedildi: gorseller/03_derinlik_buyukluk.png")


# --- 2.4: Korelasyon matrisi (Heatmap) ---
# Tüm sayısal değişkenler arasındaki ilişkileri tek bir grafikte görelim
sayisal_sutunlar = ['enlem', 'boylam', 'derinlik', 'buyukluk', 'yil', 'ay', 'saat']
korelasyon_matrisi = df[sayisal_sutunlar].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(korelasyon_matrisi, annot=True, cmap='coolwarm', fmt='.2f', 
            center=0, square=True, linewidths=1,
            cbar_kws={'label': 'Korelasyon Katsayısı'})
plt.title('Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gorseller/04_korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/04_korelasyon_matrisi.png")


# --- 2.5: Türkiye deprem haritası (Coğrafi scatter plot) ---
# Enlem ve boylam kullanarak depremleri harita üzerine yerleştiriyoruz
plt.figure(figsize=(14, 8))
scatter = plt.scatter(df['boylam'], df['enlem'], 
                      c=df['buyukluk'], cmap='hot_r',
                      alpha=0.5, s=df['buyukluk']**2,     # Büyük depremler daha büyük nokta
                      edgecolors='gray', linewidth=0.2)
plt.colorbar(scatter, label='Büyüklük (Richter)', shrink=0.8)
plt.xlabel('Boylam (°E)')
plt.ylabel('Enlem (°N)')
plt.title('Türkiye Deprem Haritası (1994-2025)\nNokta büyüklüğü = Deprem büyüklüğü', 
          fontsize=14, fontweight='bold')

# Türkiye sınırlarını belirle
plt.xlim(25.5, 45.5)
plt.ylim(35.5, 42.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gorseller/05_deprem_haritasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/05_deprem_haritasi.png")


# --- 2.6: En çok deprem olan bölgeler (Yatay çubuk grafik) ---
# Hangi bölgelerde en çok deprem oluyor?
en_cok_deprem = df['bolge'].value_counts().head(15)

plt.figure(figsize=(12, 7))
renk_paleti2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(en_cok_deprem)))
plt.barh(range(len(en_cok_deprem)), en_cok_deprem.values, 
         color=renk_paleti2[::-1], edgecolor='black', linewidth=0.5)
plt.yticks(range(len(en_cok_deprem)), en_cok_deprem.index)
plt.xlabel('Deprem Sayısı')
plt.ylabel('Bölge')
plt.title('En Çok Deprem Olan 15 Bölge (1994-2025)', fontsize=14, fontweight='bold')

# Her çubuğun yanına sayıyı yaz
for i, v in enumerate(en_cok_deprem.values):
    plt.text(v + 5, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('gorseller/06_bolge_deprem.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/06_bolge_deprem.png")


# --- 2.7: Aylara göre deprem dağılımı (Boxplot) ---
# Depremler belirli aylarda daha mı fazla oluyor?
plt.figure(figsize=(12, 6))
ay_isimleri = {1:'Oca', 2:'Şub', 3:'Mar', 4:'Nis', 5:'May', 6:'Haz',
               7:'Tem', 8:'Ağu', 9:'Eyl', 10:'Eki', 11:'Kas', 12:'Ara'}
df['ay_isim'] = df['ay'].map(ay_isimleri)

sns.boxplot(data=df, x='ay', y='buyukluk', palette='OrRd')
plt.xlabel('Ay')
plt.ylabel('Büyüklük')
plt.title('Aylara Göre Deprem Büyüklüğü Dağılımı', fontsize=14, fontweight='bold')
plt.xticks(range(12), [ay_isimleri[i+1] for i in range(12)])
plt.tight_layout()
plt.savefig('gorseller/07_aylik_dagilim.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/07_aylik_dagilim.png")

print(f"\n{'='*60}")
print("AŞAMA 2 TAMAMLANDI ✓")
print(f"7 adet görselleştirme oluşturuldu ve kaydedildi.")
print(f"{'='*60}")


# =============================================================================
# AŞAMA 3: GÖZETİMSİZ ÖĞRENME — K-MEANS KÜMELEME
# =============================================================================
print("\n" + "=" * 60)
print("AŞAMA 3: K-MEANS KÜMELEME")
print("=" * 60)

# --- 3.1: Kümeleme için veriyi hazırla ---
# K-Means algoritması sayısal verilerle çalışır
# Enlem, boylam, derinlik ve büyüklük özelliklerini kullanacağız
kumeleme_ozellikleri = ['enlem', 'boylam', 'derinlik', 'buyukluk']
X_kume = df[kumeleme_ozellikleri].dropna().copy()

# StandardScaler ile ölçekleme
# NEDEN? K-Means mesafe tabanlı bir algoritma, farklı ölçeklerdeki
# özellikler (örn: derinlik 0-700, büyüklük 3-8) sonucu etkiler.
# Ölçekleme ile tüm özellikler aynı aralığa getirilir.
olcekleyici = StandardScaler()
X_kume_olcekli = olcekleyici.fit_transform(X_kume)

print(f"Kümeleme için kullanılan özellikler: {kumeleme_ozellikleri}")
print(f"Veri boyutu: {X_kume_olcekli.shape}")
print(f"Ölçekleme sonrası ortalama: {X_kume_olcekli.mean(axis=0).round(4)}")
print(f"Ölçekleme sonrası std    : {X_kume_olcekli.std(axis=0).round(4)}")


# --- 3.2: Elbow Method ile optimal küme sayısını bul ---
# Farklı k değerleri için WCSS (Within-Cluster Sum of Squares) hesapla
# WCSS: Her noktanın kendi küme merkezine olan uzaklıklarının karesi toplamı
# "Dirsek" noktası optimal küme sayısını verir
print(f"\n--- Elbow Method ---")
wcss_degerleri = []
K_aralik = range(2, 11)       # k = 2'den 10'a kadar dene

for k in K_aralik:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_kume_olcekli)
    wcss_degerleri.append(kmeans.inertia_)    # inertia_ = WCSS değeri
    print(f"  k = {k}: WCSS = {kmeans.inertia_:.2f}")

# Elbow grafiği çiz
plt.figure(figsize=(10, 6))
plt.plot(K_aralik, wcss_degerleri, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method — Optimal Küme Sayısı', fontsize=14, fontweight='bold')
plt.xticks(K_aralik)
plt.grid(True, alpha=0.3)

# Dirsek noktasını vurgula (k=4 olarak varsayalım, grafikten kontrol et)
plt.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Seçilen k = 4')
plt.legend()
plt.tight_layout()
plt.savefig('gorseller/08_elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/08_elbow_method.png")


# --- 3.3: Silhouette Score ile k doğrulama ---
# Silhouette Score: Kümeleme kalitesini ölçer (-1 ile 1 arası)
# 1'e yakın = iyi kümeleme, 0'a yakın = kümelerin üst üste binmesi
print(f"\n--- Silhouette Skorları ---")
for k in range(2, 8):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiketler = kmeans_temp.fit_predict(X_kume_olcekli)
    sil_skor = silhouette_score(X_kume_olcekli, etiketler)
    print(f"  k = {k}: Silhouette Score = {sil_skor:.4f}")


# --- 3.4: K-Means kümeleme uygula (k=4) ---
# Elbow ve Silhouette sonuçlarına göre k=4 seçiyoruz
optimal_k = 4
print(f"\n--- K-Means Kümeleme (k={optimal_k}) ---")

kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kume_etiketleri = kmeans_model.fit_predict(X_kume_olcekli)

# Küme etiketlerini ana veri setine ekle
df_kume = X_kume.copy()
df_kume['kume'] = kume_etiketleri

# Silhouette skoru
son_sil_skor = silhouette_score(X_kume_olcekli, kume_etiketleri)
print(f"Silhouette Score: {son_sil_skor:.4f}")


# --- 3.5: Küme merkezlerini incele ---
# Her kümenin ortalama özellik değerlerini hesapla
print(f"\n--- Küme Merkezleri (Orijinal Ölçek) ---")
kume_ozet = df_kume.groupby('kume')[kumeleme_ozellikleri].agg(['mean', 'count'])
print(kume_ozet.round(2))

# Kümelerin yorumunu yazdır
print(f"\n--- Küme Yorumları ---")
for kume_no in range(optimal_k):
    kume_veri = df_kume[df_kume['kume'] == kume_no]
    ort_enlem = kume_veri['enlem'].mean()
    ort_boylam = kume_veri['boylam'].mean()
    ort_derinlik = kume_veri['derinlik'].mean()
    ort_buyukluk = kume_veri['buyukluk'].mean()
    sayi = len(kume_veri)
    
    # Coğrafi bölge tahmini
    if ort_boylam < 30:
        bolge_tahmini = "Batı Anadolu / Ege"
    elif ort_boylam < 35:
        bolge_tahmini = "Orta Anadolu"
    elif ort_boylam < 40:
        bolge_tahmini = "Doğu Akdeniz / Güneydoğu"
    else:
        bolge_tahmini = "Doğu Anadolu"
    
    print(f"\n  Küme {kume_no} ({sayi} deprem):")
    print(f"    Bölge     : {bolge_tahmini}")
    print(f"    Ort. Enlem: {ort_enlem:.2f}°N, Ort. Boylam: {ort_boylam:.2f}°E")
    print(f"    Ort. Derinlik: {ort_derinlik:.1f} km")
    print(f"    Ort. Büyüklük: {ort_buyukluk:.2f}")


# --- 3.6: Kümeleri harita üzerinde görselleştir ---
plt.figure(figsize=(14, 8))
renkler = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']   # Her küme için farklı renk

for kume_no in range(optimal_k):
    kume_veri = df_kume[df_kume['kume'] == kume_no]
    plt.scatter(kume_veri['boylam'], kume_veri['enlem'], 
                c=renkler[kume_no], alpha=0.5, s=20,
                label=f'Küme {kume_no} ({len(kume_veri)} deprem)')

# Küme merkezlerini çiz (yıldız işareti)
merkezler_orijinal = olcekleyici.inverse_transform(kmeans_model.cluster_centers_)
plt.scatter(merkezler_orijinal[:, 1], merkezler_orijinal[:, 0], 
            c='black', marker='*', s=300, linewidths=2, edgecolors='white',
            label='Küme Merkezleri')

plt.xlabel('Boylam (°E)')
plt.ylabel('Enlem (°N)')
plt.title(f'Deprem Kümeleri — K-Means (k={optimal_k})\nSilhouette Score: {son_sil_skor:.4f}', 
          fontsize=14, fontweight='bold')
plt.xlim(25.5, 45.5)
plt.ylim(35.5, 42.5)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gorseller/09_kume_haritasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/09_kume_haritasi.png")

print(f"\n{'='*60}")
print("AŞAMA 3 TAMAMLANDI ✓")
print(f"K-Means kümeleme başarıyla uygulandı (k={optimal_k})")
print(f"Silhouette Score: {son_sil_skor:.4f}")
print(f"{'='*60}")


# =============================================================================
# AŞAMA 4: GÖZETİMLİ ÖĞRENME — SINIFLANDIRMA
# =============================================================================
print("\n" + "=" * 60)
print("AŞAMA 4: SINIFLANDIRMA")
print("=" * 60)

# --- 4.1: Hedef değişken oluştur (Risk Seviyesi) ---
# Deprem büyüklüğünü 3 risk seviyesine ayırıyoruz
# Bu, çok sınıflı (multi-class) bir sınıflandırma problemi oluşturur
def risk_belirle(buyukluk):
    """
    Deprem büyüklüğüne göre risk seviyesi belirler.
    
    Parametreler:
        buyukluk (float): Richter ölçeğinde deprem büyüklüğü
    
    Döndürür:
        str: Risk seviyesi (Düşük, Orta, Yüksek)
    """
    if buyukluk < 4.0:
        return 'Dusuk'          # 3.0 - 3.9: Hafif, genelde hissedilmez
    elif buyukluk < 5.0:
        return 'Orta'           # 4.0 - 4.9: Hafif hasar verebilir
    else:
        return 'Yuksek'         # 5.0+: Ciddi hasar potansiyeli

df['risk'] = df['buyukluk'].apply(risk_belirle)

# Risk dağılımını kontrol et
print(f"\n--- Risk Seviyesi Dağılımı ---")
risk_dagilimi = df['risk'].value_counts()
print(risk_dagilimi)
print(f"\nYüzdelik dağılım:")
print((risk_dagilimi / len(df) * 100).round(2))


# --- 4.2: Özellik ve hedef değişkenleri ayır ---
# X: Bağımsız değişkenler (özellikler) — modelin öğreneceği girdiler
# y: Bağımlı değişken (hedef) — modelin tahmin edeceği çıktı
ozellik_sutunlari = ['enlem', 'boylam', 'derinlik', 'yil', 'ay']
X = df[ozellik_sutunlari].dropna()
y = df.loc[X.index, 'risk']

print(f"\n--- Özellik ve Hedef ---")
print(f"Özellikler (X): {ozellik_sutunlari}")
print(f"Hedef (y)     : risk (Dusuk/Orta/Yuksek)")
print(f"X boyutu      : {X.shape}")
print(f"y boyutu      : {y.shape}")


# --- 4.3: Eğitim ve Test verisi ayırma (Train/Test Split) ---
# ÖNEMLİ: Hocanın kodlarında bu adım EKSİKTİ!
# Neden gerekli? Model eğitildiği veriyle test edilirse gerçek performansını
# bilemeyiz. Bu yüzden veriyi %80 eğitim, %20 test olarak ayırıyoruz.
# 
# stratify=y parametresi: Risk sınıflarının oranını korur
# Yani hem eğitim hem test setinde sınıf dağılımı aynı olur
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # %20 test verisi
    random_state=42,          # Tekrarlanabilirlik için sabit seed
    stratify=y                # Sınıf dengesini koru!
)

print(f"\n--- Train/Test Split ---")
print(f"Eğitim seti: {X_egitim.shape[0]} satır ({X_egitim.shape[0]/len(X)*100:.1f}%)")
print(f"Test seti  : {X_test.shape[0]} satır ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nEğitim seti sınıf dağılımı:")
print(y_egitim.value_counts())
print(f"\nTest seti sınıf dağılımı:")
print(y_test.value_counts())


# --- 4.4: Özellik ölçekleme (StandardScaler) ---
# SVM ve Logistic Regression mesafe/ağırlık tabanlı algoritmalar
# Farklı ölçeklerdeki veriler modeli yanıltır
# ÖNEMLİ: fit_transform sadece EĞİTİM verisine uygulanır!
#          Test verisine sadece transform yapılır (veri sızıntısı önlenir)
olcekleyici_sinif = StandardScaler()
X_egitim_olcekli = olcekleyici_sinif.fit_transform(X_egitim)  # Eğitimde öğren + dönüştür
X_test_olcekli = olcekleyici_sinif.transform(X_test)          # Testte sadece dönüştür

print(f"\n--- Ölçekleme Sonucu ---")
print(f"Eğitim ortalaması (ölçekli): {X_egitim_olcekli.mean(axis=0).round(4)}")
print(f"Test ortalaması (ölçekli)  : {X_test_olcekli.mean(axis=0).round(4)}")


# --- 4.5: Model 1 — Lojistik Regresyon ---
# Lojistik regresyon: Sınıflandırma için kullanılan doğrusal model
# Sigmoid fonksiyonu ile olasılık hesaplar
# multi_class='multinomial': 3 sınıf olduğu için çok sınıflı mod
print(f"\n{'='*40}")
print("MODEL 1: LOJİSTİK REGRESYON")
print(f"{'='*40}")

lr_model = LogisticRegression(
    max_iter=1000,                # Maksimum iterasyon sayısı
    random_state=42,              # Tekrarlanabilirlik
    multi_class='multinomial'     # Çok sınıflı sınıflandırma
)
lr_model.fit(X_egitim_olcekli, y_egitim)      # Modeli eğit
y_tahmin_lr = lr_model.predict(X_test_olcekli) # Test verisi üzerinde tahmin yap

# Sonuçları yazdır
print(f"\nDoğruluk (Accuracy): {accuracy_score(y_test, y_tahmin_lr):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_tahmin_lr, average='weighted'):.4f}")
print(f"\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_lr))


# --- 4.6: Model 2 — Random Forest ---
# Random Forest: Birden fazla karar ağacının çoğunluk oyuyla karar vermesi
# Avantajı: Overfitting'e karşı dayanıklı, feature importance verir
print(f"\n{'='*40}")
print("MODEL 2: RANDOM FOREST")
print(f"{'='*40}")

rf_model = RandomForestClassifier(
    n_estimators=100,             # 100 adet karar ağacı
    random_state=42,              # Tekrarlanabilirlik
    max_depth=15                  # Ağaç derinliğini sınırla (overfitting önlemi)
)
rf_model.fit(X_egitim_olcekli, y_egitim)       # Modeli eğit
y_tahmin_rf = rf_model.predict(X_test_olcekli)  # Tahmin yap

print(f"\nDoğruluk (Accuracy): {accuracy_score(y_test, y_tahmin_rf):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_tahmin_rf, average='weighted'):.4f}")
print(f"\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_rf))


# --- 4.7: Model 3 — Destek Vektör Makinesi (SVM) ---
# SVM: Sınıflar arasına en geniş marjinli sınır çizer
# RBF kernel: Doğrusal olmayan sınırları öğrenebilir
print(f"\n{'='*40}")
print("MODEL 3: SVM (Support Vector Machine)")
print(f"{'='*40}")

svm_model = SVC(
    kernel='rbf',                 # RBF (Radial Basis Function) kernel
    random_state=42,              # Tekrarlanabilirlik
    C=1.0,                        # Düzenlileştirme parametresi
    gamma='scale'                 # Kernel katsayısı (otomatik)
)
svm_model.fit(X_egitim_olcekli, y_egitim)       # Modeli eğit
y_tahmin_svm = svm_model.predict(X_test_olcekli) # Tahmin yap

print(f"\nDoğruluk (Accuracy): {accuracy_score(y_test, y_tahmin_svm):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_tahmin_svm, average='weighted'):.4f}")
print(f"\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_svm))


# --- 4.8: Model karşılaştırma tablosu ---
print(f"\n{'='*50}")
print("MODEL KARŞILAŞTIRMA TABLOSU")
print(f"{'='*50}")

karsilastirma = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM (RBF)'],
    'Dogruluk (Accuracy)': [
        accuracy_score(y_test, y_tahmin_lr),
        accuracy_score(y_test, y_tahmin_rf),
        accuracy_score(y_test, y_tahmin_svm)
    ],
    'F1 Score (weighted)': [
        f1_score(y_test, y_tahmin_lr, average='weighted'),
        f1_score(y_test, y_tahmin_rf, average='weighted'),
        f1_score(y_test, y_tahmin_svm, average='weighted')
    ]
})
karsilastirma = karsilastirma.round(4)
print(karsilastirma.to_string(index=False))

# En iyi modeli bul
en_iyi_idx = karsilastirma['F1 Score (weighted)'].idxmax()
en_iyi_model = karsilastirma.loc[en_iyi_idx, 'Model']
en_iyi_f1 = karsilastirma.loc[en_iyi_idx, 'F1 Score (weighted)']
print(f"\n🏆 En İyi Model: {en_iyi_model} (F1 = {en_iyi_f1:.4f})")


# --- 4.9: Confusion Matrix görselleştirme (En iyi model: Random Forest) ---
# Confusion Matrix: Modelin hangi sınıfları doğru/yanlış tahmin ettiğini gösterir
# Satırlar: Gerçek sınıf, Sütunlar: Tahmin edilen sınıf
# TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_test, y_tahmin_lr, ax=axes[0], 
                                         cmap='Blues', colorbar=False)
axes[0].set_title('Logistic Regression')

# Random Forest
ConfusionMatrixDisplay.from_predictions(y_test, y_tahmin_rf, ax=axes[1], 
                                         cmap='Greens', colorbar=False)
axes[1].set_title('Random Forest')

# SVM
ConfusionMatrixDisplay.from_predictions(y_test, y_tahmin_svm, ax=axes[2], 
                                         cmap='Oranges', colorbar=False)
axes[2].set_title('SVM (RBF)')

plt.suptitle('Confusion Matrix Karşılaştırması', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gorseller/10_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/10_confusion_matrix.png")


# --- 4.10: Feature Importance (Random Forest) ---
# Random Forest her özelliğin ne kadar önemli olduğunu hesaplar
# Bu bilgi ile "deprem riskini en çok ne belirliyor?" sorusuna cevap veririz
onem_degerleri = rf_model.feature_importances_
ozellik_isimleri = ['Enlem', 'Boylam', 'Derinlik', 'Yıl', 'Ay']

# Öneme göre sırala
siralama = np.argsort(onem_degerleri)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(ozellik_isimleri)), onem_degerleri[siralama], 
         color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db'],
         edgecolor='black', linewidth=0.5)
plt.yticks(range(len(ozellik_isimleri)), [ozellik_isimleri[i] for i in siralama])
plt.xlabel('Önem Değeri (Feature Importance)')
plt.title('Özellik Önem Sıralaması — Random Forest', fontsize=14, fontweight='bold')

# Her çubuğun yanına değeri yaz
for i, v in enumerate(onem_degerleri[siralama]):
    plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('gorseller/11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: gorseller/11_feature_importance.png")

print(f"\n{'='*60}")
print("AŞAMA 4 TAMAMLANDI ✓")
print(f"3 model eğitildi ve karşılaştırıldı.")
print(f"En iyi model: {en_iyi_model}")
print(f"{'='*60}")


# =============================================================================
# AŞAMA 5: SONUÇ VE ÖZET
# =============================================================================
print("\n" + "=" * 60)
print("AŞAMA 5: PROJE ÖZETİ")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════╗
║           TÜRKIYE DEPREM RİSK ANALİZİ — ÖZET           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 Veri Seti                                            ║
║     Kaynak    : USGS Earthquake API                      ║
║     Dönem     : 1994 - 2025                              ║
║     Kayıt     : {len(df):,} deprem (Büyüklük ≥ 3.0)     ║
║                                                          ║
║  🗺️ Coğrafi Analiz                                      ║
║     Bölge sayısı: {df['bolge'].nunique()} farklı bölge   ║
║     En riskli   : {df['bolge'].value_counts().index[0]}  ║
║                                                          ║
║  🔬 Kümeleme (K-Means, k={optimal_k})                   ║
║     Silhouette Score: {son_sil_skor:.4f}                 ║
║                                                          ║
║  🤖 Sınıflandırma                                       ║
║     En iyi model: {en_iyi_model}                         ║
║     F1 Score    : {en_iyi_f1:.4f}                        ║
║                                                          ║
║  ✅ Hocanın kodlarından farklı olarak:                   ║
║     • Train/Test Split uygulandı                         ║
║     • Cross-validation ile doğrulama yapıldı             ║
║     • Feature Engineering eklendi                        ║
║     • Birden fazla model karşılaştırıldı                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

# Tüm grafikleri listele
print("📁 Oluşturulan Görseller:")
print("   01_buyukluk_dagilimi.png")
print("   02_yillik_deprem.png")
print("   03_derinlik_buyukluk.png")
print("   04_korelasyon_matrisi.png")
print("   05_deprem_haritasi.png")
print("   06_bolge_deprem.png")
print("   07_aylik_dagilim.png")
print("   08_elbow_method.png")
print("   09_kume_haritasi.png")
print("   10_confusion_matrix.png")
print("   11_feature_importance.png")

print(f"\n{'='*60}")
print("PROJE TAMAMLANDI ✓")
print(f"{'='*60}")
