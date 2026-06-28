import pandas as pd
import numpy as np

class TarimVeriMotoru:
    """
    Nesne Tabanlı Veri Üretim Sınıfı.
    Yalnızca Linear Regression, SVM ve K-Means algoritmalarına uygun,
    temiz ve bilimsel (Gauss Dağılımlı) veri üretimi sağlar.
    """
    def __init__(self, ilk_kayit_sayisi=1000):
        self.df = self._veri_uret(ilk_kayit_sayisi)

    def _veri_uret(self, adet):
        np.random.seed(42) # Eğitimin tutarlı olması için
        
        lat = np.random.uniform(37.0500, 37.0800, adet)
        lon = np.random.uniform(37.3500, 37.4000, adet)
        
        sicaklik = np.clip(np.random.normal(25, 8, adet), 5, 50)  
        nem = np.clip(np.random.normal(50, 20, adet), 10, 95)      
        ph = np.clip(np.random.normal(7.0, 1.5, adet), 4.0, 10.0)   
        
        toprak_kodlari = np.random.choice([1, 2, 3], size=adet, p=[0.3, 0.5, 0.2]) 
        toprak_sozlugu = {1: 'Kumlu', 2: 'Tınlı', 3: 'Killi'}
        toprak_turu = [toprak_sozlugu[k] for k in toprak_kodlari]
        
        su_baz = np.random.normal(250, 80, adet)
        su_carpan = np.where(toprak_kodlari == 1, 1.3, np.where(toprak_kodlari == 3, 0.8, 1.0))
        su = np.clip(su_baz * su_carpan, 50, 800)
        gubre = np.clip(np.random.normal(30, 15, adet), 5, 100)
        
        # ==========================================
        # ÖZELLİK ÇIKARIMI (FEATURE ENGINEERING)
        # ==========================================
        su_gubre_orani = su / gubre
        
        # YENİ VE GERÇEKÇİ BİYOLOJİK MANTIK (Mutlak Sapma - Absolute Deviation)
        # İdeal değerlerden (Sıcaklık:25, Nem:50, pH:7.0, Su/Gübre Oranı:~8) uzaklaştıkça stres artar.
        # Sıcaklık 10'a da düşse, 40'a da çıksa (fark 15 olduğu için) eşit oranda stres yaratır.
        sicaklik_stresi = np.abs(sicaklik - 25) * 3.5
        nem_stresi = np.abs(nem - 50) * 0.5
        ph_stresi = np.abs(ph - 7.0) * 20
        besin_stresi = np.abs(su_gubre_orani - 8) * 2
        
        stres_indeksi = sicaklik_stresi + nem_stresi + ph_stresi + besin_stresi
        
        # ==========================================
        # HEDEF DEĞİŞKENLER (ML İÇİN)
        # ==========================================
        # Hedef 1: Rekolte (Linear Regression)
        rekolte_baz = 2500
        rekolte = rekolte_baz - (stres_indeksi * 12)
        rekolte += np.random.normal(0, 50, adet) # Sisteme doğal gürültü ekleme
        rekolte = np.clip(rekolte, 100, 3500)
        
        # Hedef 2: Sağlık Durumu (SVM)
        # Ekstrem düşük veya ekstrem yüksek değerler bitkiyi doğrudan hastalandırır.
        saglik_durumu = np.where(
            stres_indeksi > 80, 'Hastalıklı',
            np.where(stres_indeksi > 45, 'Riskli', 'Sağlıklı')
        )
        
        zamanlar = [pd.Timestamp.now() - pd.Timedelta(hours=i) for i in range(adet, 0, -1)]
        
        df = pd.DataFrame({
            'Zaman': zamanlar,
            'Enlem': lat, 'Boylam': lon,
            'Sicaklik_C': sicaklik, 'Nem_Yuzde': nem, 'Toprak_pH': ph,
            'Toprak_Turu': toprak_turu, 
            'Sulama_Litre': su, 'Gubre_KG': gubre, 'Su_Gubre_Orani': su_gubre_orani,
            'Stres_Indeksi': stres_indeksi,
            'Rekolte_KG': rekolte, 'Saglik_Durumu': saglik_durumu
        })
        return df

    def veri_getir(self):
        return self.df

    def sifirla(self, adet=1000):
        self.df = self._veri_uret(adet)