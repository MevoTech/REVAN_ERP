import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scipy.stats as stats

class ArayuzYoneticisi:
    def __init__(self, veri_motoru):
        self.veri_motoru = veri_motoru
        self.df = self.veri_motoru.veri_getir()
        self._sistemi_baslat()

    def css_enjekte_et(self):
        """Kurumsal Açık Tema (Clean Light) SaaS UI/UX Tasarım Katmanı"""
        custom_css = """
        <style>
            /* Kurumsal Açık Tema */
            .stApp { background-color: #F8FAFC; color: #0F172A; font-family: 'Inter', sans-serif; }
            header {visibility: hidden;} 
            
            /* Ortak Konteyner ve Kartlar */
            .section-container { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 30px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
            
            /* Tahmin Çıktı Kartları */
            .pred-card { text-align: center; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; background-color: #F8FAFC; transition: all 0.3s ease;}
            .pred-card:hover { border-color: #0284C7; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.1); transform: translateY(-2px);}
            .pred-title { font-size: 0.85rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; }
            .pred-value { font-size: 1.4rem; font-weight: 800; }
            
            /* Tipografi */
            .section-title { color: #0F172A; font-size: 1.6rem; font-weight: 700; margin-bottom: 20px; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px;}
            .sub-title { color: #0284C7; font-size: 1.1rem; font-weight: 700; margin-bottom: 15px;}
            
            /* Tablo Stilleri */
            .data-table { width: 100%; border-collapse: collapse; background-color: #FFFFFF; border-radius: 8px; overflow: hidden; border: 1px solid #E2E8F0; margin-top:15px;}
            .data-table th { background-color: #F1F5F9; color: #0284C7; padding: 15px; font-weight: 700; text-align: center; font-size: 0.95rem; border-bottom: 2px solid #E2E8F0;}
            .data-table td { padding: 15px; border-bottom: 1px solid #E2E8F0; color: #334155; font-size: 0.9rem; vertical-align: middle; text-align: center;}
            .data-table tr:hover td { background-color: #F8FAFC; }
            .text-left { text-align: left !important; }
            
            /* Mimari Kartları */
            .arch-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; text-align: left; height:100%; }
            .arch-title { color: #0284C7; font-weight: 800; font-size: 1rem; margin-bottom: 10px; }
            .arch-desc { color: #475569; font-size: 0.85rem; line-height: 1.5; }
            
            hr { border-color: #E2E8F0; margin: 2rem 0;}
            
            /* Streamlit Girdi Bileşenleri (Açık tema uyumu) */
            div[data-testid="stSlider"] label { color: #0F172A !important; font-weight: 600; }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def _sistemi_baslat(self):
        """Makine öğrenmesi modellerini bellekte hazırlar (Sıfır Gecikme)"""
        self.X = self.df[['Sicaklik_C', 'Nem_Yuzde', 'Toprak_pH', 'Sulama_Litre', 'Gubre_KG', 'Su_Gubre_Orani']]
        self.y_reg = self.df['Rekolte_KG']

        if 'sistem_hazir' not in st.session_state:
            self.le = LabelEncoder()
            self.y_clf = self.le.fit_transform(self.df['Saglik_Durumu'])
            
            # Eğitim ve Test Ayırımı
            X_train, X_test, y_train_reg, y_test_reg = train_test_split(self.X, self.y_reg, test_size=0.2, random_state=42)
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(self.X, self.y_clf, test_size=0.2, random_state=42)
            
            # Normalizasyon (Min-Max)
            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 1. Regresyon Modeli
            lr_model = LinearRegression().fit(X_train_scaled, y_train_reg)
            
            # 2. Sınıflandırma Modelleri
            svm_rbf = SVC(kernel='rbf', probability=True, random_state=42).fit(X_train_scaled, y_train_clf)
            k_val = int(np.sqrt(len(X_train_scaled)))
            k_val = k_val + 1 if k_val % 2 == 0 else k_val
            knn_model = KNeighborsClassifier(n_neighbors=k_val).fit(X_train_scaled, y_train_clf)
            
            # 3. Kümeleme Modeli
            kmeans_scaler = MinMaxScaler()
            X_kmeans_scaled = kmeans_scaler.fit_transform(self.df[['Toprak_pH', 'Nem_Yuzde']])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_kmeans_scaled)

            st.session_state.update({
                'lr_model': lr_model, 'svm_model': svm_rbf, 'knn_model': knn_model, 'kmeans': kmeans,
                'scaler': self.scaler, 'kmeans_scaler': kmeans_scaler, 'le': self.le,
                'X_test_scaled': X_test_scaled, 'y_test_reg': y_test_reg, 'y_test_clf': y_test_clf,
                'k_degeri': k_val,
                'sistem_hazir': True
            })

    def _apply_theme(self, fig):
        """Açık temaya uygun Plotly arka plan ve ızgara ayarları"""
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#475569'),
            title_font=dict(color='#0F172A', size=15, family='Inter', weight='bold'), 
            xaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0'), 
            yaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0'),
            margin=dict(t=50, l=10, r=10, b=20)
        )
        return fig

    def render_dashboard(self):
        # ==========================================
        # 1. KİMLİK VE PLATFORM AMACI
        # ==========================================
        st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style='color: #0F172A; font-weight: 900; font-size: 3.2rem; margin-bottom: 5px; letter-spacing: 1px;'>REVAN AI</h1>
            <p style='color: #0284C7; font-size: 1.3rem; font-weight: 700; letter-spacing: 2px;'>TARIMSAL KARAR DESTEK VE MERKEZİ ANALİZ PLATFORMU</p>
            <div style="background-color: #F8FAFC; display: inline-block; padding: 10px 25px; border-radius: 20px; margin-top: 15px; border: 1px solid #E2E8F0;">
                <p style='color: #334155; font-size: 0.95rem; margin: 0;'>
                    <b>Sistem Mimarı:</b> Mehmet GÜNEŞ (230007047) | <b>Danışman:</b> Dr. Ögr. Mehmet Sait VURAL | <b>Proje:</b> Tarımsal Yapay Zeka ve IoT Destekli Karar Sistemi
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- SİSTEM MİMARİSİ ----------------
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🏗️ Sistem Mimarisi ve Bilgi Akışı</div>", unsafe_allow_html=True)
        
        c_arc1, c_arc2, c_arc3, c_arc4 = st.columns(4)
        c_arc1.markdown("<div class='arch-box'><div class='arch-title'>1. Veri Kaynağı</div><div class='arch-desc'>IoT ağlarından toplanan Sentetik Gauss Dağılımlı çevresel sensör okumaları (Sıcaklık, Nem, pH, Su, Gübre).</div></div>", unsafe_allow_html=True)
        c_arc2.markdown("<div class='arch-box'><div class='arch-title'>2. Veri Ön İşleme</div><div class='arch-desc'>Mantıksal özellik çıkarımları (Su/Gübre Oranı) ve algoritmaların tutarlılığı için Min-Max Normalizasyonu işlemleri.</div></div>", unsafe_allow_html=True)
        c_arc3.markdown("<div class='arch-box'><div class='arch-title'>3. Yapay Zeka (AI)</div><div class='arch-desc'>Eğitilmiş Çoklu Algoritma Ağı: Rekolte için <b>Lineer Regresyon</b>, teşhis için <b>SVM & KNN</b>, segmentasyon için <b>K-Means</b>.</div></div>", unsafe_allow_html=True)
        c_arc4.markdown("<div class='arch-box'><div class='arch-title'>4. Karar Destek</div><div class='arch-desc'>Gelen anlık verilerin milisaniyeler içinde işlenerek operasyonel kararlara (Sağlıklı/Riskli teşhisi) dönüştürülmesi.</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # 2. GERÇEK ZAMANLI TOPLU SİMÜLASYON
        # ==========================================
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚡ Gerçek Zamanlı Algoritma Simülatörü</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#475569; margin-bottom:25px;'>Aşağıdaki sensör değerlerini değiştirdiğiniz anda, sistem veriyi anlık olarak işler ve tüm algoritmalar (LR, SVM, KNN, K-Means) yeni duruma göre teşhislerini günceller.</p>", unsafe_allow_html=True)
        
        col_input, col_output = st.columns([1, 2], gap="large")
        
        # Etkileşimli Girdiler
        with col_input:
            st.markdown("<div class='sub-title'>📍 Çevresel Sensör Girdileri</div>", unsafe_allow_html=True)
            s = st.slider("Sıcaklık (°C)", 5.0, 50.0, 25.0, help="Havanın anlık sıcaklığı. İdeal 25°C'den sapmalar bitki stresini artırır.")
            n = st.slider("Nem Oranı (%)", 10.0, 95.0, 50.0, help="Havadaki nem oranı. Düşük nem kuraklık stresi yaratır.")
            ph = st.slider("Toprak pH", 4.0, 10.0, 7.0, help="Toprak asitlik seviyesi. İdeal değer 7.0 civarıdır.")
            su = st.slider("Sulama Hacmi (Litre)", 50, 800, 250, help="Toprağa verilen su miktarı. Rekolteyi doğrudan etkiler.")
            g = st.slider("Gübre Miktarı (KG)", 5, 100, 30, help="Verilen besin miktarı. Suyu dengelemezse kökleri yakabilir.")

        # Arka Plan Hesaplamaları
        yeni_veri = pd.DataFrame([[s, n, ph, su, g, (su/g)]], columns=['Sicaklik_C', 'Nem_Yuzde', 'Toprak_pH', 'Sulama_Litre', 'Gubre_KG', 'Su_Gubre_Orani'])
        veri_scaled = st.session_state.scaler.transform(yeni_veri)
        veri_kmeans_scaled = st.session_state.kmeans_scaler.transform(pd.DataFrame([[ph, n]], columns=['Toprak_pH', 'Nem_Yuzde']))
        
        # Tahminler
        tahmin_rekolte = st.session_state.lr_model.predict(veri_scaled)[0]
        
        t_svm_val = st.session_state.svm_model.predict(veri_scaled)[0]
        t_svm = st.session_state.le.inverse_transform([t_svm_val])[0]
        
        t_knn_val = st.session_state.knn_model.predict(veri_scaled)[0]
        t_knn = st.session_state.le.inverse_transform([t_knn_val])[0]
        
        t_kme = st.session_state.kmeans.predict(veri_kmeans_scaled)[0]

        # Açık Tema Renk Paleti (Daha Belirgin)
        def get_color(durum):
            if str(durum) == "Sağlıklı": return "#059669" # Yeşil
            elif str(durum) == "Riskli": return "#D97706" # Turuncu
            elif str(durum) == "Hastalıklı": return "#DC2626" # Kırmızı
            else: return "#475569" # Gri

        with col_output:
            st.markdown("<div class='sub-title'>🎯 Merkezi Yapay Zeka Karar Paneli</div>", unsafe_allow_html=True)
            
            # Rekolte (Linear Regression)
            fig_bullet = go.Figure(go.Indicator(
                mode = "number+gauge", value = tahmin_rekolte,
                number = {'font': {'color': '#0F172A', 'size': 40, 'family':'Inter'}, 'suffix': " KG"},
                title = {'text': "Beklenen Rekolte<br><span style='font-size:12px;color:#0284C7'>Lineer Regresyon</span>", 'font': {'color': '#475569', 'size': 14}},
                gauge = {
                    'shape': "bullet", 'axis': {'range': [None, 3500], 'tickcolor': "#334155"},
                    'bar': {'color': "#0284C7", 'thickness': 0.8}, 'bgcolor': "#F1F5F9",
                    'steps': [{'range': [0, 1500], 'color': '#E2E8F0'}, {'range': [1500, 3500], 'color': '#F8FAFC'}]
                }
            ))
            fig_bullet = self._apply_theme(fig_bullet)
            fig_bullet.update_layout(height=130, margin=dict(l=150, r=20, t=20, b=20))
            st.plotly_chart(fig_bullet, use_container_width=True)
            
            # Sınıflandırma Modelleri Karşılaştırması
            st.markdown("<p style='color:#475569; font-size:0.9rem; margin-top:10px;'><b>Sınıflandırma & Segmentasyon (Eşzamanlı Teşhis):</b></p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            
            c1.markdown(f"""<div class='pred-card'>
                <div class='pred-title'>SVM (RBF Kernel)</div>
                <div class='pred-value' style='color:{get_color(t_svm)};'>{t_svm}</div>
            </div>""", unsafe_allow_html=True)
            
            c2.markdown(f"""<div class='pred-card'>
                <div class='pred-title'>KNN (Mesafe Tabanlı)</div>
                <div class='pred-value' style='color:{get_color(t_knn)};'>{t_knn}</div>
            </div>""", unsafe_allow_html=True)
            
            c3.markdown(f"""<div class='pred-card'>
                <div class='pred-title'>K-Means (Kümeleme)</div>
                <div class='pred-value' style='color:#0284C7;'>Bölge {t_kme + 1}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # 3. ALGORİTMA KARŞILAŞTIRMALARI VE PERFORMANS
        # ==========================================
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Performans Metrikleri ve Model Doğrulaması</div>", unsafe_allow_html=True)
        
        y_test_reg = st.session_state.y_test_reg
        lr_pred = st.session_state.lr_model.predict(st.session_state.X_test_scaled)
        
        y_test_clf = st.session_state.y_test_clf
        pred_svm = st.session_state.svm_model.predict(st.session_state.X_test_scaled)
        pred_knn = st.session_state.knn_model.predict(st.session_state.X_test_scaled)

        # ---------------- REGRESYON VE SINIFLANDIRMA TABLOLARI YAN YANA ----------------
        col_reg_table, col_clf_table = st.columns([1, 1.4], gap="large")

        with col_reg_table:
            st.markdown("<div class='sub-title'>📈 Regresyon (Lineer) Performansı</div>", unsafe_allow_html=True)
            st.markdown("<p style='color:#475569; margin-bottom:15px; font-size:0.9rem;'>Sezon sonu rekolte tahminindeki başarı oranları ve hata sapmaları.</p>", unsafe_allow_html=True)
            
            html_reg_table = f"""
            <table class="data-table">
                <tr>
                    <th class="text-left" style='width:50%;'>Değerlendirme Metriği</th>
                    <th>Lineer Regresyon</th>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>R² Skoru (Açıklanabilirlik)</b><br><span style='font-size:11px; color:#475569;'>Bağımsız değişkenlerin, rekoltedeki değişimi açıklama gücüdür. %100'e yakınlık modelin güvenilirliğini artırır.</span></td>
                    <td style="color:#0284C7; font-weight:bold; font-size:17px;">% {r2_score(y_test_reg, lr_pred)*100:.1f}</td>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>RMSE (Kök Ortalama Kare Hata)</b><br><span style='font-size:11px; color:#475569;'>Tahminlerin gerçek değerlerden standart sapmasıdır. Büyük hataları daha sert cezalandırır.</span></td>
                    <td style="color:#0F172A; font-weight:bold; font-size:16px;">± {np.sqrt(mean_squared_error(y_test_reg, lr_pred)):.0f} KG</td>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>MAE (Ortalama Mutlak Hata)</b><br><span style='font-size:11px; color:#475569;'>Tahminlerin gerçek değerlerden ortalama sapma miktarını gösterir. Tolerans aralığını belirler.</span></td>
                    <td style="color:#0F172A; font-weight:bold; font-size:16px;">± {mean_absolute_error(y_test_reg, lr_pred):.0f} KG</td>
                </tr>
            </table>
            """
            st.markdown(html_reg_table, unsafe_allow_html=True)

        with col_clf_table:
            st.markdown("<div class='sub-title'>⚔️ Sınıflandırma: SVM vs KNN Karşılaştırması</div>", unsafe_allow_html=True)
            st.markdown("<p style='color:#475569; margin-bottom:15px; font-size:0.9rem;'>Modelin hiç görmediği ayrılmış test verisi (%20) üzerinde algoritmaların teşhis (hastalık tespit) başarısı.</p>", unsafe_allow_html=True)
            
            html_clf_table = f"""
            <table class="data-table">
                <tr>
                    <th class="text-left" style='width:35%;'>Metrik ve Bilimsel Açıklaması</th>
                    <th>SVM (RBF Kernel)</th>
                    <th>KNN (K={st.session_state.k_degeri})</th>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>Accuracy (Doğruluk)</b><br><span style='font-size:11px; color:#475569;'>Tüm tahminlerin toplam veri sayısına bölümüdür. Sistemin genel teşhis gücünü gösterir.</span></td>
                    <td style="color:#0284C7; font-weight:bold; font-size:17px;">% {accuracy_score(y_test_clf, pred_svm)*100:.1f}</td>
                    <td>% {accuracy_score(y_test_clf, pred_knn)*100:.1f}</td>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>Precision (Kesinlik)</b><br><span style='font-size:11px; color:#475569;'>Modelin 'Hasta' dediği vakaların yüzde kaçının gerçekten 'Hasta' olduğunu ölçer. Yanlış alarmı engeller.</span></td>
                    <td style="color:#0284C7; font-weight:bold; font-size:17px;">% {precision_score(y_test_clf, pred_svm, average='weighted', zero_division=0)*100:.1f}</td>
                    <td>% {precision_score(y_test_clf, pred_knn, average='weighted', zero_division=0)*100:.1f}</td>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>Recall (Duyarlılık)</b><br><span style='font-size:11px; color:#475569;'>Gerçekte 'Hasta' olan vakaların ne kadarını başarıyla yakalayabildiğini gösterir. Tehlikeyi kaçırmama oranıdır.</span></td>
                    <td style="color:#0284C7; font-weight:bold; font-size:17px;">% {recall_score(y_test_clf, pred_svm, average='weighted')*100:.1f}</td>
                    <td>% {recall_score(y_test_clf, pred_knn, average='weighted')*100:.1f}</td>
                </tr>
                <tr>
                    <td class="text-left"><b style='color:#0F172A;'>F1-Score</b><br><span style='font-size:11px; color:#475569;'>Kesinlik ve Duyarlılığın harmonik ortalamasıdır. Veri dengesiz olduğunda en güvenilir akademik metriktir.</span></td>
                    <td style="color:#0284C7; font-weight:bold; font-size:17px;">% {f1_score(y_test_clf, pred_svm, average='weighted')*100:.1f}</td>
                    <td>% {f1_score(y_test_clf, pred_knn, average='weighted')*100:.1f}</td>
                </tr>
            </table>
            """
            st.markdown(html_clf_table, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Görsel Analiz: Regresyon Dağılımları, Hata Matrisi ve SVM Karar Sınırları</div>", unsafe_allow_html=True)
        
        c_graf1, c_graf2, c_graf3 = st.columns(3)
        
        with c_graf1:
            fig_scatter = px.scatter(x=y_test_reg, y=lr_pred, color_discrete_sequence=['#0284C7'], title="Lineer Regresyon: Gerçek vs Tahmin")
            fig_scatter.add_shape(type="line", x0=y_test_reg.min(), y0=y_test_reg.min(), x1=y_test_reg.max(), y1=y_test_reg.max(), line=dict(color="#7C3AED", dash="dot"))
            st.plotly_chart(self._apply_theme(fig_scatter), use_container_width=True)
            st.markdown(f"<p style='text-align:center; color:#475569; font-size:0.85rem;'>Lineer Regresyon algoritmasının hedefe yakınlaşma dağılımı.</p>", unsafe_allow_html=True)

        with c_graf2:
            le_classes = st.session_state.le.classes_
            cm_svm = confusion_matrix(y_test_clf, pred_svm, labels=range(len(le_classes)))
            fig_cm = px.imshow(cm_svm, text_auto=True, color_continuous_scale=[[0, '#F8FAFC'], [1, '#0284C7']], x=le_classes, y=le_classes, title="SVM Karmaşıklık (Confusion) Matrisi")
            st.plotly_chart(self._apply_theme(fig_cm), use_container_width=True)
            st.markdown("<p style='text-align:center; color:#475569; font-size:0.85rem;'>Köşegen üzerindeki değerler doğru teşhis edilen kayıtları temsil eder.</p>", unsafe_allow_html=True)

        with c_graf3:
            # Tüm veri seti üzerinden SVM tahminlerini alarak görselleştirme
            svm_all_pred = st.session_state.svm_model.predict(st.session_state.scaler.transform(self.X))
            svm_all_labels = st.session_state.le.inverse_transform(svm_all_pred)
            df_visual = self.df.copy()
            df_visual['SVM_Tahmini'] = svm_all_labels
            
            fig_svm_dist = px.scatter(df_visual, x='Sicaklik_C', y='Nem_Yuzde', color='SVM_Tahmini', 
                                      title="SVM Karar Dağılımı (Sıcaklık vs Nem)",
                                      color_discrete_map={'Sağlıklı':'#10B981', 'Riskli':'#F59E0B', 'Hastalıklı':'#EF4444'})
            st.plotly_chart(self._apply_theme(fig_svm_dist), use_container_width=True)
            st.markdown("<p style='text-align:center; color:#475569; font-size:0.85rem;'>SVM algoritmasının 2D düzlemde sınıfları nasıl izole ettiği gösterilmektedir.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # 4. VERİ SETİ, ÖN İŞLEME VE EDA
        # ==========================================
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🗄️ Veri Merkezi ve İstatistiksel Analiz (EDA)</div>", unsafe_allow_html=True)
        
        c_veri1, c_veri2 = st.columns([1, 1], gap="large")
        with c_veri1:
            st.markdown("<div class='sub-title'>Ham Veri Matrisi (Önizleme)</div>", unsafe_allow_html=True)
            st.dataframe(self.df[['Sicaklik_C', 'Nem_Yuzde', 'Toprak_pH', 'Sulama_Litre', 'Gubre_KG', 'Rekolte_KG', 'Saglik_Durumu']].head(10), height=250, use_container_width=True)
            
        with c_veri2:
            st.markdown("<div class='sub-title'>İstatistiksel Analiz (Pearson Korelasyonu)</div>", unsafe_allow_html=True)
            sayisal_df = self.df[['Sicaklik_C', 'Nem_Yuzde', 'Toprak_pH', 'Sulama_Litre', 'Gubre_KG', 'Rekolte_KG']]
            fig_corr = px.imshow(sayisal_df.corr(), text_auto=".2f", color_continuous_scale='Tealgrn', aspect="auto")
            st.plotly_chart(self._apply_theme(fig_corr), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)