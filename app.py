import streamlit as st
import warnings

from core.veri_motoru import TarimVeriMotoru
from arayuz.sayfalar import ArayuzYoneticisi

warnings.filterwarnings('ignore')

# Profesyonel SaaS Sayfa Yapılandırması
st.set_page_config(
    page_title="REVAN AI | Akıllı Tarım Platformu", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="collapsed" # Geniş bir görünüm için sol menü kapalı başlar
)

# Backend Veri Motoru Başlatma (Singleton)
if 'veri_motoru' not in st.session_state:
    st.session_state.veri_motoru = TarimVeriMotoru(ilk_kayit_sayisi=1000)

arayuz = ArayuzYoneticisi(st.session_state.veri_motoru)

def main():
    # Sol Menü Artık Sadece Platform Bilgisi ve Ayarlar İçin
    with st.sidebar:
        st.markdown("<div style='text-align: center;'><h1 style='color: #06B6D4; letter-spacing: 2px;'>REVAN AI</h1><p style='color: #8E9BAE; font-size: 12px; margin-top:-15px;'>Tarım Karar Destek Sistemi</p></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #2A313E;'>", unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Sistem Yönetimi")
        if st.button("🔄 Veri Ağını Sıfırla", use_container_width=True):
            st.session_state.veri_motoru.sifirla(1000)
            if 'sistem_hazir' in st.session_state:
                del st.session_state['sistem_hazir']
            st.rerun()
            
        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        st.caption("Platform v2.1.0\nSecure Environment")

    # Tüm içerik tek sayfada yukarıdan aşağıya akacak şekilde render edilir
    arayuz.css_enjekte_et()
    arayuz.render_dashboard()

if __name__ == "__main__":
    main()