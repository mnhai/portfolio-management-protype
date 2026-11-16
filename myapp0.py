import streamlit as st

st.set_page_config(
    page_title="Portfolio Management App - Bước 1",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Management App (Bước 1 — Hello World)")
st.write("Chào mừng đến với hệ thống Portfolio Management App! Ứng dụng Streamlit đang hoạt động bình thường.")
st.success("✅ Môi trường Streamlit và các thư viện cốt lõi đã sẵn sàng cho các bước phân tích tiếp theo.")

# Kiểm tra cơ bản
user_input = st.text_input("Nhập mã cổ phiếu thử nghiệm (vd: AAPL, SPY):", "SPY")
if user_input:
    st.write(f"Bạn vừa nhập: **{user_input.upper()}**")
