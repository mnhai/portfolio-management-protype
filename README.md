# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-8-myapp10`
- **Tệp chính:** `myapp10.py`
- **Mô tả:** Bull/Bear Regime Switching Strategy:
  - Cỗ máy nhận diện chế độ thị trường **Market Regime Engine** phân tích chỉ số chuẩn `SPY`:
    * **Trend (0/1):** Xu hướng MA50 so với MA200 (Golden Cross).
    * **Momentum (0/1):** Lợi nhuận 12 ngày dương/âm ($P(t)/P(t-12)-1 > 0$).
    * **Volatility (0/1):** Biến động 20 ngày so với trung vị 60 ngày (Risk-off hurdle).
  - Công thức tổng hợp: $\text{Regime Score} = \text{Trend} + \text{Momentum} - \text{Volatility}$
  - Phân loại chế độ:
    * $\text{Score} \ge 1$: **BULL** (Tăng trưởng - Màu xanh)
    * $\text{Score} = 0$: **NEUTRAL** (Trung lập - Màu vàng)
    * $\text{Score} \le -1$: **BEAR** (Phòng thủ - Màu đỏ)
  - **Chiến lược phân bổ tự động (Regime Switching Strategy):**
    * Khi BULL: 100% Growth Portfolio.
    * Khi NEUTRAL: 50% Growth + 50% Defensive (SPY).
    * Khi BEAR: 100% Defensive (SPY/Cash).
  - Trực quan hóa bản đồ chế độ thị trường **SPY Price Regime Map** phân đoạn màu sắc theo từng phiên.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp10.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
