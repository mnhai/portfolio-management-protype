# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-6-myapp8`
- **Tệp chính:** `myapp8.py`
- **Mô tả:** Momentum Exposure Factor Panel:
  - Xây dựng **Synthetic Momentum Factor** từ chỉ số chuẩn `SPY` theo công thức quỹ phòng hộ:
    $$\text{Momentum}(t) = \frac{P(t)}{P(t-12)} - 1$$
  - Đo lường mức độ phơi nhiễm nhân tố (Factor Exposure) thông qua hệ số tương quan Rolling 20 ngày giữa Danh mục và Yếu tố Momentum.
  - Tính toán **Factor Beta** cho nhân tố Momentum: $\beta_{\text{mom}} = \frac{\text{Cov}(R_p, \text{Factor}_{\text{mom}})}{\text{Var}(\text{Factor}_{\text{mom}})}$.
  - Bổ sung **Tab 5: Momentum Factors** với nhận định chiến lược phân loại tự động:
    * **Momentum Rider** (Bắt đà xu hướng, corr > 0.3)
    * **Contrarian** (Đi ngược đà, corr < -0.3)
    * **Momentum Neutral** (Trung lập với đà thị trường)
  - Biểu đồ đường Rolling Momentum Correlation và Biểu đồ so sánh 12-day Momentum giữa Portfolio vs SPY.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp8.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
