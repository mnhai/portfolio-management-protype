# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-5-myapp7`
- **Tệp chính:** `myapp7.py`
- **Mô tả:** Enhanced Risk Metrics: Rolling Beta vs SPY:
  - Bổ sung chỉ số rủi ro nâng cao **Rolling Beta (20 ngày)** đo lường độ nhạy cảm của danh mục so với thị trường:
    $$\beta_{20, t} = \frac{\text{Cov}_{20}(R_{p,t}, R_{\text{SPY},t})}{\text{Var}_{20}(R_{\text{SPY},t})}$$
  - Tính toán **Rolling Alpha (CAPM adjustment)** trong data pipeline.
  - Hiển thị Latest Beta và phân tích ý nghĩa định lượng tự động (ví dụ: $\beta = 1.49 \rightarrow$ nhạy cảm hơn thị trường ~49%, mang tính đòn bẩy thay vì phòng hộ).
  - Biểu đồ đường Rolling Beta theo thời gian với đường tham chiếu chuẩn thị trường $\text{Market Beta} = 1.0$.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp7.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
