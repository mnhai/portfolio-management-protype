# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-3-myapp2`
- **Tệp chính:** `myapp2.py`
- **Mô tả:** Multi-Asset Equally Weighted Portfolio App (Danh mục đa tài sản 1/N):
  - Hỗ trợ nhập danh sách nhiều mã cổ phiếu linh hoạt (mặc định: `AAPL, MSFT, NVDA`).
  - Tự động chuẩn hóa phân bổ đồng đều trọng số $w_i = 1/N$.
  - Tính toán các chỉ số hiệu suất danh mục tổng thể: Total Return, Annualized Volatility, Annualized Sharpe Ratio, Max Drawdown.
  - Trực quan hóa danh mục:
    - Biểu đồ Donut thể hiện phân bổ trọng số tài sản.
    - Biểu đồ đường so sánh hiệu suất tích lũy của Danh mục so với từng cổ phiếu thành phần.
    - Biểu đồ vùng sụt giảm Drawdown danh mục (Underwater plot).
    - Biểu đồ Histogram tần suất phân phối lợi nhuận ngày.
    - Bảng thống kê chi tiết tỷ trọng, lợi nhuận, biến động từng cổ phiếu.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp2.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
