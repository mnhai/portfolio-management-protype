# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-2-myapp1`
- **Tệp chính:** `myapp1.py`
- **Mô tả:** Personal Trading Analytics Dashboard:
  - Tải dữ liệu Yahoo Finance (`yfinance`) linh hoạt theo Ticker và khoảng thời gian.
  - Tính toán chỉ báo kỹ thuật: MA5, MA20, MA50, Daily Returns.
  - Các chỉ số hiệu suất chuẩn định lượng: Total Return, Annualized Volatility, Annualized Sharpe Ratio, Max Drawdown.
  - Tín hiệu giao dịch MA5/MA20 crossover (BUY/SELL).
  - Trực quan hóa tương tác: Biểu đồ giá + Moving Averages, Biểu đồ Drawdown (vùng chìm), Biểu đồ phân phối lợi nhuận (Histogram).

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp1.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
