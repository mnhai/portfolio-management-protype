# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-4-myapp5`
- **Tệp chính:** `myapp5.py`
- **Mô tả:** Mini Bloomberg Terminal với Trọng số Tùy chỉnh (Interactive Weights) & 4 Tab chuyên sâu:
  - Cho phép người dùng tùy biến trọng số qua Sliders và tự động chuẩn hóa $\sum w_i = 100\%$.
  - Tải tự động Benchmark chuẩn `SPY` để đối chuẩn hiệu suất.
  - **Tab 1: Portfolio** — Lợi nhuận danh mục, Lợi nhuận SPY, Alpha vượt trội ($R_p - R_{\text{SPY}}$), Sharpe Ratio, biểu đồ phân bổ Donut, đường hiệu suất lũy kế so với SPY.
  - **Tab 2: Risk** — Ma trận tương quan (Correlation Heatmap), biểu đồ so sánh mức sụt giảm Drawdown danh mục vs SPY, bảng đối chiếu Volatility & Max Drawdown.
  - **Tab 3: Signals** — Tín hiệu giao dịch MA5/MA20 (BUY/SELL) và biểu đồ xu hướng kỹ thuật cho từng mã cổ phiếu.
  - **Tab 4: Backtest** — Thống kê lịch sử: Tỷ lệ ngày thắng (Win Rate), phiên giao dịch tốt nhất/kém nhất, biểu đồ histogram so sánh phân phối lợi nhuận ngày của Portfolio vs SPY.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp5.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
