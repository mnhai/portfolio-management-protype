# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-9-myapp13` (FINAL RELEASE)
- **Tệp chính:** `myapp13.py`
- **Mô tả:** Final Bloomberg-Style Quantitative Portfolio Terminal hoàn chỉnh gồm 5 Tab chuyên nghiệp:
  1. **Tab 1: Portfolio & Efficient Frontier**
     - Hiệu suất danh mục, phân bổ trọng số, tối ưu hóa Max Sharpe (SLSQP).
     - Biểu đồ **Monte-Carlo Efficient Frontier** (mô phỏng 1,500 - 3,000 danh mục) highlight sao đỏ **Max Sharpe Portfolio** và kim cương vàng **Minimum Volatility Portfolio**.
     - So sánh đường tích lũy 3 danh mục (User vs Optimized vs SPY).
  2. **Tab 2: Risk & Rolling Beta Heatmap**
     - Ma trận tương quan toàn diện.
     - So sánh Drawdown Portfolio vs SPY.
     - **Rolling Beta Heatmap** trực quan hóa biến thiên độ nhạy của từng cổ phiếu qua các chu kỳ thị trường.
  3. **Tab 3: Trading Signals**
     - Chỉ báo mua/bán (MA5/MA20 crossover) và trạng thái kỹ thuật cho từng mã.
  4. **Tab 4: Backtest Performance**
     - Thống kê tỷ lệ thắng (Win Rate), Histogram phân phối lợi nhuận ngày so sánh dạng overlay.
  5. **Tab 5: Factors, CAPM & Active Attribution**
     - **CAPM Alpha:** $\alpha = R_p - [R_f + \beta (R_{\text{SPY}} - R_f)]$
     - **Treynor Ratio:** $\text{Treynor} = \frac{R_p - R_f}{\beta}$
     - **Active Return Attribution vs SPY:** Phân tích đóng góp lợi nhuận chủ động từng mã:
       $$\text{Contribution}_i = w_i \times (R_i - R_{\text{SPY}})$$
     - Synthetic Momentum Factor Correlation & Factor Beta.
  - **Top Bloomberg Ticker Tape:** Băng hiển thị chế độ thị trường **Market Regime Monitor** (`BULL` / `NEUTRAL` / `BEAR`), Regime Score, Beta và Alpha.

## 🚀 Hướng dẫn chạy ứng dụng cuối cùng
```bash
streamlit run myapp13.py
```

## 🧪 Chạy toàn bộ 35 unit tests
```bash
pytest tests/ -v
```
