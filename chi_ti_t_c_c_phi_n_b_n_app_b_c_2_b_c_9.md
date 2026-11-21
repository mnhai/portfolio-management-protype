# Nội dung chi tiết từng phiên bản app (Bước 2 → Bước 9)
*(Bám sát đúng code và các ảnh chụp giao diện trong file PDF bài báo)*

---

## Bước 2 — `myapp1.py`: Personal Trading Analytics App
* **Bố cục:** Không có tab — toàn bộ nội dung nằm trên 1 trang.
* **Sidebar:** 
  * `Ticker` (mặc định `"AAPL"`)
  * `Start Date`
* **Nội dung chính:**
  * **Tiêu đề:** `"📈 Personal Trading Analytics App"` / **Subheader:** `"Portfolio + Signal Lab"`
  * **3 metric (cột):** `Total Return`, `Annual Volatility`, `Sharpe Ratio`
  * **Price Chart (biểu đồ đường):** `Close`, `MA20`, `MA50` (3 đường)
  * **Returns Distribution:** Histogram lợi nhuận hàng ngày (`50 bins`)
  * **Drawdown Analysis:** Area chart drawdown
  * **Signal Lab:** 
    * Metric `"Current Signal"` (`BUY`/`SELL` dựa vào `MA20 > MA50`)
    * Bảng dataframe 20 dòng cuối (`Close`, `High`, `Low`, `Open`, `Volume`, `MA20`, `MA50`, `Returns`, `Signal`)

---

## Bước 3 — `myapp2.py`: Multi-Asset Equally Weighted Portfolio App
* **Bố cục:** Không có tab — 1 trang duy nhất.
* **Sidebar:** 
  * `Tickers` (text, phân cách bằng dấu phẩy, mặc định `"AAPL,MSFT,NVDA"`)
  * `Start Date`
* **Nội dung chính:**
  * **Tiêu đề:** `"📈 Personal Trading Analytics App"` / **Subheader:** `"Portfolio Mode + Signal Lab"`
  * **3 metric (cột):** `Portfolio Return`, `Volatility`, `Sharpe Ratio`
  * **Portfolio Allocation:** Pie chart (tỷ trọng đều giữa các mã)
  * **Portfolio Performance:** Line chart tăng trưởng lũy kế danh mục
  * **Drawdown Analysis:** Area chart drawdown
  * **Returns Distribution:** Histogram lợi nhuận danh mục
  * **Signal Lab (Simplified):** 
    * Metric `"Portfolio Signal"` (`BUY`/`SELL` từ giao cắt `MA5` vs `MA20` của lợi nhuận danh mục)
    * Bảng 20 dòng cuối (`Portfolio Return`, `Signal`)

---

## Bước 4 — `myapp5.py`: Interactive Portfolio App with User-Defined Weights
*(Bản đầu tiên của "mini Bloomberg terminal")*

* **Sidebar:** `Tickers`, `Start Date`, `Weights` — mỗi mã có 1 slider trọng số riêng
* **Tiêu đề:** `"📈 Bloomberg-Style Portfolio Terminal"`
* **4 Tab:** `📊 Portfolio` | `⚠️ Risk` | `🚦 Signals` | `🔄 Backtest`

| Tab | Nội dung |
| :--- | :--- |
| **Portfolio** | Subheader `"Portfolio Engine"` → **4 metric:** `Portfolio Return`, `SPY Return`, `Alpha vs SPY`, `Sharpe` → Pie chart `Allocation` → Line chart `Portfolio vs SPY Benchmark` |
| **Risk** | Subheader `"Risk Engine"` → Heatmap `Correlation Matrix` → Area/line chart `Drawdown Comparison (Portfolio vs SPY)` → Line chart `Rolling Volatility (20D Annualized) (Portfolio vs SPY)` *(3 biểu đồ)* |
| **Signals** | Subheader `"Signal Engine"` → Metric `"Portfolio Signal"` → Line chart tín hiệu trung bình → Dataframe 20 dòng cuối (tín hiệu từng mã) |
| **Backtest** | Subheader `"Backtest Engine"` → Dataframe 10 dòng cuối (`Portfolio vs SPY`) → Line chart `Portfolio vs SPY` → Histogram lợi nhuận danh mục |

* **Kết quả ví dụ hiển thị:** Portfolio Return `576.49%`, SPY Return `105.07%`, Alpha `471.42%`, Sharpe `1.87`.

---

## Bước 5 — `myapp7.py`: Thêm Rolling Beta Engine
*(Dựa trên `myapp5`, chỉ thêm phần Beta)*

| Tab | Thay đổi |
| :--- | :--- |
| **Portfolio** | Thêm metric thứ 5: `"Rolling Beta (SPY)"` (giá trị beta mới nhất, ví dụ `1.49`) |
| **Risk** | Thêm biểu đồ thứ 4: `"Rolling Beta vs SPY (20D)"` — line chart + đường ngang nét đứt tại `y=1.0` làm mốc tham chiếu |
| **Signals, Backtest** | Không đổi |

> *Ghi chú: `Rolling Alpha` cũng được tính trong code nhưng tác giả ghi rõ "chưa hiển thị trên dashboard".*

---

## Bước 6 — `myapp8.py`: Thêm Momentum Exposure Factor Panel

| Tab | Thay đổi |
| :--- | :--- |
| **Portfolio** | Thêm 2 metric: `"Market Beta"`, `"Momentum Exposure"` → **tổng 7 metric:** 4 cột (`Portfolio Return`, `SPY Return`, `Alpha`, `Sharpe`) + `Rolling Beta (SPY)`, `Market Beta`, `Momentum Exposure` |
| **Risk** | Thêm biểu đồ thứ 5: `"Momentum Factor Exposure (Rolling)"` — line chart tương quan rolling giữa danh mục và yếu tố momentum tổng hợp từ SPY, có đường ngang nét đứt tại `y=0.0` |
| **Signals, Backtest** | Không đổi |

> *→ Tab Risk lúc này có đủ 5 biểu đồ: Correlation Matrix, Drawdown Comparison, Rolling Volatility, Rolling Beta, Momentum Exposure.*

---

## Bước 7 — `myapp9.py`: Thêm Max Sharpe Portfolio Optimization Engine (`scipy`)

| Tab | Thay đổi |
| :--- | :--- |
| **Portfolio** | • Thêm 3 metric: `"Optimized Sharpe"`, `"Optimized Return"`, `"Optimized Volatility"` → **tổng 10 metric**<br>• Thêm nội dung mới: Subheader `"Optimized Portfolio (Max Sharpe)"` → Pie chart `Optimized Allocation` (trọng số từ bộ tối ưu hóa) → Line chart `"Optimized vs Current vs SPY"` (3 đường: `Current Portfolio`, `Max Sharpe Portfolio`, `SPY`) |
| **Risk, Signals, Backtest** | Không đổi (giống `myapp8`) |

* **Kết quả ví dụ:** Optimized Sharpe `1.90`, Optimized Return `59.51%`, Optimized Volatility `31.29%`.
* *(Lưu ý: Ở bước này chưa có scatter `"Individual Assets"` và chưa có `Efficient Frontier` — 2 thành phần này chỉ xuất hiện ở bản cuối `myapp13`).*

---

## Bước 8 — `myapp10.py`: Thêm Bull/Bear Regime Switching Strategy

| Tab | Thay đổi |
| :--- | :--- |
| **Portfolio, Risk, Signals** | Không đổi (giữ nguyên `myapp9`) |
| **Backtest** | Giữ nguyên nội dung cũ, thêm vào cuối:<br>• Subheader `"Regime Switching Strategy"` → Line chart cùng tên (3 đường: `Regime Strategy`, `Static Portfolio`, `SPY`) → Dataframe `Regime` (`Date`/`Regime`: `BULL`/`BEAR`/`NEUTRAL`) 20 dòng cuối<br>• Subheader `"Market Regime"` → Line chart `Regime Score` (điểm số thô theo thời gian) → Dataframe `Regime` 20 dòng cuối |

* **Quy tắc tính điểm:** `Regime Score = trend (MA50 > MA200 của SPY) + momentum (lợi nhuận 12 ngày của SPY > 0) − volatility (vol cao = risk-off)`.

---

## Bước 9 — `myapp13.py`: Final Bloomberg-Style Portfolio Terminal (5 tab)
*(Bản hoàn chỉnh nhất — thêm tab Factors + nâng cấp Tab 1 và Tab 4)*

* **5 Tab:** `📊 Portfolio` | `⚠️ Risk` | `🚦 Signals` | `🔄 Backtest` | `📑 Factors`

### Tab 1 — Portfolio (13 metric + 5 biểu đồ)
* **Metrics:**
  * **4 cột:** `Portfolio Return`, `SPY Return`, `Alpha vs SPY`, `Sharpe`
  * **Hàng 2:** `Rolling Beta (SPY)`, `Market Beta`, `Momentum Exposure`
  * **Hàng 3:** `Optimized Sharpe`, `Optimized Return`, `Optimized Volatility`
  * **Hàng 4:** `CAPM Alpha`, `Treynor Ratio`, `Regime` (`RISK ON` / `RISK OFF` — tính riêng từ SPY, QQQ, VIX)
* **Biểu đồ:**
  1. Pie chart `Allocation`
  2. Subheader `"Optimized Portfolio (Max Sharpe)"` → Pie chart `Optimized Allocation`
  3. Line chart `Portfolio vs SPY Benchmark`
  4. Line chart `"Optimized vs Current vs SPY"` — nâng cấp có thêm 4 trace: `Current Portfolio`, `Max Sharpe Portfolio`, `SPY`, + scatter `"Individual Assets"` (marker hình thoi, text = tên ticker, vị trí theo Volatility/Return từng tài sản)
  5. Scatter chart `"Monte Carlo Efficient Frontier"` — 5.000 danh mục ngẫu nhiên (`x=Volatility`, `y=Return`, màu theo `Sharpe`) + marker riêng `Individual Assets` (hình thoi đen) + marker `Max Sharpe` (ngôi sao đỏ, cỡ lớn)

### Tab 2 — Risk *(không đổi, 5 biểu đồ)*
* `Correlation Matrix` (heatmap) → `Drawdown Comparison` → `Rolling Volatility (20D)` → `Rolling Beta vs SPY (20D)` → `Momentum Factor Exposure (Rolling)`

### Tab 3 — Signals *(không đổi)*
* Metric `Portfolio Signal` + Line chart tín hiệu + Dataframe 20 dòng cuối

### Tab 4 — Backtest *(giữ nguyên + thêm 1 biểu đồ mới)*
* Dataframe 10 dòng cuối (`Portfolio vs SPY`)
* Line chart `Portfolio vs SPY` + Histogram lợi nhuận
* Line chart `Regime Switching Strategy` (3 đường) + Dataframe `Regime`
* Line chart `Regime Score` + Dataframe `Regime`
* **Mới:** Scatter chart `"SPY Price Regime Map (Bull/Neutral/Bear)"` — màu theo chế độ thị trường (xanh lá = `BULL`, cam = `NEUTRAL`, đỏ = `BEAR`), trục `x = Date`, `y = SPY Price`

### Tab 5 — Factors *(hoàn toàn mới)*
1. **Subheader:** `"Factor Analytics"`
2. **Hàng metric 1 (3 cột):** `SPY Beta`, `QQQ Beta`, `IWM Beta`
3. **Hàng metric 2 (3 cột):** `VIX Beta`, `CAPM Alpha`, `Treynor Ratio`
4. **Metric đơn:** `Market Regime` (`RISK ON`/`RISK OFF`)
5. **Factor Exposures:** Bar chart `"Portfolio Factor Betas"` (4 cột: `SPY`/`QQQ`/`IWM`/`VIX`, có label số trên đầu cột)
6. **Rolling Beta Heatmap vs SPY:** Heatmap `"60-Day Rolling Beta"` — hàng là từng ticker, cột là thời gian (cửa sổ rolling 60 ngày, khác với 20 ngày ở Tab Risk)
7. **Active Return Attribution vs SPY:** Line chart `"Portfolio Outperformance vs SPY"` — `Cumulative Active Return = tích lũy (1 + (lợi nhuận danh mục − lợi nhuận SPY))`
8. **Factor Summary:** Bảng dataframe (`Factor`, `Beta`) cho `SPY`/`QQQ`/`IWM`/`VIX`
9. **Top/Bottom Beta Tickers:** 2 cột song song — `"Highest Beta Names"` (top 5 beta cao nhất) và `"Lowest Beta Names"` (top 5 beta thấp nhất), tính từ ma trận beta 60 ngày

* **Kết quả ví dụ cuối:** Portfolio Return `180%`, SPY Return `105.07%`, Alpha `74.92%`, Sharpe `1.45`, CAPM Alpha `6.21%`, Treynor Ratio `0.27`, Regime: `RISK OFF`.