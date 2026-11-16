# Portfolio Management App 📈

Dự án xây dựng hệ thống quản lý danh mục và phân tích đầu tư định lượng chuẩn Bloomberg Terminal theo tiến trình 9 bước từ bài báo.

## 🌿 Nhánh hiện tại: `step-7-myapp9`
- **Tệp chính:** `myapp9.py`
- **Mô tả:** Max Sharpe Portfolio Optimization Engine:
  - Ứng dụng giải thuật tối ưu hóa phi tuyến **SLSQP (`scipy.optimize.minimize`)** để tìm vector trọng số $\mathbf{w}^*$ tối đa hóa Sharpe Ratio:
    $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le w_{\max}$$
  - Ràng buộc thực tế: **No Shorting** ($w_i \ge 0$) và **Giới hạn tỷ trọng tối đa** ($w_i \le \text{Max Weight}$).
  - Bảng so sánh 3 danh mục trực quan: **User Defined Portfolio** vs **Max Sharpe Optimized Portfolio** vs **Benchmark (SPY)**.
  - Biểu đồ Bar so sánh trọng số và Biểu đồ đường tích lũy 3 danh mục.

## 🚀 Hướng dẫn chạy
```bash
streamlit run myapp9.py
```

## 🧪 Chạy kiểm thử
```bash
pytest tests/ -v
```
