# Model card: churn + CLV (v1.0.0)

**Mục đích.** Cho mỗi khách đang hoạt động (có đơn hợp lệ trong 3 năm qua): xác suất không mua trong 12 tháng tới (`p_churn`), lợi nhuận kỳ vọng nếu khách quay lại (`clv`), phân khúc, và có đáng phát voucher không (`net`, `s_star`). Dùng để ưu tiên chăm sóc khách; **không** dùng để tự động phát giảm giá hàng loạt.

**Dữ liệu.** DATATHON 2026 (e-commerce thời trang, dữ liệu tự sinh, 2012–2022). Train mốc cắt 2019-01/2019-07/2020-01, hiệu chỉnh và chọn model trên 2021-01, test 2022-01 (xem một lần). Đơn vị tiền không xác định, chỉ so sánh tương đối.

**Model.**
- Churn: LightGBM 12 biến, hiệu chỉnh sigmoid. Test: PR-AUC 0,843, ROC-AUC 0,755, Brier 0,1865, ECE 0,004 (tỷ lệ churn nền 64,8%). Top 20% rủi ro bắt 28% số churn.
- CLV: hurdle Negative Binomial: `E[số đơn | quay lại] × lợi nhuận giá gốc TB mỗi đơn theo danh mục × scale (0,775)`. Test: Spearman số đơn 0,336, Spearman CLV 0,127, tổng dự đoán/thực tế 0,73 (lợi nhuận mỗi đơn 2022 cao hơn). CLV chỉ xếp hạng yếu; đừng dùng số tiền tuyệt đối.
- Phân khúc: `p_churn` ≥ `p_cut` (0,6745, TB validation) và `clv` ≥ 12% × giá trị đơn TB (chi phí một voucher).

**Cách dùng.** `predict.score(clean_tables(tables), cutoff)` trả `p_churn, clv, voucher_cost, segment, net, s_star, top_reasons`. `net` = `p×0,30×(clv − voucher_cost) − (1−p)×voucher_cost` (dạng từng khách của công thức Neslin et al. 2006, kịch bản tỷ lệ cứu 30%); `s_star` = `(1−p)×voucher_cost / (p×(clv − voucher_cost))` là tỷ lệ cứu cần để hòa vốn (vô cùng nếu `clv ≤ voucher_cost`). Chỉ cân nhắc voucher khi `s_star` thấp (< 0,3) và sau khi đo được tỷ lệ cứu thật bằng nhóm đối chứng.

**Giới hạn.**
- Voucher % lỗ ở dữ liệu này (biên gộp ~11% so với giảm 12%); phát đại trà lỗ ở mọi tỷ lệ cứu đến 100%; 15,4% khách có `s_star` ≤ 30%. Xếp hạng theo `p_churn` hay `p×CLV` mà không tính chi phí vẫn lỗ (thêm CLV còn tệ hơn chỉ dùng `p_churn`); chỉ quy tắc `net > 0` có lãi (Task 10).
- Tỷ lệ cứu chưa đo được; `net` chỉ là kịch bản.
- `frequency` gánh gần hết độ chính xác; feature hành vi gần như nhiễu. Khách rất gắn bó mà bỏ đi không phát hiện được.
- `cancel_rate` nghi rò rỉ nhẹ, tác động trái trực giác, bị ẩn khỏi `top_reasons`. Bản model cho web (Giai đoạn B) sẽ bỏ biến này.
- Phân phối đổi từ 2019; cần chấm lại và theo dõi metric theo năm.
- Test 2022-01 đã bị tính nhiều hơn một lần (Task 6, 9), xem ghi chú trong kế hoạch.

## Bản model cho web (`churn_web.joblib`, `clv_web.joblib`, `metadata_web.json`)

Build: `python src/predict.py build web`; chấm: `predict.score(web_adapter.load_tables(db), cutoff, web=True)`.

**Khác bản chính.** Chỉ dùng feature tính được từ `users` + `orders` của Web: churn gồm `recency_days, frequency, monetary, mean_gap_days, std_gap_days, n_categories, main_category`; CLV bỏ `cancel_rate`, review, trả hàng, giao hàng, trả góp, kênh, vùng. Cùng tham số đóng băng; hiệu chỉnh sigmoid và `scale` CLV (0,7745) fit lại trên validation. Web không có giá vốn: `cogs = giá bán × trung vị(cogs/price) theo danh mục` (danh mục lạ dùng trung vị chung).

**Chênh metric trên test 2022-01** (cùng hàm `predict._test_metrics` cho cả hai bản):

| Metric | Bản chính | Bản web |
| :--- | ---: | ---: |
| Churn ROC-AUC | 0,7549 | 0,7533 |
| Churn PR-AUC | 0,8428 | 0,8413 |
| Churn Brier | 0,1865 | 0,1871 |
| CLV Spearman số đơn | 0,3355 | 0,3348 |
| CLV Spearman CLV | 0,1272 | 0,1252 |
| Tổng dự đoán / thực tế | 0,731 | 0,731 |

Chênh rất nhỏ vì `frequency` gánh gần hết độ chính xác. (Số bản chính ở bảng này tính lại bằng `_test_metrics`, hơi khác con số ở phần Model do cách đo khác một chút.) Test 2022-01 được xem thêm một lần cho bản web.

**Giới hạn riêng bản web.** `monetary` của Web chưa trừ tiền hoàn (Web chưa có dữ liệu trả hàng); danh mục thật của Web (nội thất) khác dataset (thời trang) nên `main_category` lạ rơi về mức trung bình; dữ liệu demo là DATATHON nên điểm số chỉ để minh họa; các số lấy từ dataset nên chưa kiểm chứng trên khách thật của Web.

