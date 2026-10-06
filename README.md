# 🧠 Dự Đoán Nguy Cơ Đột Quỵ (Stroke Prediction)

Dự án áp dụng quy trình **Khoa học Dữ liệu (Data Science)** và **Học máy (Machine Learning)** hoàn chỉnh nhằm dự đoán nguy cơ đột quỵ ở bệnh nhân dựa trên các thông số nhân khẩu học và chỉ số sức khỏe lâm sàng.

---

## 📌 1. Giới thiệu bài toán

Đột quỵ (Stroke) là một trong những nguyên nhân gây tử vong và tàn tật hàng đầu trên toàn cầu. Việc phát hiện sớm các yếu tố nguy cơ có ý nghĩa sống còn trong công tác phòng ngừa và can thiệp y tế kịp thời.

Mục tiêu của dự án:
- Phân tích khám phá dữ liệu (EDA) để xác định các đặc trưng có mối tương quan mạnh nhất với nguy cơ đột quỵ.
- Xây dựng quy trình tiền xử lý dữ liệu chuẩn hóa, xử lý ngoại lai (Outliers) và mất cân bằng dữ liệu nghiêm trọng.
- Thử nghiệm, so sánh nhiều thuật toán phân loại và tối ưu hóa ngưỡng quyết định (**Threshold Tuning**) để tối đa hóa độ nhạy (**Recall**) - ưu tiên không bỏ sót bệnh nhân.

---

## 🗂️ 2. Cấu trúc thư mục

```plaintext
STROKE_PREDITION/
├── 1.Tiền xử lí dữ liệu/
│   ├── File dữ liệu thô.csv                    # Dữ liệu gốc
│   ├── Tiền xử lý dữ liệu (1).ipynb            # Notebook phân tích và làm sạch dữ liệu
│   └── README.md                               # Hướng dẫn chi tiết bước tiền xử lý
├── 2. Xây dựng mô hình dự đoán/
│   ├── Dữ liệu đã xử lí.csv                    # Dữ liệu sau khi làm sạch hoàn chỉnh
│   ├── Dữ liệu sau tiền xử lí train.csv        # Tập train (đã qua xử lý)
│   ├── Dữ liệu sau tiền xử lí test.csv         # Tập test (đã qua xử lý)
│   ├── build_model.ipynb                       # Notebook huấn luyện mô hình cơ sở
│   └── test7.ipynb                             # [FINAL] Pipeline huấn luyện, so sánh & tối ưu mô hình
├── KiemThu_SauCung.ipynb                       # Script kiểm thử tự động quy trình và dữ liệu
├── .gitignore                                  # Các file tạm và checkpoint bỏ qua
└── README.md                                   # Tài liệu tổng quan dự án
```

---

## 🔬 3. Quy trình thực hiện

### 3.1. Tiền xử lý dữ liệu & Khám phá đặc trưng (EDA)
- **Làm sạch:** Loại bỏ trường `id` (không mang giá trị thống kê), xử lý giá trị khuyết thiếu (`bmi`) bằng phương pháp nội suy trung vị / trung bình nhóm.
- **Xử lý ngoại lai (Outliers):** Sử dụng phương pháp **IQR (Interquartile Range)** kết hợp kỹ thuật `np.clip()` nhằm hạn chế ảnh hưởng của các giá trị dị biệt nhưng vẫn giữ lại thông tin bệnh án.
- **Mã hóa & Chuẩn hóa:** 
  - One-Hot Encoding cho các biến phân loại danh mục (`gender`, `work_type`, `Residence_type`, `smoking_status`, `ever_married`).
  - `StandardScaler` cho các đặc trưng số liên tục (`age`, `avg_glucose_level`, `bmi`).

### 3.2. Xử lý mất cân bằng dữ liệu (Class Imbalance)
- Dữ liệu đột quỵ có tỷ lệ mất cân bằng cao (số ca mắc bệnh chiếm thiểu số).
- Áp dụng kỹ thuật điều chỉnh trọng số lớp (`class_weight='balanced'`) giúp mô hình tăng cường độ nhạy với lớp thiểu số (Stroke = 1).

### 3.3. So sánh mô hình & Lựa chọn
Đã tiến hành so sánh nhiều thuật toán phân loại phổ biến:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

### 3.4. Tinh chỉnh ngưỡng quyết định (Threshold Optimization)
Trong bài toán y tế, **Recall** là chỉ số quan trọng hàng đầu (tránh bỏ sót ca bệnh nguy hiểm). 
- Với ngưỡng mặc định `0.5`, mô hình bắt được phần lớn ca bệnh nhưng độ chính xác (`Precision`) thấp do phát sinh nhiều báo động giả (False Positives).
- Qua phân tích đường cong Precision-Recall Trade-off, ngưỡng quyết định tối ưu được chọn là **`0.73`**, giúp cân bằng hoàn hảo giữa việc duy trì Recall cao và tăng vọt độ chính xác cũng như F1-Score.

---

## 📊 4. Kết quả đánh giá trên tập kiểm thử (Final Test Results)

Đánh giá mô hình hoàn thiện (**Logistic Regression with Threshold = 0.73**) trên tập dữ liệu kiểm thử độc lập (**Test Set**):

| Chỉ số đánh giá | Kết quả | Ý nghĩa y khoa |
| :--- | :---: | :--- |
| **Accuracy** | **88%** | Tỷ lệ dự đoán đúng tổng thể trên toàn bộ tập dữ liệu |
| **Recall (Độ nhạy)** | **70.27%** | Phát hiện thành công hơn **70%** số bệnh nhân thực sự có nguy cơ đột quỵ |
| **ROC-AUC** | **0.8279** (~83%) | Khả năng phân biệt xuất sắc giữa nhóm có nguy cơ và không có nguy cơ |
| **Precision** | **24.07%** | Tỷ lệ bệnh nhân được cảnh báo thực sự mắc bệnh |
| **F1-Score** | **0.3586** | Điểm cân hòa tối ưu giữa Recall và Precision |

---

## 🚀 5. Hướng dẫn cài đặt & Chạy dự án

### Yêu cầu môi trường
- Python >= 3.8
- Jupyter Notebook hoặc VS Code / JupyterLab

### Cài đặt thư viện
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Chạy kiểm thử & Huấn luyện
1. Mở file `1.Tiền xử lí dữ liệu/Tiền xử lý dữ liệu (1).ipynb` để chạy quy trình làm sạch dữ liệu.
2. Mở file `2. Xây dựng mô hình dự đoán/test7.ipynb` để theo dõi toàn bộ pipeline huấn luyện, tinh chỉnh ngưỡng và đánh giá mô hình cuối cùng.
3. Mở file `KiemThu_SauCung.ipynb` để chạy kiểm tra kiểm thử logic của mô hình.

---

## 👥 Tác giả & Đóng góp
- **Repository:** [STROKE_PREDITION (GitHub)](https://github.com/Bvt9506/STROKE_PREDITION)
- Dự án phục vụ mục đích nghiên cứu, học tập và ứng dụng phân tích dữ liệu chăm sóc sức khỏe.
