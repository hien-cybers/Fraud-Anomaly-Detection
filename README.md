# 🛡️ Fraud Anomaly Detection: Credit Card Transactions

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-FF6A00.svg)](https://scikit-learn.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2.svg)](https://git-lfs.github.com/)

## 📌 Tổng quan dự án
Dự án thực hiện toàn diện quy trình khai thác dữ liệu (Data Mining) để giải quyết bài toán phát hiện giao dịch gian lận thẻ tín dụng. Trong bối cảnh an toàn thông tin, việc phát hiện gian lận mang bản chất của bài toán **Nhận diện bất thường (Anomaly Detection)**, đòi hỏi hệ thống phải phân biệt được các hành vi tinh vi ẩn giấu giữa hàng nghìn giao dịch hợp lệ.

Thách thức cốt lõi của dự án là xử lý **Dữ liệu mất cân bằng nghiêm trọng (Highly Imbalanced Data)** khi lớp gian lận (Fraud) chỉ chiếm **0.1727%** tổng số giao dịch.

## 📊 Dữ liệu
*   **Nguồn dữ liệu:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
*   **Kích thước:** 284,807 dòng x 31 cột.
*   **Phân bổ nhãn (Class):** `0` (Giao dịch hợp lệ - 99.8273%) | `1` (Giao dịch gian lận - 0.1727%).
*   **Lưu trữ:** Để tối ưu hóa giới hạn băng thông, file dữ liệu gốc đã được nén thành `creditcard.zip` và quản lý trực tiếp trên GitHub thông qua hệ thống **Git LFS (Large File Storage)**.

## 🛠️ Phương pháp luận và Công nghệ
Dự án được triển khai qua 4 giai đoạn phân tích chuyên sâu:

### 1. Tiền xử lý & Khám phá dữ liệu (EDA)
* Phân tích tính toàn vẹn, kiểm tra giá trị thiếu và cấu trúc phân bố các biến.
* Chuẩn hóa dữ liệu bằng `StandardScaler` để đảm bảo độ tin cậy cho các thuật toán tính toán khoảng cách và phương sai.

### 2. Giảm chiều dữ liệu & Trực quan hóa (Dimensionality Reduction)
* Ứng dụng **PCA** để chiếu và đánh giá cấu trúc tuyến tính tổng thể của không gian dữ liệu.
* Áp dụng **t-SNE** (thiết kế chiến lược lấy mẫu tối ưu tài nguyên) để khai phá các mối quan hệ phi tuyến, bảo toàn cấu trúc lân cận cục bộ nhằm bóc tách rõ ràng ranh giới giữa giao dịch thật và giả trên không gian 2D.

### 3. Phân lớp có giám sát (Supervised Learning)
* Triển khai 3 thuật toán tiêu biểu: **Naive Bayes**, **AdaBoost**, và **Random Forest**.
* **Metric tối ưu:** Sử dụng **Macro F1-Score** thay vì Accuracy để tránh "bẫy" độ chính xác ảo trên dữ liệu mất cân bằng, buộc mô hình phải học cách phát hiện chính xác lớp thiểu số (gian lận).
* Tinh chỉnh siêu tham số (Hyperparameter Tuning) với `GridSearchCV` và kiểm định chéo khách quan qua `Stratified 10-Fold Cross-Validation`.

### 4. Gom cụm không giám sát (Unsupervised Learning)
* Tiếp cận bài toán dưới góc độ "săn lùng mối đe dọa mù" (blind threat hunting) sử dụng **K-Means** và **DBSCAN**.
* Vẽ đồ thị *k-distance* để xác định bán kính `eps` tối ưu cho thuật toán DBSCAN.

## 🏆 Kết quả Nổi bật
Kết quả kiểm tra chéo 10-Fold CV cho thấy mô hình **Random Forest** đạt hiệu năng vượt trội nhất trong việc kiểm soát cả tỷ lệ bỏ lọt và báo động giả:

| Mô hình | Macro F1-Score | Weighted F1 | Fraud F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8847** | **0.9992** | **0.7699** | 0.9992 |
| **AdaBoost** | 0.8531 | 0.9990 | 0.7067 | 0.9991 |
| **Naive Bayes** | 0.5510 | 0.9872 | 0.1133 | 0.9777 |

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

**Bước 1: Clone repository và kéo dữ liệu LFS**
```bash
git clone [https://github.com/hien-cybers/Fraud-Anomaly-Detection.git](https://github.com/hien-cybers/Fraud-Anomaly-Detection.git)
cd Fraud-Anomaly-Detection
git lfs pull
