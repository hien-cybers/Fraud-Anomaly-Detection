# 🛡️ Phát hiện gian lận giao dịch thẻ tín dụng

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-FF6A00.svg)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2.svg)

Dự án **Khai phá dữ liệu (Data Mining)** nhằm phát hiện các giao dịch thẻ tín dụng có dấu hiệu gian lận bằng cách kết hợp các phương pháp **học máy có giám sát và không giám sát**.

---

## 📌 Tổng quan dự án

Phát hiện gian lận thẻ tín dụng là một bài toán quan trọng trong lĩnh vực **an ninh mạng và khai phá dữ liệu**. Bài toán trở nên đặc biệt khó khăn do số lượng giao dịch gian lận chiếm một tỷ lệ rất nhỏ so với tổng số giao dịch hợp lệ.

Dự án xây dựng một quy trình khai phá dữ liệu hoàn chỉnh nhằm phát hiện các giao dịch gian lận từ một tập dữ liệu có mức độ **mất cân bằng lớp (Class Imbalance)** rất cao.

### 🎯 Mục tiêu

- Khám phá và phân tích dữ liệu giao dịch.
- Tiền xử lý và chuẩn hóa dữ liệu.
- Phân tích cấu trúc dữ liệu bằng các phương pháp giảm chiều.
- Trực quan hóa dữ liệu trong không gian 2 chiều.
- Xây dựng các mô hình học máy có giám sát.
- Áp dụng các thuật toán học không giám sát để tìm kiếm điểm bất thường.
- Đánh giá mô hình bằng các chỉ số phù hợp với dữ liệu mất cân bằng.

Một trong những thách thức lớn nhất là **giao dịch gian lận chỉ chiếm 0,1727%** tổng số giao dịch.

---

# 📊 Dữ liệu

## Nguồn dữ liệu

**Credit Card Fraud Detection Dataset**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Đặc điểm dữ liệu

| Thuộc tính | Giá trị |
|:---|---:|
| Tổng số giao dịch | 284.807 |
| Số lượng đặc trưng | 30 |
| Tổng số cột | 31 |
| Giao dịch hợp lệ | 284.315 |
| Giao dịch gian lận | 492 |
| Tỷ lệ gian lận | 0,1727% |

### Phân bố lớp

- `0` — Giao dịch hợp lệ: **99,8273%**
- `1` — Giao dịch gian lận: **0,1727%**

Do dữ liệu có mức độ mất cân bằng rất lớn, **Accuracy không thể được sử dụng như chỉ số đánh giá duy nhất**.

---

# 🗂️ Cấu trúc dự án

```text
Fraud-Anomaly-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.zip
├── requirements.txt
├── README.md
└── .gitattributes
```

Trong đó:

- `Credit_Card_Fraud_Detection.ipynb`: Notebook chứa toàn bộ quá trình phân tích và xây dựng mô hình.
- `creditcard.zip`: Tập dữ liệu giao dịch thẻ tín dụng được nén.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.
- `README.md`: Tài liệu mô tả dự án.
- `.gitattributes`: Cấu hình Git LFS.

Tập dữ liệu được quản lý bằng **Git LFS (Git Large File Storage)** nhằm hỗ trợ lưu trữ các tệp dữ liệu có kích thước lớn.

---

# 🛠️ Phương pháp thực hiện

Dự án được chia thành **4 giai đoạn chính**.

## 1. 🔍 Phân tích khám phá dữ liệu và tiền xử lý

Các công việc chính:

- Kiểm tra cấu trúc và kiểu dữ liệu.
- Kiểm tra dữ liệu bị thiếu.
- Phân tích các thống kê mô tả.
- Phân tích phân phối của các đặc trưng.
- Kiểm tra mức độ mất cân bằng giữa hai lớp.
- Phân tích sự khác biệt giữa giao dịch hợp lệ và giao dịch gian lận.
- Chuẩn hóa các đặc trưng số bằng `StandardScaler`.

### Chuẩn hóa dữ liệu

```text
z = (x - μ) / σ
```

Trong đó:

- `x`: giá trị ban đầu.
- `μ`: giá trị trung bình.
- `σ`: độ lệch chuẩn.

Việc chuẩn hóa đặc biệt quan trọng đối với PCA, t-SNE, K-Means và DBSCAN.

---

## 2. 📉 Giảm chiều dữ liệu và trực quan hóa

### PCA — Phân tích thành phần chính

**PCA (Principal Component Analysis)** được sử dụng để chuyển đổi dữ liệu từ không gian nhiều chiều sang không gian có số chiều thấp hơn, đồng thời cố gắng giữ lại phần lớn phương sai của dữ liệu.

Mục đích:

- Giảm số chiều dữ liệu.
- Phân tích cấu trúc tuyến tính tổng thể.
- Hỗ trợ trực quan hóa dữ liệu.
- Quan sát sự phân bố của giao dịch gian lận và giao dịch hợp lệ.

### t-SNE

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** được sử dụng để khám phá các mối quan hệ **phi tuyến tính** trong dữ liệu.

t-SNE tập trung vào việc bảo toàn mối quan hệ lân cận giữa các điểm dữ liệu. Do tập dữ liệu có kích thước lớn, dự án sử dụng chiến lược **lấy mẫu tối ưu** nhằm giảm thời gian tính toán.

---

## 3. 🤖 Học máy có giám sát

Bài toán được xem như một bài toán **phân loại nhị phân**.

Ba thuật toán được sử dụng:

- **Naive Bayes**
- **AdaBoost**
- **Random Forest**

### Tối ưu mô hình

Các siêu tham số được tối ưu bằng:

```text
GridSearchCV
```

Đánh giá mô hình bằng:

```text
Stratified 10-Fold Cross-Validation
```

Phương pháp Stratified K-Fold giúp duy trì tỷ lệ tương đối giữa giao dịch hợp lệ và giao dịch gian lận trong các fold.

### 📏 Các chỉ số đánh giá

Do dữ liệu mất cân bằng nghiêm trọng, **Macro F1 được sử dụng làm chỉ số quan trọng thay vì chỉ dựa vào Accuracy**.

#### Macro F1-Score

```text
Macro F1 = (F1_valid + F1_fraud) / 2
```

Macro F1 giúp hai lớp được coi trọng như nhau trong quá trình đánh giá.

#### Fraud F1-Score

Fraud F1 là F1-score tính riêng cho lớp gian lận, phản ánh sự cân bằng giữa:

- **Precision**: Tỷ lệ dự đoán gian lận là đúng.
- **Recall**: Tỷ lệ giao dịch gian lận thực tế được phát hiện.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Weighted F1-Score

Weighted F1 tính đến số lượng mẫu của từng lớp. Vì lớp hợp lệ chiếm phần lớn dữ liệu, chỉ số này có thể rất cao ngay cả khi khả năng phát hiện gian lận chưa cao.

Vì vậy, Weighted F1 cần được xem xét cùng với **Macro F1 và Fraud F1**.

---

# 🏆 Kết quả mô hình

Kết quả đánh giá bằng **Stratified 10-Fold Cross-Validation**:

| Mô hình | Macro F1 | Weighted F1 | Fraud F1 | Accuracy |
|:---|---:|---:|---:|---:|
| **Random Forest** | **0,8847** | **0,9992** | **0,7699** | **0,9992** |
| AdaBoost | 0,8531 | 0,9990 | 0,7067 | 0,9991 |
| Naive Bayes | 0,5510 | 0,9872 | 0,1133 | 0,9777 |

### Nhận xét

Dựa trên kết quả thực nghiệm:

- **Random Forest** đạt Macro F1 là **0,8847**.
- Random Forest đạt Fraud F1 là **0,7699**.
- **AdaBoost** đạt Macro F1 là **0,8531** và Fraud F1 là **0,7067**.
- **Naive Bayes** có Fraud F1 là **0,1133**.
- Accuracy của các mô hình đều cao, cho thấy Accuracy không phản ánh đầy đủ khả năng phát hiện lớp gian lận trong dữ liệu mất cân bằng.

> **Lưu ý:** Kết quả có thể thay đổi tùy thuộc vào cách tiền xử lý dữ liệu, phương pháp lấy mẫu, cấu hình siêu tham số và `random_state`.

---

# 4. 🔎 Học máy không giám sát

Giai đoạn này tiếp cận bài toán dưới góc nhìn **phát hiện bất thường (Anomaly Detection)** mà không trực tiếp sử dụng nhãn gian lận trong quá trình phân cụm.

Hai thuật toán được sử dụng:

- **K-Means**
- **DBSCAN**

## K-Means

K-Means phân chia dữ liệu thành một số lượng cụm `K` được xác định trước.

Mục đích:

- Phân tích cấu trúc tự nhiên của dữ liệu.
- Tìm kiếm các nhóm giao dịch có đặc điểm tương đồng.
- Quan sát khả năng phân tách giữa các nhóm giao dịch.

## DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên mật độ.

DBSCAN xác định các vùng có mật độ điểm dữ liệu cao và đánh dấu những điểm nằm ngoài các vùng này là **nhiễu (Noise)**.

Dự án sử dụng **k-distance graph** để hỗ trợ xác định giá trị `eps`.

Hai tham số quan trọng:

- `eps`: Bán kính tối đa để xác định vùng lân cận.
- `min_samples`: Số lượng điểm tối thiểu trong vùng lân cận để hình thành một vùng có mật độ đủ cao.

Các điểm được DBSCAN xác định là **Noise** có thể tiếp tục được phân tích như những ứng viên tiềm năng cho các giao dịch bất thường.

---

# 🧠 Tại sao sử dụng cả học có giám sát và không giám sát?

Hai phương pháp tiếp cận bài toán từ hai góc độ khác nhau.

### Học có giám sát

Mô hình sử dụng dữ liệu đã có nhãn:

```text
Giao dịch → Hợp lệ / Gian lận
```

Mục tiêu là học các đặc điểm từ những giao dịch đã được gắn nhãn để dự đoán nhãn cho các giao dịch mới.

### Học không giám sát

Thuật toán không trực tiếp sử dụng nhãn gian lận trong quá trình phân cụm:

```text
Giao dịch → Cụm / Nhiễu
```

Mục tiêu là tìm kiếm những mẫu dữ liệu bất thường hoặc những điểm nằm biệt lập so với phần lớn dữ liệu.

Việc kết hợp hai phương pháp giúp cung cấp góc nhìn toàn diện hơn về bài toán phát hiện gian lận.

---

# 🧰 Công nghệ sử dụng

| Công nghệ | Mục đích |
|:---|:---|
| **Python 3.x** | Ngôn ngữ lập trình |
| **Jupyter Notebook** | Phân tích và thực nghiệm |
| **Pandas** | Xử lý dữ liệu |
| **NumPy** | Tính toán số học |
| **Matplotlib** | Trực quan hóa dữ liệu |
| **Seaborn** | Trực quan hóa thống kê |
| **Scikit-learn** | Xây dựng mô hình Machine Learning |
| **Git** | Quản lý mã nguồn |
| **Git LFS** | Quản lý tập dữ liệu lớn |

---

# 🚀 Cài đặt và sử dụng

## Bước 1 — Clone repository

```bash
git clone https://github.com/hien-cybers/Fraud-Anomaly-Detection.git
cd Fraud-Anomaly-Detection
```

## Bước 2 — Tải dữ liệu bằng Git LFS

Đảm bảo Git LFS đã được cài đặt trên máy:

```bash
git lfs install
git lfs pull
```

## Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Bước 4 — Chạy Notebook

Mở file:

```text
Credit_Card_Fraud_Detection.ipynb
```

bằng:

- Jupyter Notebook
- JupyterLab
- Visual Studio Code

Notebook sẽ đọc dữ liệu từ:

```text
creditcard.zip
```

Nếu notebook được cấu hình đọc trực tiếp file ZIP, không cần giải nén thủ công tập dữ liệu.

---

# 👥 Thành viên nhóm

## Nhóm 3

| Thành viên | Phân công |
|:---|:---|
| **Nguyen Duc Hien** | Quản lý dự án & Khám phá dữ liệu / Tiền xử lý |
| **Hoang Thi Ngoc Tram** | Giảm chiều dữ liệu: PCA / t-SNE & Trực quan hóa 2D |
| **Tran Truong Thinh** | Xây dựng mô hình phân loại & Tối ưu siêu tham số |
| **Nguyen Luong Vinh Chi** | Học không giám sát: K-Means / DBSCAN |
| **Nguyen Anh Vu** | Đánh giá Stratified 10-Fold Cross-Validation & Báo cáo cuối kỳ |

**Đơn vị:** University of Transport and Communications / UTH

---

# 📌 Kết luận

Dự án xây dựng một quy trình hoàn chỉnh cho bài toán **phát hiện gian lận giao dịch thẻ tín dụng**, bao gồm:

```text
Thu thập dữ liệu
       ↓
Khám phá dữ liệu
       ↓
Tiền xử lý dữ liệu
       ↓
Chuẩn hóa dữ liệu
       ↓
PCA / t-SNE
       ↓
Học máy có giám sát
       ↓
Tối ưu siêu tham số
       ↓
Stratified 10-Fold Cross-Validation
       ↓
Học máy không giám sát
       ↓
Đánh giá mô hình
```

Dự án cho thấy:

- Accuracy có thể gây hiểu nhầm khi dữ liệu mất cân bằng nghiêm trọng.
- Macro F1 giúp đánh giá cân bằng hơn giữa các lớp.
- Fraud F1 là chỉ số quan trọng để đánh giá khả năng phát hiện giao dịch gian lận.
- PCA và t-SNE hỗ trợ trực quan hóa dữ liệu nhiều chiều.
- Học có giám sát và học không giám sát cung cấp hai góc nhìn bổ sung cho bài toán phát hiện bất thường.
- Stratified 10-Fold Cross-Validation giúp đánh giá mô hình ổn định hơn so với chỉ sử dụng một lần chia tập train/test.

---

# 📄 Giấy phép

Dự án được thực hiện với mục đích **học tập và nghiên cứu học thuật**.

Tập dữ liệu được cung cấp thông qua Kaggle và cần được sử dụng theo các điều khoản và giấy phép áp dụng của bộ dữ liệu.

---

# 🙏 Lời cảm ơn

Xin cảm ơn các tác giả và cộng đồng mã nguồn mở đã cung cấp bộ dữ liệu **Credit Card Fraud Detection** cũng như các công cụ Python hỗ trợ quá trình phân tích và xây dựng mô hình Machine Learning.

### Nguồn dữ liệu

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
