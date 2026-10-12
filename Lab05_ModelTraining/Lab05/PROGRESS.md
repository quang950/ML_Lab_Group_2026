# PROGRESS - Lab05

Quy ước: mỗi lần chỉ làm đúng các notebook được giao. Xong thì đánh dấu [x] và ghi 3-5 dòng
kết quả chính (số liệu quan trọng, giả định đã đặt, vấn đề còn tồn đọng).

- [x] 01_credit_card_dt_rf
- [x] 02_titanic_dt_rf
  - Random Forest tốt hơn Decision Tree: accuracy 0.8715, F1 0.8296, ROC AUC 0.8873 (DT: ROC AUC 0.8292).
  - Đã dùng `FamilySize`, bỏ PassengerId/Name/Ticket/Cabin; Age median và Embarked mode được fit trong Pipeline.
  - Best RF dùng `class_weight='balanced'`, `max_depth=10`, `n_estimators=200`; test stratify, random_state=24.
- [x] 03_diabetes_dt_rf
  - Random Forest tốt hơn Decision Tree về ROC AUC/F1: accuracy 0.9320, recall 0.8544, F1 0.6892, ROC AUC 0.9772.
  - Đã in và loại 3.854 dòng trùng lặp; không xử lý số 0 vì đây là Diabetes prediction dataset theo phần ghi đè.
  - Categorical `gender`/`smoking_history` dùng OneHotEncoder; đã thử `class_weight=None` và `balanced`, split stratify, random_state=24.
- [x] 04_iris_svm
  - SVM chọn kernel `linear` bằng 5-fold CV; test accuracy 0.9667, precision 0.9697, recall 0.9667, F1 0.9666.
  - Dùng Iris trực tiếp từ `sklearn.datasets`, split stratify 80/20 với `random_state=101`; `StandardScaler` nằm trong Pipeline.
  - Giả định: dùng weighted-average cho precision/recall/F1 vì Iris là bài toán đa lớp; ROC AUC để trống vì chỉ yêu cầu thêm cho bài toán nhị phân.
- [x] 05_digits_svm
  - SVC RBF với `C=100`, `gamma=0.001` đạt test accuracy 0.9158, precision 0.9232, recall 0.9158, F1 0.9152.
  - Giữ đúng đề mẫu: train `[:1500]`, test `[1500:]`; hiển thị lưới ảnh 8x8, confusion matrix và classification report.
  - Giả định: thêm `StandardScaler` trong Pipeline theo quy tắc chung SVM; không tuning vì đề mẫu chỉ định sẵn C/gamma/kernel.
- [x] 06_diabetes_svm
  - SVM trên mẫu phân tầng 10.000 dòng sau khi bỏ 3.854 dòng trùng lặp; best `C=0.1`, `kernel=linear`, `class_weight='balanced'`, CV ROC AUC 0.9632.
  - Test: accuracy 0.8860, precision 0.4297, recall 0.9034, F1 0.5824, ROC AUC 0.9622; đã lưu confusion matrix và target distribution.
  - Giả định: dùng sample 10.000 dòng theo phần ghi đè để tuning/đánh giá; giữ nguyên các giá trị 0 và dùng Pipeline để encode/scale, tránh leakage.
- [x] 07_animal_svm
  - Sau khi bỏ 2 dòng thiếu `Dangerous`, có 869 dòng; 3 `AnimalName` ánh xạ nhiều nhãn nên loại `AnimalName` khỏi feature để tránh leakage.
  - Best `C=10`, `kernel=rbf`, `gamma='scale'`; test accuracy/precision/recall/F1/ROC AUC đều 1.0000; đã lưu confusion matrix và target distribution.
  - Giả định: loại dòng thiếu target, giữ năm cột symptoms categorical, dùng OneHotEncoder + StandardScaler trong Pipeline.
- [x] 08_spam_nb
  - CountVectorizer + MultinomialNB, chia stratify 80/20 `random_state=42`, đúng 1115 mẫu test; best `alpha=0.5`, unigram.
  - Test: accuracy 0.9857, precision 0.9650, recall 0.9262, F1 0.9452, ROC AUC 0.9843; đã lưu target distribution và confusion matrix.
  - Giả định: giữ `ham`/`spam` dạng chuỗi để báo cáo lớp spam, dùng F1 macro cho CV vì scorer mặc định không hỗ trợ `pos_label='spam'`.
- [x] 09_customer_nb
  - Bỏ `User ID`, one-hot `Gender`, dùng GaussianNB; GridSearchCV 5-fold so sánh `StandardScaler` với không scale.
  - CV chọn không scale; test: accuracy 0.8750, precision 0.8519, recall 0.7931, F1 0.8214, ROC AUC 0.9500.
  - Giả định: `Purchased` là target nhị phân 0/1, giữ nguyên 400 dòng và split stratify `random_state=42`.
- [x] 10_mushroom_nb
  - OrdinalEncoder + CategoricalNB; giữ `?` trong `stalk-root` như category và bỏ `veil-type` vì chỉ có một giá trị.
  - Test: accuracy 0.9458, precision 0.9901, recall 0.8966, F1 0.9410, ROC AUC 0.9972; accuracy cao là bình thường với Mushroom.
  - Giả định: dùng `class` với `e=0` (ăn được), `p=1` (độc); `min_categories` tính từ toàn bộ feature để tránh lỗi category lạ.
- [x] 11_compare_models
  - Dùng cùng stratified sample 10.000 dòng và cùng split `random_state=24` sau khi loại 3.854 dòng trùng; so sánh DecisionTree, RandomForest, SVM và GaussianNB.
  - RandomForest tốt nhất theo ROC AUC/F1: accuracy 0.9545, precision 0.7249, recall 0.7784, F1 0.7507, ROC AUC 0.9717.
  - Giả định: Diabetes target là `diabetes`; categorical `gender`/`smoking_history` OneHot, số còn lại median imputation; dùng `class_weight='balanced'` cho tree/RF/SVM.

## Ghi chú