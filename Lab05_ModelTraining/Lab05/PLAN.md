# PLAN.md - Lab05: Các giải thuật phân loại cơ bản (chỉ phần CODE)

## VAI TRÒ

Bạn là trợ lý viết code cho bài thực hành Machine Learning (Lab05 - Classification).
Chỉ làm phần code và notebook. KHÔNG viết báo cáo Word/PDF.

## MÔI TRƯỜNG (đã cài sẵn, không cần cài lại)

- Python venv ở .venv, đã có: numpy, pandas, scikit-learn, matplotlib, seaborn, graphviz, jupyter.
- Cấu trúc thư mục: data/ (dữ liệu), notebooks/ (notebook), figures/ (hình xuất ra).
- Dữ liệu có sẵn trong data/: credit_card.csv, titanic_train.csv, diabetes.csv, animal.csv,
  spam.csv, customer.csv, mushrooms.csv.
- Iris và Digits dùng trực tiếp từ sklearn.datasets, không cần file.
- Nếu thiếu file hoặc tên cột khác dự đoán: đọc df.head(), df.info() trước, rồi chỉnh code theo
  dữ liệu thật. Nếu vẫn không xử lý được thì dừng lại và hỏi tôi, KHÔNG tự bịa dữ liệu.

## QUY TẮC CHUNG

- Mỗi bài là 1 notebook riêng trong notebooks/, đặt tên đúng như danh sách bên dưới.
- Comment bằng tiếng Việt, thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
- Cấu trúc mỗi notebook (dùng markdown cell để chia mục):
  1. Mục tiêu 2) Load data 3) EDA ngắn (shape, missing, phân bố target)
  2. Tiền xử lý 5) Train 6) Tuning 7) Đánh giá 8) Nhận xét ngắn (overfitting/underfitting,
     model nào tốt, vì sao) 9) Bảng kết quả tóm tắt.
- Đánh giá đầy đủ: accuracy, precision, recall, F1, confusion matrix; thêm ROC AUC nếu nhị phân.
- Lưu hình vào figures/ (dpi 150), tên file có tiền tố số notebook, ví dụ 01_tree_depth2.png.
- Cuối mỗi notebook, append 1 dòng kết quả vào figures/results_summary.csv
  (cột: notebook, dataset, model, best_params, accuracy, precision, recall, f1, roc_auc).
- Dùng random_state cố định. Dùng Pipeline khi có scale/encode để tránh data leakage.
- Chọn tham số/kernel bằng cross-validation, KHÔNG chọn bằng test set.
- Mỗi notebook phải chạy sạch từ đầu (Restart & Run All) không lỗi trước khi báo xong.

## LỖI TRONG ĐỀ MẪU CẦN SỬA

- error_score trong GridSearchCV phải là np.nan, không phải chuỗi 'np.nan'.
- confusion_matrix(y_true, y_pred): đúng thứ tự, không đảo.
- Digits: dùng slice [1500:] cho tập test, không dùng [1501:].
- SVM luôn scale dữ liệu (StandardScaler trong Pipeline).
- Iris là hoa diên vĩ (không phải cẩm chướng) khi viết markdown.
- Graphviz: nếu export_graphviz lỗi do chưa có chương trình Graphviz trong PATH,
  dùng sklearn.tree.plot_tree thay thế.

## DANH SÁCH NOTEBOOK (làm theo thứ tự, xong mỗi cái thì dừng báo kết quả)

### Phần 1: Decision Tree & Random Forest

01_credit_card_dt_rf.ipynb (data/credit_card.csv - bài mẫu)

- File data/credit_card.csv là bản gốc, 30000 dòng, 25 cột, đã xác nhận các cột:
  ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2..PAY_6, BILL_AMT1..6,
  PAY_AMT1..6, 'default payment next month'.
- Cleaning: đổi PAY_0 thành PAY_1; loại các dòng có EDUCATION thuộc {0,5,6} hoặc MARRIAGE = 0;
  in ra số dòng trước và sau khi lọc.
- Bỏ các cột: ID, SEX, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 (không cần EDUCATION_CAT hay cột dummy
  vì file này không có). Giữ 17 feature + target 'default payment next month'.
- Split 80/20, random_state=24.
- Nhiệm vụ 1: DecisionTreeClassifier(max_depth=2), vẽ cây (filled, rounded, proportion).
- Nhiệm vụ 2: GridSearchCV max_depth [1,2,4,6,8,10,12], scoring='roc_auc', cv=4,
  return_train_score=True; vẽ train vs test ROC AUC (errorbar, ±1 SE).
- Nhiệm vụ 3: RandomForestClassifier, GridSearchCV n*estimators 10..100 bước 10; vẽ 2 biểu đồ
  (mean fit time và mean test ROC AUC theo số cây); in best_params*; vẽ feature importance (barh).
  02_titanic_dt_rf.ipynb (data/titanic_train.csv, target Survived)
- Xử lý missing (Age, Embarked, Cabin), encode Sex/Embarked, DecisionTree + RandomForest,
  GridSearchCV, feature importance, so sánh DT vs RF.
  03_diabetes_dt_rf.ipynb (data/diabetes.csv)
- Xác định cột target từ dữ liệu (thường là Outcome), xử lý giá trị 0 bất hợp lý nếu có,
  DT + RF, tuning, feature importance.

### Phần 2: SVM

04_iris_svm.ipynb (sklearn iris - bài mẫu)

- Split 80/20 random_state=101, SVC mặc định, so sánh kernel linear/poly/rbf/sigmoid
  bằng cross-validation (có thể dùng GridSearchCV).
  05_digits_svm.ipynb (sklearn load_digits - bài mẫu)
- Hiển thị lưới 8x8 ảnh, hàm view_digit; SVC(gamma=0.001, C=100), train 1500 mẫu đầu,
  test phần còn lại; confusion matrix (heatmap) + classification_report.
  06_diabetes_svm.ipynb (data/diabetes.csv)
- Pipeline(StandardScaler, SVC), GridSearchCV C/gamma/kernel, đánh giá đầy đủ.
  07_animal_svm.ipynb (data/animal.csv)
- Xác định target từ dữ liệu, encode categorical, Pipeline scale + SVC, GridSearchCV, đánh giá.

### Phần 3: Naive Bayes

08_spam_nb.ipynb (data/spam.csv, encoding latin-1 - bài mẫu)

- Bỏ cột thừa (Unnamed), đổi tên cột thành label/text, split 80/20 random_state=42,
  CountVectorizer + MultinomialNB, accuracy/confusion matrix/classification_report.
  09_customer_nb.ipynb (data/customer.csv)
- Chọn loại NB phù hợp kiểu dữ liệu (GaussianNB cho số, scale nếu cần), đánh giá đầy đủ.
  10_mushroom_nb.ipynb (data/mushrooms.csv)
- Toàn categorical: dùng OrdinalEncoder + CategoricalNB (hoặc OneHot + BernoulliNB),
  giải thích ngắn lý do chọn trong markdown, đánh giá đầy đủ.

### Tổng hợp

11_compare_models.ipynb

- Trên dataset Diabetes: so sánh DecisionTree, RandomForest, SVM, NaiveBayes cùng
  split và cùng bộ metric; xuất bảng so sánh + biểu đồ cột vào figures/.

## KẾT THÚC

- Kiểm tra lại: tất cả notebook chạy sạch từ đầu, figures/ có đủ hình,
  figures/results_summary.csv có đủ dòng cho các notebook.
- Liệt kê cho tôi: danh sách file đã tạo + những chỗ bạn phải giả định (tên cột, target...) để tôi kiểm tra.

## GHI ĐÈ THEO DỮ LIỆU THẬT (đã kiểm tra các file trong data/; nếu mâu thuẫn với mô tả phía trên thì LÀM THEO PHẦN NÀY)

### Titanic (02) - data/titanic_train.csv, 891 dòng

- Target: Survived. Cột: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.
- Bỏ PassengerId, Name, Ticket, Cabin (Cabin thiếu quá nhiều). Age: điền median; Embarked: điền mode; encode Sex, Embarked. Có thể thêm FamilySize = SibSp + Parch + 1.
- Dùng Pipeline/ColumnTransformer để imputer và encoder chỉ fit trên tập train.

### Diabetes (03, 06, 11) - data/diabetes.csv, 100000 dòng

- Đây là bộ "Diabetes prediction dataset" (KHÔNG phải Pima). Cột: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes. Target: diabetes (0/1).
- Bỏ bước "xử lý giá trị 0 bất hợp lý" (không áp dụng cho bộ này).
- Categorical: gender, smoking_history (OneHot). Số: age, bmi, HbA1c_level, blood_glucose_level. hypertension, heart_disease đã là 0/1.
- Dữ liệu mất cân bằng (khoảng 8-9% lớp dương): split có stratify=y; ngoài accuracy phải báo cáo recall, F1 và ROC AUC của lớp 1; thử thêm class_weight='balanced'.
- In số dòng trùng lặp; được phép drop_duplicates, ghi rõ trong notebook.
- SVM trên 100000 dòng rất chậm: ở notebook 06 lấy mẫu phân tầng (stratified) 10000 dòng để tuning bằng GridSearchCV (lưới nhỏ, cv=3), rồi đánh giá trên test set của mẫu đó. Notebook 03 (DT/RF) có thể dùng toàn bộ dữ liệu.
- Notebook 11: dùng CÙNG một stratified sample 10000 dòng và cùng split cho cả 4 mô hình (DT, RF, SVM, NB) để so sánh công bằng.

### Animal (07) - data/animal.csv, 871 dòng

- Cột: AnimalName, symptoms1..symptoms5, Dangerous. Target: Dangerous. Tất cả là categorical.
- In unique values của Dangerous, kiểm tra missing. Kiểm tra mỗi AnimalName có map sang đúng 1 nhãn không; nếu có thì bỏ AnimalName khỏi feature (tránh leakage) và ghi chú trong notebook.
- Encode symptoms1..5 bằng OneHotEncoder(handle_unknown='ignore'), rồi Pipeline + SVC + GridSearchCV.

### Spam (08) - data/spam.csv, 5572 dòng (bộ SMS Spam Collection)

- Đọc với encoding='latin-1'. Cột v1 (ham/spam), v2 (text), bỏ 3 cột Unnamed. Đổi tên thành label, text.
- Chia 80/20 random_state=42; test set phải đúng 1115 mẫu (đối chiếu với đề).

### Customer (09) - data/customer.csv, 400 dòng

- Cột: User ID, Gender, Age, EstimatedSalary, Purchased. Target: Purchased (0/1).
- Bỏ User ID; encode Gender; feature gồm Gender, Age, EstimatedSalary. Dùng GaussianNB; so sánh có/không StandardScaler để nhận xét.

### Mushroom (10) - data/mushrooms.csv, 8124 dòng

- Target: class (e = ăn được, p = độc). 22 feature đều là categorical (ký tự đơn).
- stalk-root có giá trị '?' (missing): giữ '?' như một category riêng. veil-type chỉ có 1 giá trị: bỏ.
- Dùng OrdinalEncoder + CategoricalNB, đặt min_categories theo số category mỗi cột (tính trên toàn bộ X) để tránh lỗi category lạ ở tập test. Nếu vẫn lỗi thì chuyển sang OneHotEncoder + BernoulliNB.
- Ghi nhận xét: accuracy rất cao là bình thường với dữ liệu này.
- Ghi file có tiếng Việt: KHÔNG dùng heredoc, echo, Set-Content hay redirect (>) của PowerShell. 
  Dùng công cụ sửa file của editor, hoặc viết script Python lưu thành file .py và ghi bằng 
  encoding='utf-8' (nbformat.write hoặc json.dump(..., ensure_ascii=False)). 
  Đặt $env:PYTHONUTF8='1' trước khi chạy Python. Sau khi ghi phải đọc lại để chắc chữ có dấu còn nguyên.