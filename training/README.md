# Random Forest trên đặc trưng repetition

Pipeline hiện tại dùng **một dòng cho mỗi repetition**. Mỗi dòng chứa các đặc trưng
`start`, `turning` và `rom`; không dùng chuỗi frame và không có epoch/batch.

Quy ước nhãn:

```text
0 = false = incorrect
1 = true  = correct
```

## Dữ liệu đầu vào

Thư mục dữ liệu phải có sáu file:

```text
angles_train.csv   labels_train.csv
angles_val.csv     labels_val.csv
angles_test.csv    labels_test.csv
```

`angles_*.csv`:

```csv
exercise,rep_id,LEFT_ELBOW_start,LEFT_ELBOW_turning,LEFT_ELBOW_rom,...
Abduction,R001,169.54,167.96,4.18,...
```

`labels_*.csv`:

```csv
exercise,rep_id,label
Abduction,R001,1
Abduction,R002,0
```

Ô trống được giữ là `NaN`. Pipeline fit median imputer trên tập train, đồng thời thêm
missing indicator trước khi đưa dữ liệu vào Random Forest. Validation dùng để chọn
`max_depth`, `min_samples_leaf` và `max_features`; test chỉ dùng để đánh giá cuối.

## Train

```powershell
python -m training.train --data-dir "C:\path\to\dataset" --exercise Abduction
```

Kết quả:

```text
artifacts/abduction/
├── classifier.joblib
├── reference_profile.json
└── training_report.json
```

`training_report.json` chứa Accuracy, Balanced Accuracy, Macro F1, metric riêng cho
incorrect/correct, ROC-AUC, PR-AUC, confusion matrix và baseline chỉ nhìn ô bị thiếu.

## Predict

CSV dự đoán có cùng các cột feature, không cần file labels:

```powershell
python -m training.predict --angles datasets/new_repetitions.csv --exercise Abduction
```

`rep_id` và `exercise` chỉ dùng làm khóa định danh, không được đưa vào model.

## Predict trực tiếp từ camera

`realtime_predict.py` load `classifier.joblib` một lần, nhận feature repetition
trực tiếp trong bộ nhớ và không tạo CSV tạm:

```powershell
python realtime_predict.py --source 0 --exercise Abduction
```

Model không dự đoán từng frame. `src/repetition_tracker.py` theo dõi chuỗi góc,
phát hiện điểm đổi chiều và chỉ gửi 12 feature vào model sau khi người tập quay
về tư thế ban đầu. Nếu quan sát được dưới 50% feature, chương trình từ chối dự
đoán thay vì tự thay khớp bị khuất bằng góc 0.
