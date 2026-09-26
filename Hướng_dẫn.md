# Hướng dẫn chạy chương trình từ đầu đến cuối

Tài liệu này hướng dẫn chạy pipeline xử lý video hàng loạt. Sau khi chạy, chương trình ghi dữ liệu góc vào `angles/angles.csv` và tự tạo nhãn rep tại `labels/labels.csv`.

## 1. Yêu cầu

- Windows và PowerShell.
- Python tương thích với các thư viện trong `requirements.txt`.
- Các video bài tập cần xử lý.
- File chấm điểm `Physiotherapist Exercise Marking .csv` đặt ở thư mục gốc dự án. Giữ nguyên tên file, bao gồm dấu cách trước `.csv`.

## 2. Chuẩn bị thư mục và video

Đặt video trong thư mục `data`. Chương trình tìm video trong cả các thư mục con; định dạng hỗ trợ gồm `.mp4`, `.mov`, `.avi`, `.mkv`, `.wmv` và `.m4v`.

Tên không có phần mở rộng được dùng làm `rep_id` gốc. Mỗi rep phát hiện trong video sẽ có `rep_id` dạng `<tên_video>_1`, `<tên_video>_2`, v.v. Chương trình dùng mã như `E01` để xác định tên bài tập.

## 3. Tạo môi trường Python

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\DATN\Physical_Therapy_Model
```

Tạo và kích hoạt môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường ảo, có thể chạy Python trong môi trường bằng đường dẫn trực tiếp, ví dụ:

```powershell
.\.venv\Scripts\python.exe main.py --input-dir data --output-file angles/angles.csv
```

## 4. Chạy pipeline

Từ thư mục gốc dự án, chạy:

```powershell
python main.py --input-dir data --output-file angles/angles.csv
```

Chương trình sẽ lần lượt đọc video, nhận diện tư thế, phát hiện các rep hoàn chỉnh, tính các góc khớp và ghi `angles/angles.csv`. Sau đó chương trình đọc điểm `Average Score` trong file marking, ghép điểm với tên video của từng `rep_id`, rồi ghi `labels/labels.csv`.

Nhãn được gán theo điểm trung bình của video:

- `1` nếu `Average Score` lớn hơn hoặc bằng 80.
- `0` nếu `Average Score` nhỏ hơn 80.

## 5. Kiểm tra kết quả

Sau khi chạy thành công, kiểm tra:

```text
angles/angles.csv
labels/labels.csv
```

`angles.csv` có cột `exercise`, `rep_id` và các cột đo góc. `labels.csv` có ba cột `exercise`, `rep_id`, `label`; mỗi dòng nhãn ứng với một rep có video khớp trong file marking.

Nếu `labels.csv` chỉ có dòng tiêu đề, hãy kiểm tra `angles.csv` có các dòng rep hay không, tên video có khớp với `Video Name` trong file marking hay không, và `rep_id` có bắt đầu bằng tên video hay không. Các rep không ghép được với video trong file marking sẽ được bỏ qua và số lượng bị bỏ qua được báo ở cuối lần chạy.

## 6. Tùy chọn thường dùng

Có thể đổi thư mục video hoặc đường dẫn file góc:

```powershell
python main.py --input-dir data --output-file angles/angles.csv
```

Chạy không hiển thị cửa sổ xem video:

```powershell
python main.py --input-dir data --output-file angles/angles.csv --no-display
```

Một số tùy chọn khác:

- `--angle-dimension 2d_smooth`: chọn loại góc ghi ra; các lựa chọn gồm `2d`, `2d_smooth`, `3d`, `3d_smooth`.
- `--no-smooth`: tắt bộ lọc làm mượt.
- `--smooth-window 5`: độ dài cửa sổ lọc trung vị.
- `--smooth-alpha 0.4`: hệ số EMA.
- `--min-visibility 0.5`: ngưỡng tin cậy tối thiểu của landmark.
- `--rep-min-amplitude 4.0`: biên độ góc tối thiểu để tính một rep.
- `--rep-min-distance 6`: khoảng cách frame tối thiểu giữa các rep.

Ví dụ điều chỉnh ngưỡng nhận diện rep:

```powershell
python main.py --input-dir data --output-file angles/angles.csv --rep-min-amplitude 5 --rep-min-distance 8
```

## 7. Tạo lại riêng file nhãn

Nếu đã có `angles/angles.csv` và muốn tạo lại nhãn mà không xử lý video lần nữa:

```powershell
python create_labels.py
```

Mặc định lệnh này đọc file marking ở thư mục gốc và ghi `labels/labels.csv`.

## 8. Sự cố thường gặp

- **Không tìm thấy video:** kiểm tra thư mục truyền vào `--input-dir` và phần mở rộng video.
- **Không tìm thấy file marking:** đặt `Physiotherapist Exercise Marking .csv` ở thư mục gốc dự án, giữ nguyên tên.
- **`labels.csv` không có dòng dữ liệu:** kiểm tra `angles.csv` có rep và `rep_id` có khớp tên video trong cột `Video Name` không. Rep không có điểm khớp sẽ bị bỏ qua.
- **Không phát hiện rep:** thử điều chỉnh `--rep-min-amplitude`, `--rep-min-distance`; đồng thời kiểm tra video có thấy rõ người tập và các khớp cần thiết không.
- **Thiếu thư viện:** kích hoạt đúng môi trường `.venv`, sau đó chạy lại `pip install -r requirements.txt`.

## 9. Chạy từ đầu sau khi thay video

1. Đặt các video cần xử lý trong `data`.
2. Xác nhận file marking nằm ở thư mục gốc và tên video khớp với `Video Name`.
3. Kích hoạt môi trường `.venv`.
4. Chạy `python main.py --input-dir data --output-file angles/angles.csv`.
5. Mở `angles/angles.csv` và `labels/labels.csv` để kiểm tra kết quả.
