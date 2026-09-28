# AI_model

Nhận diện tư thế bằng MediaPipe Pose, tính góc khớp theo từng frame và xuất
chuỗi góc theo trục thời gian.

## Cấu trúc

| Tệp | Vai trò |
|---|---|
| `main.py` | Pipeline chính: nhận diện → tính góc → vẽ → ghi log |
| `realtime_predict.py` | Tách từng repetition từ camera và gọi model đúng/sai |
| `export_keypoints.py` | Chỉ xuất tọa độ khớp thô ra JSON |
| `src/pose_detector.py` | Bọc MediaPipe Pose, trả landmark và vẽ skeleton/FPS |
| `src/angle_calculator.py` | Tính góc 2D/3D bằng tích vô hướng, vẽ cung tròn + số đo |
| `src/angle_smoother.py` | Lọc nhiễu chuỗi góc (trung vị + EMA) |
| `src/angle_logger.py` | Gom chuỗi góc theo thời gian, xuất CSV/JSON |
| `src/repetition_tracker.py` | Phát hiện bắt đầu/đổi chiều/kết thúc và tạo feature repetition |
| `tests/` | Bộ test tự động (`python -m unittest discover -s tests -t .`) |

## Chạy

```bash
python main.py                          # webcam
python main.py --source data/11.mp4     # file video
python main.py --source data/20.mov --no-display   # chỉ xuất log, không mở cửa sổ
```

Kết quả ghi vào `angles/angles_<tên video>.csv` và `.json`.

## Dự đoán realtime từ webcam

Cài cả dependency thị giác máy tính và training trong Python 3.11 (phù hợp với
`mediapipe==0.10.14` mà dự án đang dùng), sau đó chạy:

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
python realtime_predict.py
```

Giữ tư thế bắt đầu ổn định khoảng 0,5 giây, thực hiện trọn một repetition rồi
quay về tư thế ban đầu. Chương trình chỉ dự đoán sau khi repetition hoàn thành.
Nhấn `r` để lấy lại mốc góc ban đầu và `q` để thoát.

Webcam mặc định được yêu cầu ở độ phân giải `1280x720` và cửa sổ có thể kéo lớn
nhỏ mà vẫn giữ tỷ lệ. Có thể chọn kích thước khác:

```powershell
python realtime_predict.py --camera-width 1920 --camera-height 1080 `
  --window-width 1440 --window-height 810
```

Panel trạng thái tự co theo độ phân giải và dùng nền bán trong suốt. Nếu hiện
`Framing: ... joints - move back`, hãy lùi xa camera cho tới khi thấy vai, khuỷu,
cổ tay và hông trong khung hình.

Mặc định feature được tạo từ `angle_2d_smooth`. Có thể đổi nguồn góc, nhưng phải
giống nguồn đã dùng để tạo dataset train:

```powershell
python realtime_predict.py --angle-dimension angle_3d_smooth
```

Quy ước của tracker realtime hiện tại là: `start` = trung vị trước chuyển động,
`turning` = trung vị quanh điểm xa vị trí bắt đầu nhất, `rom` = max - min trong
repetition. Cần đối chiếu quy ước này với script tạo dataset gốc trước khi dùng
kết quả ngoài mục đích thử nghiệm.

## Đầu ra tọa độ của mỗi frame

`PoseDetector.get_landmarks(frame, normalize_pixel=True)` trả về, cho mỗi khớp:

- `pixel` — tọa độ điểm ảnh thật, dùng để vẽ và để tính góc 2D
- `coor_2d` — tọa độ chuẩn hóa `[0, 1]`
- `coor_3d` — tọa độ chuẩn hóa kèm `z`
- `visibility` — độ tin cậy `[0, 1]`, kiểm tra trước khi tính góc

`PoseDetector.get_world_landmarks()` trả về tọa độ 3D theo đơn vị mét, gốc tại
trung điểm hông.

## Hai lưu ý về hệ tọa độ

**Góc 2D tính trên pixel, không tính trên tọa độ chuẩn hóa.** Tọa độ chuẩn hóa
là `x = px/w, y = py/h`; với khung 1280×720 thì hai trục co giãn khác hệ số nhau
nên góc bị méo. Đo trên một bộ 3 điểm xiên bất kỳ, hai cách lệch nhau **22,19°**.

**Góc 3D lấy từ world landmarks, không lấy từ `coor_3d`.** Trường `coor_3d` trộn
`x, y` đã chuẩn hóa theo khung hình với `z` ở thang đo khác, không đồng nhất giữa
ba trục.

## Lọc nhiễu chuỗi góc

Mặc định bật. Trung vị cửa sổ trượt khử gai đột biến khi MediaPipe nhảy landmark,
rồi EMA làm mượt phần rung còn lại. Bộ lọc là nhân quả nên dùng được cho webcam,
đổi lại có độ trễ.

Đo trên `data/11.mp4` (video rung mạnh, tay vung nhanh, khớp khuỷu trái):

| `--smooth-window` | `--smooth-alpha` | Nhiễu trung bình | Nhiễu lớn nhất | Độ trễ @30fps |
|---|---|---|---|---|
| tắt lọc | — | 8,59° | 142,15° | 0 ms |
| 5 | 0,4 *(mặc định)* | 4,80° | 57,22° | 117 ms |
| 7 | 0,4 | 4,06° | 37,46° | 150 ms |
| 9 | 0,25 | 2,99° | 20,38° | 233 ms |

Mặc định `5 / 0,4` ưu tiên độ trễ thấp cho phản hồi real-time. Với video quay
sẵn, chất lượng kém, nên dùng `--smooth-window 9 --smooth-alpha 0.25`.

Giá trị **thô vẫn được giữ nguyên** trong file log bên cạnh giá trị đã lọc
(cột `*_2d` và `*_2d_smooth`), để còn đối chiếu mức nhiễu gốc khi phân tích.

## Giới hạn đã biết

- `arccos` chỉ cho góc 0–180° không dấu, nên chưa phân biệt được gập với duỗi
  quá mức (hyperextension), cũng không biết khớp đang mở ra hay đóng lại.
- Bộ lọc trung vị chỉ khử được gai kéo dài dưới `window // 2` frame. Khi
  MediaPipe bám sai liên tục nhiều frame (như quanh frame 105–110 của
  `data/11.mp4`), không bộ lọc nhân quả nào cứu được — phải cải thiện chất
  lượng video đầu vào.
- Khớp có `visibility` dưới ngưỡng bị bỏ qua, nên video chỉ quay nửa thân trên
  sẽ không có dữ liệu gối và cổ chân.
