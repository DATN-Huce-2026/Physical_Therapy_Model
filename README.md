# AI_model

Nhận diện tư thế bằng MediaPipe Pose, tính góc khớp theo từng frame và xuất
chuỗi góc theo trục thời gian.

## Cấu trúc

| Tệp | Vai trò |
|---|---|
| `main.py` | Pipeline chính: nhận diện → tính góc → vẽ → ghi log |
| `export_keypoints.py` | Chỉ xuất tọa độ khớp thô ra JSON |
| `src/pose_detector.py` | Bọc MediaPipe Pose, trả landmark và vẽ skeleton/FPS |
| `src/angle_calculator.py` | Tính góc 2D/3D bằng tích vô hướng, vẽ cung tròn + số đo |
| `src/angle_smoother.py` | Lọc nhiễu chuỗi góc (trung vị + EMA) |
| `src/angle_logger.py` | Gom chuỗi góc theo thời gian, xuất CSV/JSON |
| `tests/` | Bộ test tự động (`python -m unittest discover -s tests -t .`) |

## Chạy

```bash
python main.py                          # webcam
python main.py --source data/11.mp4     # file video
python main.py --source data/20.mov --no-display   # chỉ xuất log, không mở cửa sổ
```

Kết quả ghi vào `angles/angles_<tên video>.csv` và `.json`.

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
