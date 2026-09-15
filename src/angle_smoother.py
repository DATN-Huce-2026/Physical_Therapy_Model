from collections import deque
from statistics import median


class AngleSmoother:
    """Lọc nhiễu chuỗi góc theo thời gian, chạy được real-time.

    Hai tầng nối tiếp nhau, mỗi tầng trị một loại nhiễu khác nhau:

    1. Median trên cửa sổ trượt - khử GAI đột biến. Khi MediaPipe nhảy landmark
       một frame (ảnh mờ do chuyển động nhanh), góc vọt vài chục tới hơn trăm
       độ rồi lập tức về chỗ cũ. Trung bình cộng bị kéo theo, còn trung vị thì
       loại hẳn giá trị lạc đàn.
    2. EMA (trung bình trượt lũy thừa) - làm mượt phần rung biên độ nhỏ còn lại
       sau bước 1.

    Bộ lọc là NHÂN QUẢ (chỉ dùng frame quá khứ) nên dùng được cho luồng webcam,
    đổi lại có độ trễ khoảng (window // 2) + 1/alpha frame.
    """

    def __init__(self, window=5, alpha=0.4, max_gap=5):
        if window < 1:
            raise ValueError("window phải >= 1")
        if not 0 < alpha <= 1:
            raise ValueError("alpha phải nằm trong khoảng (0, 1]")

        self.window = window
        self.alpha = alpha
        self.max_gap = max_gap

        self._buffers = {}
        self._ema = {}
        self._last_frame = {}

    def reset(self, key=None):
        """Xóa trạng thái của một khớp, hoặc toàn bộ nếu không truyền key."""
        if key is None:
            self._buffers.clear()
            self._ema.clear()
            self._last_frame.clear()
            return

        self._buffers.pop(key, None)
        self._ema.pop(key, None)
        self._last_frame.pop(key, None)

    def update(self, key, value, frame_id):
        """Đưa vào một giá trị thô, nhận lại giá trị đã lọc (hoặc None).

        Khi khớp mất dấu quá max_gap frame, trạng thái được xóa: nối tiếp giá
        trị cũ qua một quãng dài mất dấu sẽ tạo ra chuyển động giả không có
        thật trong video.
        """
        last = self._last_frame.get(key)

        if last is not None and frame_id - last > self.max_gap:
            self.reset(key)

        if value is None:
            return None

        buffer = self._buffers.setdefault(key, deque(maxlen=self.window))
        buffer.append(value)

        # Median chỉ có nghĩa khi đủ số mẫu; trước đó dùng thẳng giá trị thô
        # để không phải chờ đầy cửa sổ mới có đầu ra.
        filtered = median(buffer)

        previous = self._ema.get(key)
        if previous is None:
            smoothed = filtered
        else:
            smoothed = self.alpha * filtered + (1 - self.alpha) * previous

        self._ema[key] = smoothed
        self._last_frame[key] = frame_id

        return smoothed

    def apply(self, angles, frame_id, dimensions=("angle_2d", "angle_3d")):
        """Bổ sung khóa *_smooth vào kết quả của AngleCalculator.compute().

        Giá trị thô được GIỮ NGUYÊN bên cạnh giá trị đã lọc, để khi phân tích
        còn đối chiếu được mức nhiễu gốc.
        """
        for joint_name, data in angles.items():
            for dim in dimensions:
                key = (joint_name, dim)

                raw = data.get(dim) if data else None
                smoothed = self.update(key, raw, frame_id)

                if data is not None:
                    data[f"{dim}_smooth"] = smoothed

        return angles
