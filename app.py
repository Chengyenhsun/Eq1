from flask import Flask, Response, render_template
import cv2
import numpy as np

app = Flask(__name__)

# 定義 MJPG-Streamer 提供的 URL
url = "http://192.168.0.160:8080/?action=stream"


def detect_black_object_edge_and_average_gray(frame):
    # 轉換為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 應用高斯模糊來降低噪點
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 邊緣檢測器
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 查找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾出最大的輪廓，假設這是黑色圓盤的輪廓
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 創建一個與原始影像同樣大小的遮罩
        mask = np.zeros_like(gray)

        # 在遮罩上繪製輪廓
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 使用遮罩計算輪廓內的灰階值平均
        mean_val = cv2.mean(gray, mask=mask)[0]  # 取得灰階平均值

        # 設定閾值判斷是否有CD片
        threshold = 80
        if mean_val > threshold:
            status = "O"
            color = (0, 255, 0)  # 綠色
        else:
            status = "X"
            color = (0, 0, 255)  # 紅色

        # 在影像上顯示平均值和狀態
        # cv2.putText(
        #     frame,
        #     f"Mean Gray: {mean_val:.2f}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        # )
        # cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 繪製輪廓
        cv2.drawContours(frame, [largest_contour], -1, color, 8)

    return frame


def generate_frames():
    # 創建 VideoCapture 物件
    cap = cv2.VideoCapture(url)

    while True:
        # 從視頻流中讀取幀
        success, frame = cap.read()
        if not success:
            break

        # 偵測黑色物體邊緣並計算灰階平均值和CD狀態
        frame = detect_black_object_edge_and_average_gray(frame)

        # 將影像轉換為 JPEG 格式
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # 生成 MJPEG 格式的幀
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
