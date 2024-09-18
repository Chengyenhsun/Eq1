from flask import Flask, Response, render_template, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

# 定義 MJPG-Streamer 提供的 URL
url = "http://192.168.1.123:8080/?action=stream"

# 設定資料夾路徑
save_folder = "static/wafer"
result_folder = "static/result"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 初始化變數以追蹤狀態
last_detection_time = 0
photo_taken = False

# 載入自定義訓練好的模型
model = YOLO("best.pt")


def detect_black_object_edge_and_average_gray(frame):
    global last_detection_time, photo_taken

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=80, threshold2=200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mean_val = cv2.mean(gray, mask=mask)[0]

        threshold = 80
        current_time = time.time()

        if mean_val > threshold:
            status = "O"
            color = (0, 255, 0)  # 綠色

            # 如果是第一次偵測到 wafer，記錄偵測開始時間
            if last_detection_time == 0:
                last_detection_time = current_time

            # 檢查 wafer 片是否連續偵測超過 5 秒
            if current_time - last_detection_time >= 2:
                if not photo_taken:
                    background_mask = cv2.bitwise_not(mask)
                    white_background = np.ones_like(frame) * 255
                    frame_with_mask = cv2.bitwise_and(
                        white_background, white_background, mask=background_mask
                    )
                    frame_with_mask = cv2.bitwise_or(frame_with_mask, frame)
                    timestamp = time.strftime("%m%d%H%M%S")
                    filepath = os.path.join(save_folder, f"{timestamp}.jpg")
                    cv2.imwrite(filepath, frame_with_mask)
                    print(f"wafer detected, image saved to {filepath}")

                    # 模型預測
                    results = model.predict(frame_with_mask)
                    for i, result in enumerate(results):
                        result_filepath = os.path.join(
                            result_folder, f"result_{timestamp}_{i}.jpg"
                        )
                        result.save(result_filepath)
                        print(f"Recognition result saved to {result_filepath}")

                    # 標記已拍照
                    photo_taken = True

                    # 發送最新辨識結果的訊息
                    socketio.emit(
                        "new_result",
                        {
                            "latest_image": os.path.join(
                                result_folder, f"result_{timestamp}_0.jpg"
                            )
                        },
                    )
        else:
            status = "X"
            color = (0, 0, 255)  # 紅色
            photo_taken = False
            last_detection_time = 0  # 重置偵測時間

        cv2.drawContours(frame, [largest_contour], -1, color, 8)

    return frame


def generate_frames():
    cap = cv2.VideoCapture(url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_black_object_edge_and_average_gray(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/default.jpg")
def default_image():
    return send_file("default.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
