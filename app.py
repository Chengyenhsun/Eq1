from flask import Flask, Response, render_template, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# 定義 MJPG-Streamer 提供的 URL
# url = "http://192.168.1.123:8080/?action=stream"
url = "http://192.168.0.160:8080/?action=stream"

# 設定資料夾路徑
# save_folder = "static/wafer"
# result_folder = "static/result"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# if not os.path.exists(result_folder):
#     os.makedirs(result_folder)

# 初始化變數以追蹤狀態
last_detection_time = 0
photo_taken = False

# 載入自定義訓練好的模型
model = YOLO("best.pt")


def detect_black_object_edge_and_average_gray(frame):
    global last_detection_time, photo_taken

    # 影像處理和辨識邏輯
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
            color = (0, 255, 0)

            if last_detection_time == 0:
                last_detection_time = current_time

            if current_time - last_detection_time >= 2:
                if not photo_taken:
                    background_mask = cv2.bitwise_not(mask)
                    white_background = np.ones_like(frame) * 255
                    frame_with_mask = cv2.bitwise_and(
                        white_background, white_background, mask=background_mask
                    )
                    frame_with_mask = cv2.bitwise_or(frame_with_mask, frame)
                    print("偵測到wafer")
                    results = model.predict(frame_with_mask)

                    scratch_count = 0
                    stain_count = 0
                    for result in results:
                        for box in result.boxes:
                            if box.cls == 0:
                                scratch_count += 1
                            elif box.cls == 1:
                                stain_count += 1
                    print("辨識完成")
                    for result in results:
                        result_image = result.plot()
                        _, buffer = cv2.imencode(".jpg", result_image)
                        result_image_bytes = base64.b64encode(buffer).decode("utf-8")

                        socketio.emit(
                            "new_result",
                            {
                                "latest_image": result_image_bytes,
                                "scratch_count": scratch_count,
                                "stain_count": stain_count,
                            },
                        )
                        print("傳到前端")
                    photo_taken = True
                    socketio.emit("detection_status", "偵測到 Wafer，檢查缺陷中")

        else:
            status = "X"
            color = (0, 0, 255)
            photo_taken = False
            last_detection_time = 0
            # socketio.emit("detection_status", "未偵測到 Wafer")

        cv2.drawContours(frame, [largest_contour], -1, color, 8)

    return frame


def generate_frames():
    # 連接到影片串流並生成影像幀
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
    socketio.run(app, host="0.0.0.0", port=5003, debug=True)
