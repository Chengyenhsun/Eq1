from flask import Flask, Response, render_template, send_file
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO  # 引入YOLO模型

app = Flask(__name__)

# 定義 MJPG-Streamer 提供的 URL
# url = "http://192.168.0.160:8080/?action=stream"
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

    # 轉換為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 應用高斯模糊來降低噪點
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 邊緣檢測器
    edges = cv2.Canny(blurred, threshold1=80, threshold2=200)

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

        # 設定閾值判斷是否有wafer片
        threshold = 80
        if mean_val > threshold:
            status = "O"
            color = (0, 255, 0)  # 綠色
            current_time = time.time()

            # 如果偵測到 wafer
            if not photo_taken:
                if current_time - last_detection_time >= 2:  # 持續偵測 2 秒
                    # 創建遮罩以外的部分
                    background_mask = cv2.bitwise_not(mask)

                    # 將背景部分設為白色
                    white_background = np.ones_like(frame) * 255
                    frame_with_mask = cv2.bitwise_and(
                        white_background, white_background, mask=background_mask
                    )

                    # 將原始影像與白色背景進行合併
                    frame_with_mask = cv2.bitwise_or(frame_with_mask, frame)

                    # 拍照並保存原始影像
                    timestamp = time.strftime("%m%d%H%M%S")  # 格式化時間為月日時分秒
                    filepath = os.path.join(save_folder, f"{timestamp}.jpg")
                    cv2.imwrite(filepath, frame_with_mask)
                    print(f"wafer detected, image saved to {filepath}")

                    # 進行 YOLO 模型預測
                    results = model.predict(frame_with_mask)

                    # 儲存辨識後的影像結果
                    for i, result in enumerate(results):
                        result_filepath = os.path.join(
                            result_folder, f"result_{timestamp}_{i}.jpg"
                        )
                        result.save(result_filepath)
                        print(f"Recognition result saved to {result_filepath}")

                    # 標記已拍照
                    photo_taken = True
        else:
            status = "X"
            color = (0, 0, 255)  # 紅色
            # 如果未檢測到 wafer，重置狀態
            photo_taken = False
            last_detection_time = time.time()  # 重置最後偵測時間

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

        # 偵測黑色物體邊緣並計算灰階平均值和wafer狀態
        frame = detect_black_object_edge_and_average_gray(frame)

        # 將影像轉換為 JPEG 格式
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # 生成 MJPEG 格式的幀
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    # 渲染 index.html 模板，顯示即時影像和照片
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # 提供視頻流
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_latest_result")
def get_latest_result():
    # 檢查 result 資料夾中的最新結果檔案
    result_files = [f for f in os.listdir(result_folder) if f.endswith(".jpg")]
    if result_files:
        latest_file = max(
            result_files, key=lambda f: os.path.getctime(os.path.join(result_folder, f))
        )
        latest_image_url = os.path.join(result_folder, latest_file)
    else:
        # 如果沒有檔案，顯示預設圖片
        latest_image_url = "/default.jpg"

    return {"latest_image": latest_image_url}


if __name__ == "__main__":
    # 啟動 Flask 應用，並設置為 debug 模式
    app.run(host="0.0.0.0", port=5001, debug=True)
