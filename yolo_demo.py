from ultralytics import YOLO

# 載入自定義訓練好的模型
model = YOLO("best3.pt")

# 進行預測
results = model.predict("image_without_black_circle.jpg")

# 若結果是列表，則處理每個結果
if isinstance(results, list):
    for i, result in enumerate(results):
        result.save(f"output_{i}.jpg")  # 儲存每個結果到不同的檔案
else:
    results.save("output.jpg")  # 若結果不是列表，直接儲存為 output.jpg

# 若要確認儲存結果成功，可以印出結果的檔案名稱
print("Predicted image(s) saved.")
