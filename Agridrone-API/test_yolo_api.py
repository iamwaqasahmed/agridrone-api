import requests

# Path to your 4-channel TIFF image
file_path = r"D:\AgriTech\Agri-Drone\fc_regress_train\train\images\IMG_0009.tif"

# API endpoint
url = "https://agri-tech-testing-pipeline-api.hf.space/predict_yolo/"

# Send the POST request with the image file
with open(file_path, "rb") as f:
    files = {"file": (file_path.split("\\")[-1], f, "image/tiff")}
    response = requests.post(url, files=files)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
