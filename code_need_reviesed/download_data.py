import requests

url = "http://192.168.1.40:30500/"
download_url = url + "download"
local_path = "/home/lixuan/project-1/dict_train_valid_y.pkl.gz"
data = requests.post(download_url, json={"file_path": "D:\lixuan\data\dict_train_valid_y.pkl.gz"})
with open(local_path, "wb") as f:
    f.write(data.content)










