# 패키지 설치
pip install -r ./requirements.txt
# 가중치 다운로드
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

# 데모 실행
python detection_demo.py