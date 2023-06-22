import cv2 # OpenCV 라이브러리
import numpy as np

# 창 초기화
canvas = np.ones((256, 256)) * 255  # 흰 배경의 캔버스
drawing = False  # 그리기 모드 여부 확인
last_point = (0, 0)  # 마지막 그리기 위치

# 마우스 이벤트 처리 함수
def draw(event, x, y, flags, param):
    global drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼을 누르면 그리기 모드 활성화
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼을 놓으면 그리기 모드 비활성화
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스를 이동하면서 그림 그리기
        if drawing:
            cv2.line(canvas, last_point, (x, y), (0, 0, 0), 2)  # 검정색 선으로 그림 그리기
            last_point = (x, y)

# OpenCV 창 생성 및 이벤트 핸들러 등록
cv2.namedWindow('Quick, Draw!')
cv2.setMouseCallback('Quick, Draw!', draw)

while True:
    cv2.imshow('Quick, Draw!', canvas)

    # 'q' 키를 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 's' 키를 누르면 그린 그림을 파일로 저장하고 이미지 분류 수행
    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = 'MyDrawing.jpg'
        cv2.imwrite(filename, canvas)
        break

cv2.destroyAllWindows()
