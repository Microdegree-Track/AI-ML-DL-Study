# Computer Vision Projects

두 개의 과제를 통해 기본적인 CNN 이미지 분류부터 YOLO 기반 객체 인식까지 실습한 프로젝트입니다.

- Project 1: Fashion-MNIST 이미지 분류 (간단 CNN, 중간고사)
- Project 2: YOLOv5 마스크 착용 상태 인식 (기말고사)

---

## Project 1. Fashion-MNIST 이미지 분류 (Simple CNN)

### 1. Overview
간단한 컨볼루션 신경망(CNN)을 사용하여 Fashion-MNIST 의류 이미지(28×28, grayscale)를 10개 클래스(티셔츠, 바지, 드레스 등)로 분류하는 프로젝트입니다.

### 2. 주요 내용

- 데이터 로드 및 전처리  
  - `t10k-images-idx3-ubyte.gz` / `t10k-labels-idx1-ubyte.gz` 파일 로드  
  - 이미지 정규화 및 텐서 변환  
  - 학습/검증/테스트 세트 분리  

- 데이터 탐색 및 시각화  
  - 여러 샘플 이미지를 그리드 형태로 시각화  
  - 클래스별 예시 이미지 확인  

- CNN 모델 구축  
  - 최소 1개 이상의 `Conv2d + MaxPool2d` 블록 사용  
  - `ReLU` 활성화, `Dropout` 등을 포함한 간단한 아키텍처 설계  

- 모델 학습 및 모니터링  
  - 손실 함수: `CrossEntropyLoss`  
  - 최적화 알고리즘: `Adam` 또는 `SGD`  
  - Epoch별 Train Loss / Accuracy 로그 기록 및 그래프 시각화  

- 모델 평가 및 오류 분석  
  - Test 데이터셋으로 최종 정확도 측정  
  - 잘못 분류된 이미지 시각화  
  - 혼동 행렬(Confusion Matrix)을 통해 어떤 클래스에서 오분류가 많이 발생하는지 분석  

### 3. 기술 스택

- Python  
- PyTorch  
- Jupyter Notebook (Visual Studio Code)  
- Matplotlib / Seaborn (시각화)

---

## Project 2. YOLOv5 마스크 착용 상태 인식

### 1. Overview
YOLOv5 기반 객체 인식 모델을 활용하여 사람 얼굴 영역을 탐지하고, 마스크 착용 상태를 3가지 클래스로 분류하는 프로젝트입니다.

- Class 0: 마스크 잘 착용  
- Class 1: 마스크 미착용  
- Class 2: 마스크 잘못 착용  

총 801장의 얼굴 이미지에 대해 라벨링된 데이터셋을 직접 `train : val : test = 6 : 2 : 2` 비율로 분할하여 학습과 평가를 수행합니다.

### 2. 데이터셋 구성

- 입력 데이터: `face_mask` 폴더 내 이미지와 라벨 파일  
- 바운딩 박스 + 클래스 라벨 형식 (YOLO 포맷)  
- 사용자 정의 `data.yaml` 작성  
  - `train`, `val`, `test` 경로 설정  
  - `nc: 3`, `names: ["mask_good", "no_mask", "mask_incorrect"]` 등 정의  

### 3. YOLOv5 모델 훈련

- 사용 모델: `yolov5s` (또는 실험적으로 다른 사이즈 모델)  
- 주요 설정  
  - Epoch 수, Batch size, Learning rate 등 하이퍼파라미터 튜닝  

- 모니터링 및 로그  
  - Loss (box, obj, cls) 변화 추이 그래프  
  - mAP@0.5, Precision, Recall 곡선 시각화  
  - Best checkpoint 선택 및 저장  

### 4. 객체 인식 및 시각화

- 학습된 가중치를 이용한 추론용 노트북 작성 (Jupyter Notebook 기반)  
- OpenCV를 활용하여  
  - 검출된 얼굴 영역에 Bounding Box 표시  
  - 클래스별 색상과 라벨 텍스트(0, 1, 2 → 의미 있는 텍스트로 매핑) 출력  


### 5. 결과 분석 및 개선 아이디어

- 성능 분석  
  - mAP@0.5, Precision, Recall, F1-score 정리  
  - 클래스별 성능 비교 (예: 잘못 착용 vs 미착용 구분 난이도)  

- 오류 사례 분석  
  - 조명 변화  
  - 얼굴 일부 가려짐  
  - 작은 얼굴 영역  
  - 마스크 색상, 패턴 등으로 인한 오검출/미검출 사례 분석  

- 개선 방안 제시  
  - 데이터 증강 강화  
    - 회전, 좌우 반전, 밝기·대비 변화, 랜덤 크롭 등  
  - 입력 해상도 변경 및 앵커 튜닝  
  - 더 큰 YOLO 모델 사용 (예: `yolov5m`, `yolov5l`)  
  - 실제 CCTV, 웹캠 영상 기반 도메인 적응(domain adaptation) 고려  

### 6. 기술 스택

- Python  
- YOLOv5 (Ultralytics)  
- PyTorch  
- OpenCV  
- Jupyter Notebook  
