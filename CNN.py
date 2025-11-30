import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 데이터 전처리 및 로드
# 학습 데이터에 랜덤 크롭, 수평 뒤집기, 정규화 적용
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 이미지 가장자리를 패딩 후 랜덤으로 자르기 (데이터 증강)
    transforms.RandomHorizontalFlip(),  # 이미지를 50% 확률로 수평으로 뒤집기 (데이터 증강)
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 평균 및 표준편차로 정규화
])

# 테스트 데이터는 데이터 증강 없이 정규화만 적용
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 동일한 정규화 적용
])

# CIFAR-10 데이터셋 다운로드 및 로드
# train=True는 학습 데이터를, train=False는 테스트 데이터를 로드
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 학습 데이터와 검증 데이터를 80:20 비율로 분리 (검증 데이터는 모델 성능 확인용)
train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_train_dataset, train_indices)  # 학습 데이터 서브셋
val_dataset = Subset(full_train_dataset, val_indices)  # 검증 데이터 서브셋

# 데이터로더 정의 (미니 배치로 데이터를 로드)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 학습 데이터 로더, 무작위로 데이터 섞기
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # 검증 데이터 로더, 순차적 로드
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 테스트 데이터 로더, 순차적 로드

# 2. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 합성곱 계층과 활성화 함수, 풀링 계층을 포함한 네트워크 정의
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 첫 번째 합성곱 계층 (3채널 -> 32채널)
            nn.ReLU(),  # 활성화 함수: ReLU
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 두 번째 합성곱 계층 (32채널 -> 64채널)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 최대 풀링으로 크기 축소
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 세 번째 합성곱 계층 (64채널 -> 128채널)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 최대 풀링
        )
        # 완전 연결 계층 정의
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 2D 텐서를 1D 벡터로 변환
            nn.Linear(128 * 8 * 8, 256),  # 완전 연결 계층 (입력: 128*8*8, 출력: 256)
            nn.ReLU(),
            nn.Dropout(0.5),  # 과적합 방지를 위한 드롭아웃
            nn.Linear(256, 128),  # 두 번째 완전 연결 계층
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # CIFAR-10 클래스(10개) 출력
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 합성곱 계층 통과
        x = self.fc_layers(x)  # 완전 연결 계층 통과
        return x

# 3. 학습 및 검증
# 학습에 GPU 사용 (가능하면), 없으면 CPU 사용
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델 초기화
model = CNN().to(device)
# 손실 함수: 교차 엔트로피 (다중 클래스 분류)
criterion = nn.CrossEntropyLoss()
# 옵티마이저: Adam (학습 속도와 수렴 속도 개선)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 관련 파라미터 설정
epochs = 20
train_acc_history = []  # 에포크별 학습 정확도 기록
val_acc_history = []  # 에포크별 검증 정확도 기록

# 학습 루프
for epoch in range(1, epochs + 1):
    model.train()  # 모델을 학습 모드로 설정
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:  # 미니 배치 단위로 데이터 로드
        images, labels = images.to(device), labels.to(device)  # 데이터 장치로 이동 (GPU/CPU)

        optimizer.zero_grad()  # 이전 단계의 그래디언트 초기화
        outputs = model(images)  # 모델 예측
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파 단계
        optimizer.step()  # 가중치 업데이트

        # 예측 결과 계산
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스를 선택
        train_correct += (predicted == labels).sum().item()  # 맞춘 샘플 수 누적
        train_total += labels.size(0)  # 전체 샘플 수 누적

    # 학습 정확도 계산
    train_acc = train_correct / train_total
    train_acc_history.append(train_acc)

    # 검증 단계
    model.eval()  # 모델을 평가 모드로 설정
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # 그래디언트 계산 비활성화 (평가 단계)
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    # 검증 정확도 계산
    val_acc = val_correct / val_total
    val_acc_history.append(val_acc)

    # 에포크별 결과 출력
    print(f"[Epoch {epoch}/{epochs}] Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    print("-" * 50)

# 학습 및 검증 정확도 시각화
epochs_range = range(1, epochs + 1)

plt.figure()
plt.plot(epochs_range, train_acc_history, label='Train Accuracy')  # 학습 정확도 그래프
plt.plot(epochs_range, val_acc_history, label='Validation Accuracy')  # 검증 정확도 그래프
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.savefig('CNN.png')  # 그래프 저장
plt.show()

# 테스트 데이터 평가
model.eval()  # 평가 모드로 전환
all_labels = []
all_predictions = []

with torch.no_grad():  # 테스트 단계는 그래디언트 계산 불필요
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())  # 실제 레이블 저장
        all_predictions.extend(predicted.cpu().numpy())  # 예측 결과 저장

# 분류 보고서 출력
print("\n최종 결과 보고서:")
print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))

# 전체 정확도 계산 및 출력
correct_predictions = sum(p == l for p, l in zip(all_predictions, all_labels))  # 올바른 예측 수
total_samples = len(all_labels)  # 전체 샘플 수
overall_accuracy = correct_predictions / total_samples
print(f"전반적인 테스트 정확도: {overall_accuracy:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'CNN.pth')
print("모델 저장 : 'CNN.pth'")
