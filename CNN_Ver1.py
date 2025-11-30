import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 데이터 전처리 및 로드
# CIFAR-10 데이터셋을 로드하고 학습 데이터에 다양한 증강 기법을 적용하여 일반화 성능을 높임
# 정규화(Normalization)는 학습 데이터와 테스트 데이터에 동일하게 적용
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 랜덤 크롭으로 이미지 다양성 확보
    transforms.RandomHorizontalFlip(),  # 이미지를 수평으로 뒤집어 데이터 증강
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 왜곡 추가
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 평균 및 표준편차로 정규화
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 랜덤 영역 삭제로 일반화 성능 향상
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # 테스트 데이터는 증강 없이 텐서로 변환만 수행
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 정규화만 적용
])

# CIFAR-10 데이터셋 다운로드 및 로드
# 학습 데이터 (train=True)와 테스트 데이터 (train=False)를 로드
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 학습 데이터와 검증 데이터를 80:20 비율로 분리하여 검증 데이터로 모델 성능을 평가
train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_train_dataset, train_indices)  # 학습 데이터 서브셋
val_dataset = Subset(full_train_dataset, val_indices)  # 검증 데이터 서브셋

# 데이터로더(DataLoader)를 사용하여 데이터를 배치 단위로 불러옴
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 학습 데이터 로더
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # 검증 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 테스트 데이터 로더

# 2. CNN 모델 정의
# CNN 모델은 합성곱 계층과 Fully Connected 계층으로 구성되며, 배치 정규화와 드롭아웃을 포함하여 학습 안정성과 일반화 성능을 향상
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 합성곱 계층 정의
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 입력 채널: 3, 출력 채널: 64
            nn.BatchNorm2d(64),  # 배치 정규화로 학습 안정화
            nn.ReLU(),  # 비선형 활성화 함수
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 출력 채널: 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 크기를 절반으로 줄이는 맥스풀링
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 출력 채널: 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 크기 절반 감소
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 출력 채널: 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 최종 크기: 4x4
        )
        # Fully Connected 계층 정의
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 텐서를 1차원으로 변환
            nn.Linear(512 * 4 * 4, 1024),  # 입력 크기: 512*4*4, 출력 크기: 1024
            nn.ReLU(),
            nn.Dropout(0.4),  # 과적합 방지를 위한 Dropout
            nn.Linear(1024, 512),  # 출력 크기: 512
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)  # CIFAR-10의 10개 클래스에 대한 확률 출력
        )
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x):
        # 모델의 순전파 정의
        x = self.conv_layers(x)  # 합성곱 계층 통과
        return self.fc_layers(x)  # Fully Connected 계층 통과

    def _init_weights(self, m):
        # 가중치 초기화: Xavier 초기화를 사용하여 학습 안정성 향상
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

# 3. 학습 및 검증
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부 확인
print(f"Using device: {device}")

# 모델 초기화 및 학습 설정
model = CNN().to(device)  # 모델을 GPU/CPU로 이동
criterion = nn.CrossEntropyLoss()  # 손실 함수: 다중 클래스 분류를 위한 Cross-Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW 옵티마이저 사용
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=40  # 학습률 스케줄링 에포크 설정
)

# 학습 파라미터
epochs = 40
train_acc_history, val_acc_history = [], []  # 정확도 기록
best_val_acc, early_stop_counter, early_stop_limit = 0, 0, 5  # Early Stopping 설정

# 학습 루프
for epoch in range(1, epochs + 1):
    model.train()  # 모델 학습 모드 설정
    train_correct, train_total = 0, 0  # 학습 정확도 초기화

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 데이터를 장치로 이동
        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(images)  # 모델 출력
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        scheduler.step()  # 학습률 스케줄링

        _, predicted = outputs.max(1)  # 예측값 (가장 높은 확률)
        train_correct += (predicted == labels).sum().item()  # 올바른 예측 수
        train_total += labels.size(0)  # 총 데이터 수

    train_acc = train_correct / train_total  # 학습 정확도 계산
    train_acc_history.append(train_acc)

    # 검증 단계
    model.eval()  # 모델 검증 모드 설정
    val_correct, val_total = 0, 0

    with torch.no_grad():  # 검증 중 그래디언트 계산 비활성화
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total  # 검증 정확도 계산
    val_acc_history.append(val_acc)

    # Early Stopping 조건 확인
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), 'CNN_v1.pth')  # 최적 모델 저장
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_limit:
        print(f"조기 종료 : {epoch}")
        break

    print(f"[Epoch {epoch}/{epochs}] Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

# 학습 결과 시각화
plt.figure()
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.savefig('CNN_v1.png', dpi=100)
plt.show()

# 테스트 데이터 평가
model.load_state_dict(torch.load('CNN_v1.pth'))  # 최적 모델 로드
model.eval()  # 평가 모드 설정
all_labels, all_predictions = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 테스트 성능 보고
print("\n최종 테스트 결과:")
print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))
overall_accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
print(f"전반적인 테스트 정확도: {overall_accuracy:.4f}")
