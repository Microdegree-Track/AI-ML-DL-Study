import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random

# 1. 데이터 전처리 및 증강 
# Mixup과 CutMix 기법을 적용하는 데이터 증강 클래스
class MixupCutMixDataLoader:
    """
    Mixup과 CutMix 데이터 증강 기법을 적용하는 클래스.
    - Mixup: 두 데이터를 랜덤 가중치로 섞음.
    - CutMix: 이미지의 일부를 다른 데이터의 일부로 대체.
    """
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha  # 베타 분포의 파라미터
        self.prob = prob  # Mixup 또는 CutMix를 적용할 확률

    def mixup_data(self, x, y):
        """
        Mixup 기법 구현.
        x: 입력 이미지, y: 라벨
        """
        lam = np.random.beta(self.alpha, self.alpha)  # Mixup 비율
        batch_size = x.size(0)
        index = torch.randperm(batch_size)  # 데이터를 섞을 인덱스
        mixed_x = lam * x + (1 - lam) * x[index, :]  # 이미지 섞기
        y_a, y_b = y, y[index]  # 라벨 섞기
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y):
        """
        CutMix 기법 구현.
        x: 입력 이미지, y: 라벨
        """
        lam = np.random.beta(self.alpha, self.alpha)  # Mixup 비율
        batch_size, _, h, w = x.size()
        index = torch.randperm(batch_size)
        cx, cy = np.random.randint(w), np.random.randint(h)  # 패치 중심
        rw, rh = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))  # 패치 크기
        x[:, :, cy:cy+rh, cx:cx+rw] = x[index, :, cy:cy+rh, cx:cx+rw]  # 패치 교체
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def __call__(self, x, y):
        """
        Mixup 또는 CutMix를 적용 (확률적으로 선택).
        """
        if random.random() < self.prob:
            if random.random() < 0.5:
                return self.mixup_data(x, y)
            else:
                return self.cutmix_data(x, y)
        return x, y, y, 1.0  # 원본 데이터 반환

# 데이터 전처리 및 증강
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 랜덤 크롭
    transforms.RandomHorizontalFlip(),  # 수평 뒤집기
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 왜곡
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 정규화
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 테스트 데이터 정규화
])

# CIFAR-10 데이터셋 로드
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 학습/검증 데이터 분리
train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Mixup 및 CutMix 클래스 생성
mixcut = MixupCutMixDataLoader(alpha=0.5, prob=0.3)

# 2. CNN 모델 정의
# 간단한 CNN 모델 정의
class CNN(nn.Module):
    """
    배치 정규화와 드롭아웃을 포함한 CNN 모델 정의.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        순전파 정의.
        """
        x = self.conv_layers(x)
        return self.fc_layers(x)

# 3. 학습 및 검증 설정 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부 확인
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 손실 함수
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)  # AdamW 옵티마이저

# 학습률 스케줄러 설정
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.05,  # 학습 중 최대 학습률
    epochs=50,  # 총 학습 에포크 수
    steps_per_epoch=len(train_loader)  # 한 에포크의 배치 수
)

# 4. 학습 루프 
epochs = 50
best_val_acc = 0
train_acc_history, val_acc_history = [], []  # 학습/검증 정확도 기록

for epoch in range(1, epochs + 1):
    model.train()
    train_correct, train_total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixcut(images, labels)  # Mixup/CutMix 적용

        optimizer.zero_grad()
        outputs = model(images)
        # Mixup/CutMix 손실 계산
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)  # 예측값 계산
        # Mixup/CutMix에서는 정확도 계산이 부정확할 수 있음 (섞인 라벨로 인해 실제와 불일치)
        train_correct += lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    train_acc_history.append(train_acc)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_acc_history.append(val_acc)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'CNN_v2.pth')  # 최적 모델 저장

    print(f"[Epoch {epoch}/{epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# 5. 결과 시각화 및 평가
plt.style.use('seaborn-v0_8-darkgrid')
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('CNN_v2.png')
plt.show()

# 최적 모델 로드 및 테스트 데이터 평가
model.load_state_dict(torch.load('CNN_v2.pth'))
model.eval()
all_labels, all_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 테스트 성능 보고
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

overall_accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"전반적인 테스트 정확도: {overall_accuracy:.4f}")
