# 라이브러리 임포트
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup

#======(1)모델 설계==========================================================================================================================

# 신경망 아키텍처를 정의하는 PyTorch의 'nn.Module'을 상속한 'Net'클래스.
# 이미지 분류를 위한 합성곱 신경망(CNN)구조를 구성하고, 완전 연결 레이어로 신경망을 완성한다.
# 각 레이어는 입력과 출력 크기를 기반으로 정의되며, 입력 데이터의 차원에 따라 조정되어야 한다.
class Net(nn.Module):
    # 파이썬 클래스 초기화 함수 _ Net 클래스의 생성자 메서드 정의.
    def __init__(self):
        super(Net, self).__init__() # Net클래스의 부모클래스인 'nn.Module'의 생성자 메서드 호출.

        # 1) filter 연산 네트워크
        self.conv1 = nn.Conv2d(3, 6, 5) # 첫번째 합성곱 레이어를 정의하는 객체.(입력 채널 수는 3,출력 채널 수는 6, 합성곱 커널의 크기는 5x5)
        self.pool = nn.MaxPool2d(2,2) # 최대 풀링 레이어를 정의하는 객체.(풀링 영역의 크기는 2x2, 스트라이드는 2이다.)
        self.conv2 = nn.Conv2d(6, 16, 5) # 두번째 합성곱 레이어를 정의하는 객체.(입력 채널 수 6, 출력 채널 수 16, 합성곱 커널의 크기는 5x5)

        # 2) full connected layer 네트워크
        self.fc1 = nn.Linear(400,120) # 첫번째 FullConnected 레이어를 정의하는 객체.(입력크기는 400, 출력크기는 120)
        self.fc2 = nn.Linear(120,84) # 두번째 FullConnected 레이어를 정의하는 객체.(입력크기는 120, 출력크기는 84)
        self.fc3 = nn.Linear(84,2) # 세번째 FullConnected 레이어를 정의하는 객체.(입력크기는 84, 출력크기는 10) 마지막 출력크기가 최종 분류 개수.

    # 위에서 정의한 네트워크들이 적용되는 순서를 정의.
    # 신경망의 순전파(forward pass)연산을 정의하는 함수, x는 입력 데이터.
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) # 입력 데이터'x'를 첫번째 합성곱 레이어에 통과.
        x = self.pool(F.relu(self.conv2(x))) # F.relu()함수는 ReLU(Activation Function)을 적용하여 반환된 값을 비선형성으로 변환.
        # self.pool은 최대 풀링 레이어를 의미하며, 괄호 안의 결과에 풀링을 적용하여 반환.

        x = x.view(-1, 400) # x를 2차원 텐서로 변환.
        # -1 : 남은 차원을 자동 조정하라는 의미.
        # 400 : x의 형상을 '(배치 크기(차원), 400)'으로 변환.

        x = F.relu(self.fc1(x)) # x를 첫번째 완전연결레이어에 통과.
        x = F.relu(self.fc2(x)) # 위의 합성곱 레이어 연산과 비슷.(풀링x)
        x = self.fc3(x) # x를 세번째 완전연결레이어에 통과.

        return x # 최종 결과 x 반환.

# ======<'Net' Class Finished>==========================================================================================================================

# 이미지를 화면에 표시하기 위한 함수.
# 인수를 넘파이 배열로 변환 후 Matplotlib를 이용하여 화면에 표시하는 기능.
def imshow(img): # img : 이미지를 나타내는 텐서.
    img = img / 2 + 0.5 # unnormalize(비정규화) : 이미지의 값을 원래 범위로 되돌리는 역할.
    npimg = img.numpy() # img텐서를 넘파이 배열로 변환.
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # plt.imshow()함수를 사용하여 이미지 표시.
    # npimg를 (1,2,0)의 순서(RGB축)로 축을 변환하여 표시.
    plt.show() # 이미지를 화면에 실제로 표시.

#======< 메인함수 >==========================================================================================================================

def main():
    #필수 코드 (1) 데이터 로드 & 가공

    # 데이터 변환(transform)을 위한 객체 정의. (크기조정, 텐서형식으로변환, 정규화)
    transform = transforms.Compose( # 객체 생성.
        [
            transforms.Resize((32, 32)), # 이미지 크기를 (32,32)로 조정하는 변환 정의.이거 시파10은 없음.
            transforms.ToTensor(),# 이미지를 텐서 형식으로 변환 정의.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 이미지를 정규화 하는 변환 정의. # 평균 값과 표준 편차를 지정하여 이미지 정규화.
        ])

    #===< 데이터 로드 >=========================================================================================================================
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 이미지 폴더에서 데이터셋 로드하기. 이 부분을 quickdraw 라이브러리에서 데이터셋 가져오게 바꾸기. 위에 시파10 참고.
    # transform : 앞서 정의한 변환 객체 'transform' 적용.
    trainset = torchvision.datasets.ImageFolder("./data", transform=transform)
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            #download=True, transform=transform)




    #2) 로드한 데이터를 iterator로 변환
    # 위에서 로드해 변환한 데이터셋을 미니배치로 나누고 반복 가능한 데이터로 변환.
    # batch_size : 미니배치의 크기. 즉, 한 번에 하나의 데이터 샘플씩 가져옴.
    # shuffle : 데이터를 섞을지 여부.
    # num_workers : 데이터 로딩을 위한 병렬 작업자의 수.
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                              shuffle=True, num_workers=2)




    #테스트 시킬 데이터 셋. 테스트 셋도 마찬가지.
    '''
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    '''

    #testset = torchvision.datasets.ImageFolder("./data", transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)



    classes = ('mermaid','panda') # 클래스 이름을 나타내는 튜플 정의.
    #=========================================================================================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ========================================
    # get some random training images
    # for문 돌리기 위한 iterator 생성
    '''
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''
    # =========================================

    # =========================================
    # 2) 네트워크 생성
    # 앞서 정의한 Net클래스의 인스턴스를 생성하여 신경망 모델 'net' 만듬.
    net = Net().to(device) # 모델을 지정한 장치(device)로 이동.
    # net = Net() # 장치 지정 안하고 CPU를 사용하고자 할 때 사용.

    print(net) # 생성된 신경망 모델을 출력(모델의 구조, 매개변수에 대한 정보 표시).

    # 3) loss 함수
    # loss함수로 크로스 앤트로피 Loss를 사용.
    # 다중 클래스 분류 문제에서 자주 사용되는 손실함수.
    criterion = nn.CrossEntropyLoss()

    # 4) activation functoin 함수
    # 확률적 경사 하강법(SGD) 옵티마이저 사용, 계산된 그래디언트에 기반하여 신경망의 매개변수를 업데이트.
    # lr : Learning Rate는 SGD 옵티마이저가 매개변수를 업데이트할 때 적용하는 스텝의 크기를 결정하는 하이퍼파라미터이다.
    # momentum : SGD 옵티마이저의 한 종류로, 이전 그래디언트 업데이트의 방향을 기억하여 그래디언트의 발산을 줄이고 수렴 속도를 향상시키는 데 도움을 준다.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    # 5) 학습 코드
    # epoch는 전체 데이터셋을 완전히 통과하는 것을 의미한다.
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0 # 각 epoch에서 훈련 중인 손실을 추적하기 위해 변수'running_loss'를 초기화.
        for i, data in enumerate(trainloader, 0):# 훈련 데이터셋의 미니 배치를 반복하는 루프를 시작. enumerate : trainloader에서 인덱스i와 해당하는 data를 반환.
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device) # data 튜플에서 입력과 레이블을 가져와 device 변수에 지정된 장치로 이동한다.

            # print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad() # 모든 최적화된 매개변수의 그래디언트를 초기화(0으로), 새로운 미니 배치에 대한 그래디언트 계산을 수행하기 전에 이 작업을 수행해야함.

            # forward + backward + optimize
            outputs = net(inputs) # 'inputs'를 신경망 'net'에 입력하고 예측된 출력을 얻음.
            loss = criterion(outputs, labels) # 앞서 정의한 criterion을 사용하여, 예측된 출력과 실제 labels 사이의 손실을 계산.
            loss.backward() # 역전파를 수행하여 손실에 대한 매개변수의 그래디언트를 계산.
            optimizer.step() # 계산된 그래디언트와 설정된 학습률 및 모멘텀을 사용하여 신경망의 매개변수를 업데이트.

            # print statistics
            running_loss += loss.item() # 현재 미니 배치에 대한 손실 값을 running_loss 변수에 누적.

            # 이전 2000개의 미니 배치에 대한 평균 손실 값을 출력하여 훈련 진행 상황을 제공.
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    ########################################################################

    # 6) 모델 저장
    PATH = './QuickDraw_net/quickdraw_net.pth'
    torch.save(net.state_dict(), PATH)
    '''
     #===< 테스트 코드 >=========================================================================================================================
    dataiter = iter(testloader) # testloader를 이터레이터로 변환. 테스트 데이터셋을 반복할 수 있게 하기 위함.
    images, labels = dataiter.next() # 다음 미니 배치의 이미지와 레이블을 가져옴.

    # print images
    imshow(torchvision.utils.make_grid(images)) # 이미지를 그리드 형태로 표시하고, imshow 함수를 사용하여 이미지를 시각화.
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))) # labels 에 해당하는 실제 값들을 출력.

    net = Net() # 신경망 객체를 생성 및 초기화.
    net.load_state_dict(torch.load(PATH)) # 미리 학습된 모델의 가중치 및 매개변수 불러옴.

    outputs = net(images) # net에 입력 이미지를 전달하여 예측 출력을 계산한다.

    _, predicted = torch.max(outputs, 1) # 각 이미지에 대한 예측된 클래스를 가져와 출력한다.

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    '''



'''
   #===< 정확도 계산을 위해 테스트 데이터셋 루프 실행 >=========================================================================================================================
    correct = 0
    total = 0
    with torch.no_grad(): # 컨텍스트 내에서 연산을 수행하여 그래디언트 계산을 비활성화.
        for data in testloader:
            # images, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # 각 이미지에 대한 예측된 클래스를 가져옴.
            total += labels.size(0) # 예측된 클래스와 실제 레이블을 비교하여 정확하게 예측된 개수를 누적.
            correct += (predicted == labels).sum().item() # 전체 테스트 이미지에 대한 정확도 출력.

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10)) # 각 클래스(0부터9까지)에 대해 적중한 예측 개수
    class_total = list(0. for i in range(10)) # 전체 샘플 수
    with torch.no_grad():# 컨텍스트 내에서 연산을 수행하여 그래디언트 계산을 비활성화.
        for data in testloader:
            # images, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(images) # net에 이미지를 전달하여 예측 출력을 계산하고,
            _, predicted = torch.max(outputs, 1)#각 이미지에 대한 예측된 클래스 가져옴.
            c = (predicted == labels).squeeze()#예측된 클래스와 실제 레이블 비교하고, 차원 축소시켜 불리언 마스크'c'를 생성.
            for i in range(4): # 4개의 이미지에 대해 반복하면서 실제 레이블을 확인하고, 정확하게 예측된 개수를 해당 클래스의 인덱스에 맞게 업데이트.
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1 # 전체 샘플 수를 증가시킴.

    for i in range(2): # 클래스 별 정확도 출력. 첫번째와 두번째 클래스에 대해 정확도를 계산하고 출력.
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
   '''


    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
    #del dataiter # 변수를 삭제하여 메모리 확보.
    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%


if __name__ == '__main__':
    main()