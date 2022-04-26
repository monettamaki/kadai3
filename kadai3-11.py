#ライブラリの準備
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

#GPUチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(device)
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn[:, -1, :])
        return x
model = RNN().to(device)

#円
def circle(r):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    x = np.linspace(0, 2*np.pi)
    cic_x = r*np.cos(x)
    cic_y = -r*np.sin(x)
    plt.plot(cic_x, cic_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return cic_x,cic_y

#ノイズを乗せて訓練データに
def circle_noiz(r):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    x = np.linspace(0, 4*np.pi)
    cic_noiz_x = r*np.cos(x)+ np.random.normal(0, 0.05, len(x))
    cic_noiz_y = -r*np.sin(x)+ np.random.normal(0, 0.05, len(x))
    plt.plot(cic_noiz_x, cic_noiz_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x,cic_noiz_x, cic_noiz_y

#学習
def train(optimizer,EPOCHS,input_data, criterion, model, train_loader,cic_train_x, cic_train_y, cic_noiz_x, cic_noiz_y, n_sample, n_time):
    record_loss_train = []
    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0
        for j, (cic_train_x, cic_train_y) in enumerate(train_loader):
            loss = criterion(model(cic_train_x), cic_train_y.to(device))
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)
        if epoch%10 == 0:
            print("Epoch:", epoch, "Loss_Train:", loss_train)
            predicted = list(input_data[0].reshape(-1))
            model.eval()
            with torch.no_grad():
                for epoch in range(n_sample):
                    cic_train_x = torch.tensor(predicted[-n_time:])
                    cic_train_x = cic_train_x.reshape(1, n_time, 1)
                    predicted.append(model(cic_train_x)[0].item())
                    cic_train_y = torch.tensor(predicted[-n_time:])
                    cic_train_y = cic_train_y.reshape(1, n_time, 1)
                    predicted.append(model(cic_train_y)[0].item())
            plt.plot(range(len(cic_train_x)), range(len(cic_train_y)), label="Actual")
            plt.plot(range(len(predicted)), predicted, label="Predicted")
            plt.plot(cic_noiz_x, cic_noiz_y,label="Actual")

            plt.legend()
            plt.show()


def main():
    EPOCHS = 100
    r=2
    circle(r) #こんな円だよー
    x, cic_noiz_x, cic_noiz_y = circle_noiz(r) #ノイズ乗せたよー

    #ハイパーパラメータ
    n_time = 10
    n_sample = len(x) - n_time

    #データを格納する空の配列を準備
    input_data = np.zeros((n_sample, n_time, 1))
    correct_data = np.zeros((n_sample, 1))
    cic_train_x = cic_noiz_x
    cic_train_y = cic_noiz_y
    #前処理
    for i in range(n_sample):
        input_data[i] = cic_train_x[i:i+n_time].reshape(-1, 1)
        correct_data[i] = [cic_train_y[i+n_time]]
    input_data = torch.FloatTensor(input_data)
    correct_data = torch.FloatTensor(correct_data)

    #バッチデータの準備
    dataset = TensorDataset(input_data, correct_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    #最適化手法の定義
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train(optimizer,EPOCHS,input_data, criterion, model, train_loader,cic_train_x, cic_train_y, cic_noiz_x, cic_noiz_y, n_sample, n_time)
    

if __name__ == "__main__":
    main()
