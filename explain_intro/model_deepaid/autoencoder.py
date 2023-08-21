import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def se2rmse(a):
    return torch.sqrt(sum(a.t()) / a.shape[1])





class autoencoder(nn.Module):

    def __init__(self, feature_size, datatype):
        super(autoencoder, self).__init__()

        if datatype == 'toy':

            self.encoder = nn.Sequential(nn.Linear(feature_size, feature_size - 1),
                                         nn.ReLU(True),
                                         nn.Linear(feature_size - 1, feature_size - 2),
                                         )

            self.decoder = nn.Sequential(nn.Linear(feature_size - 2, feature_size - 1),
                                         nn.ReLU(True),
                                         nn.Linear(feature_size - 1, feature_size),
                                         )
        elif datatype == 'NADC':

            self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size * 0.75)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.75), int(feature_size * 0.5)),
                                         )
            self.decoder = nn.Sequential(nn.Linear(int(feature_size * 0.5), int(feature_size * 0.75)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.75), int(feature_size)),
                                         )


        elif datatype in ['IDS2017','IDS2012']:

            self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size * 0.75)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.75), int(feature_size * 0.5)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.5), int(feature_size * 0.25)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.25), int(feature_size * 0.1)))

            self.decoder = nn.Sequential(nn.Linear(int(feature_size * 0.1), int(feature_size * 0.25)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.25), int(feature_size * 0.5)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.5), int(feature_size * 0.75)),
                                         nn.ReLU(True),
                                         nn.Linear(int(feature_size * 0.75), int(feature_size)),
                                         )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


criterion = nn.MSELoss()
getMSEvec = nn.MSELoss(reduction='none')


def train(X_train, feature_size, args):
    batch_size = 512
    lr = 0.001
    epoches = 50
    ae_t = args.t_deepaid

    model = autoencoder(feature_size, args.datatype).to(device)
    optimizier = optim.Adam(model.parameters(), lr=lr)
    model.train()

    X_train = torch.from_numpy(X_train).type(torch.float)
    if torch.cuda.is_available(): X_train = X_train.cuda()
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            # if step % 10 == 0:
            #     print('epoch:{}/{}'.format(epoch, step), '|Loss:', loss.item())
        if epoch % 20 == 0:
            print('epoch:{}/{}'.format(epoch+1, epoches), '|Loss:', loss.item())

    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output, X_train)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()

    # print("max AD score", max(rmse_vec))
    thres = max(rmse_vec)
    rmse_vec.sort()
    pctg = ae_t
    thres = rmse_vec[int(len(rmse_vec) * pctg)]
    # print("thres:", thres)

    return model, thres


@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_test = X_test.cuda()
    output = model(X_test)
    mse_vec = getMSEvec(output, X_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    # idx_mal = np.where(rmse_vec>thres)
    # idx_ben = np.where(rmse_vec<=thres)
    # print(len(rmse_vec[idx_ben]),len(rmse_vec[idx_mal]))
    return rmse_vec


def test_plot(X_test, rmse_vec, thres):
    plt.scatter(np.linspace(0, len(X_test) - 1, len(X_test)), rmse_vec, s=10, alpha=0.5)
    plt.plot(np.linspace(0, len(X_test) - 1, len(X_test)), [thres] * len(X_test), c='black')
    plt.show()



