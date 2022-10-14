import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from DLCPArchitecture import DLCPNet
from DLCPDataLoader import DataSet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
PATH = 'DLCP.pth'

torch.autograd.set_detect_anomaly(True)

batch_size = 4

dataset = DataSet(r'/data/students/royoz/DLCP/data4')
train_set, test_set = torch.utils.data.random_split(dataset, [18, 2])#[90, 10])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)#, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)#, num_workers=0)

net = DLCPNet()
net = nn.DataParallel(net)
#net.load_state_dict(torch.load(PATH))
#net.load_state_dict(torch.load(DLCP3.pth))
device = torch.device('cuda:0')
net = net.to(device)
criterion = nn.CrossEntropyLoss()#nn.SmoothL1Loss()  #SmoothL1Loss / MSELoss / L1Loss
optimizer = optim.Adam(net.parameters(), lr=0.01)

training_loss = []
testing_loss = []
for epoch in range(1, 1000000):  # loop over the dataset multiple times
    train_loss = 0
    for i, data in enumerate(trainloader):
        print(i)
        # get the inputs; data is a list of [inputs, tags]
        inputs, tags = data
        inputs = inputs.to(device)
        tags = tags.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #background = tags.select(1, 0) == 0
        #background = torch.stack([background]), dim=1)

        '''
        background = tags == 0
        outputs_masked = outputs.masked_fill_(background, 0.0)
        loss = criterion(outputs_masked, tags)
        '''
        loss = criterion(outputs, tags)
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()
        if epoch == 1 and i == 0:
            print("Init loss:", train_loss)
            training_loss.append(train_loss)

    # print statistics
    training_loss.append(train_loss / len(trainloader))
    print(f'[{epoch}] loss: {training_loss[-1]:.3f}')

    if epoch % 10 == 0:
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, tags = data
                inputs = inputs.to(device)
                tags = tags.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, tags)
                test_loss += loss.item()
        testing_loss.append(test_loss / len(testloader))
        t1 = range(0, epoch+1)
        t2 = range(0, epoch+1, 10)
        plt.figure()
        plt.plot(t1, training_loss, label='train')
        plt.plot(t2, training_loss[:1] + testing_loss, label='test')
        plt.legend()
        if len(testing_loss) >= 2 and testing_loss[-1] < testing_loss[-2]:
            torch.save(net.state_dict(), './DLCP.pth')
        plt.savefig('loss.png')
