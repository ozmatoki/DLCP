from DLCPArchitecture import DLCPNet
from DLCPDataLoader import DataSet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import io

PATH = 'DLCP3.pth'

torch.autograd.set_detect_anomaly(True)


dataset = DataSet(r'/data/students/royoz/DLCP/data2-3')
train_set, test_set = torch.utils.data.random_split(dataset, [18, 2])
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

net = DLCPNet()
net = nn.DataParallel(net)
net.load_state_dict(torch.load(PATH))
device = torch.device('cuda:0')
net = net.to(device)
net.eval()
i=0
for data in testloader:
    inputs, tags = data
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = net(inputs).cpu()

    fig, axs = plt.subplots(1, 3)

    inputs = inputs.cpu()

    p = axs[0].imshow(inputs[0, 0, :, :])

    p = axs[1].imshow(tags[0, :, :])

    p = axs[2].imshow(torch.max(outputs[0, :, :, :], 0)[1])
    fig.colorbar(p, ax=axs[2])

    #plt.savefig(f'output/{i}.png')
    #plt.close()

    #io.savemat(f'output/{i}.mat', {"output": outputs[0, 0, :, :].numpy()})
    i += 1
    plt.show()
