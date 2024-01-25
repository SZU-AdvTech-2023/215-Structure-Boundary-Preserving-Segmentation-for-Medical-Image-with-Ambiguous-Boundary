import numpy as np

from unet.unet_model import UNet
from util.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch,gc
from tqdm import tqdm


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.001):
    net.load_state_dict(torch.load('best_model_.pth', map_location=device))
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    #分开更新网络部件
    parametersList = []
    # for param in net.named_parameters():
    i = 0
    # net.eval()
    for param in net.named_parameters():
        # print(param)
        if "bpb" not in param[0]:
            parametersList.append(param[1])

    bpbParametersList1 = []
    bpbParametersList2 = []
    bpbParametersList3 = []
    bpbParametersList4 = []
    bpbParametersList5 = []
    bpbParametersList6 = []
    for param in net.named_parameters():
        if "downbpb2" in param[0]:
            bpbParametersList1.append(param[1])
        if "downbpb3" in param[0]:
            bpbParametersList2.append(param[1])
        if "downbpb4" in param[0]:
            bpbParametersList3.append(param[1])
        if "upbpb1" in param[0]:
            bpbParametersList4.append(param[1])
        if "upbpb2" in param[0]:
            bpbParametersList5.append(param[1])
        if "upbpb3" in param[0]:
            bpbParametersList6.append(param[1])

    # 定义RMSprop算法
    optimizer = optim.RMSprop(parametersList, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer1 = optim.RMSprop(bpbParametersList1, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer2 = optim.RMSprop(bpbParametersList2, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer3 = optim.RMSprop(bpbParametersList3, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer4 = optim.RMSprop(bpbParametersList4, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer5 = optim.RMSprop(bpbParametersList5, lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer6 = optim.RMSprop(bpbParametersList6, lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义Loss算法，二分类交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            print("这是第",epoch,"个epoch")
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            i = 0
            for image, label, label1, label2, label3, label4 in train_loader:
                optimizer.zero_grad()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer4.zero_grad()
                optimizer5.zero_grad()
                optimizer6.zero_grad()
                # print("1:{}".format(torch.cuda.memory_allocated(0)))
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                label1 = label1.to(device=device, dtype=torch.float32)
                label2 = label2.to(device=device, dtype=torch.float32)
                label3 = label3.to(device=device, dtype=torch.float32)
                label4 = label4.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果--->这个结果里有7个值

                pred = net(image)

                # 计算loss
                loss = criterion(pred[0], label)

                #------------这地方的输出和label都还没填好---------
                loss1 = criterion(pred[1], label1)
                loss2 = criterion(pred[2], label2)
                loss3 = criterion(pred[3], label3)
                loss4 = criterion(pred[4], label2)
                loss5 = criterion(pred[5], label1)
                loss6 = criterion(pred[6], label4)

                # print(loss)
                # 保存loss值最小的网络参数
                # aaa = loss
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model_.pth')
                if i % 100 == 0:
                    print("==============现在是训练的第", epoch, "轮，当前loss是", loss,"这是第", i, "个for循环================")
                i = i + 1
                # print("2.2:{}".format(torch.cuda.memory_allocated(0)))
                # 更新参数
                loss.backward(retain_graph=True)
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)
                loss4.backward(retain_graph=True)
                loss5.backward(retain_graph=True)
                loss6.backward()

                optimizer.step()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer5.step()
                optimizer6.step()
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "/home/jiayandai/mydataset"
    data_path = "/data/jiayandai/mydataset"
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    train_net(net, device, data_path, epochs=10)
