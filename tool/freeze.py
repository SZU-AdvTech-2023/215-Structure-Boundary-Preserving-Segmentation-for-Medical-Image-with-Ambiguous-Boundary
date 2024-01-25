import torch
import torch.nn as nn
import torch.optim as optim


#想要实现的效果，对两个层分别更新


# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=3):
        super(net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Linear(4, 4))
        self.fc2 = nn.Linear(4, num_class)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net()

# 冻结fc1层的参数
# for name, param in model.named_parameters():
#     if "fc1" in name:
#         param.requires_grad = False


loss_fn = nn.CrossEntropyLoss()
aoptimizer = optim.SGD(model.fc1.parameters(), lr=1e-2)  # 只传入fc1的参数
boptimizer = optim.SGD(model.fc2.parameters(), lr=1e-2)  # 只传入fc2的参数
print("==========================初始============================")
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)

for epoch in range(10):
    x = torch.randn((3, 8))
    label = torch.randint(0, 3, [3]).long()
    output = model(x)

    loss = loss_fn(output, label)

    print("==========================fc1更新完毕============================")
    aoptimizer.zero_grad()
    loss.backward(retain_graph=True)
    aoptimizer.step()
    print("model.fc1.weight", model.fc1.weight)
    print("model.fc2.weight", model.fc2.weight)

    print("==========================fc2更新完毕============================")
    boptimizer.zero_grad()
    loss.backward()
    boptimizer.step()
    print("model.fc1.weight", model.fc1.weight)
    print("model.fc2.weight", model.fc2.weight)

print("==========================最终============================")
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)
print()