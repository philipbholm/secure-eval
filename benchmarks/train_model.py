import torch
import torchvision

torch.manual_seed(1)
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

train_data = torchvision.datasets.MNIST(
    "data", transform=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=7, padding=0, stride=3
        )
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        # Square activation function
        x = x * x
        # Flatten while keeping batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    print(f"Epoch: {epoch}\tTraining loss: {train_loss:.6f}")

model.eval()
torch.save(model.state_dict(), "models/mnist_convnet.pth")
