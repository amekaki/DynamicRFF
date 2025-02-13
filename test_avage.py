import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # 对变长序列进行填充
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # 将填充后的序列包装为PackedSequence格式
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # 通过LSTM进行处理
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # 通过 pad_packed_sequence 解包
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # 使用 LSTM 输出的最后一个时间步的隐藏状态
        last_hidden_state = hn[-1]
        
        # 进行分类
        output = self.fc(last_hidden_state)
        return output

# 模拟一些数据
x_train = [torch.randn(seq_len, 24) for seq_len in [5, 10, 7, 3, 8]]  # 每个序列长度不同
y_train = torch.tensor([0, 1, 0, 1, 0])  # 对应的标签

x_test = [torch.randn(seq_len, 24) for seq_len in [6, 4]]  # 测试数据
y_test = torch.tensor([1, 0])  # 测试标签

# 获取序列的长度
train_lengths = torch.tensor([len(seq) for seq in x_train])
test_lengths = torch.tensor([len(seq) for seq in x_test])

# 使用 torch.stack 来转换 x_train 为 3D 张量
x_train_tensor = torch.nn.utils.rnn.pad_sequence(x_train, batch_first=True, padding_value=0)
x_test_tensor = torch.nn.utils.rnn.pad_sequence(x_test, batch_first=True, padding_value=0)

# 创建DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train, train_lengths)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test, test_lengths)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 初始化模型、损失函数和优化器
input_size = 24
hidden_size = 64
output_size = 2  # 假设是二分类问题
model = LSTM_Model(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x_batch, y_batch, lengths in train_loader:
        # 将数据转移到指定设备（如 GPU）
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(x_batch, lengths)
        
        # 计算损失
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        # 统计准确率
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 测试过程
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch, lengths in test_loader:
            # 将数据转移到指定设备（如 GPU）
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)
            
            # 前向传播
            output = model(x_batch, lengths)
            
            # 计算损失
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            
            # 统计准确率
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 使用GPU训练（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练和测试
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
