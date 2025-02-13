import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from src.datasetv4 import RFdataset
config = {
    'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
    'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
    'PathDelay': [0, 5e-7],  # 路径延迟范围
    'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
    'PathNum': [2, 8],  # 多径数量
    'K':[10,20],
    'fd':[0,5]
}
test_data = RFdataset(config=config, ChannelNum=3, DeviceNum=10, regenerate_flag=False, label='rff', datasetname='test')
# test_data.random_channel(3, 10, config)
# RB_data = test_data.data['x'].to(user_trainer.device)
# print(RB_data.shape)
rff_label = test_data.data['y_rff']
RB_data = test_data.data['x']
print(RB_data.shape)
# 自定义 Dataset
class IntentDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        word_indices = [self.vocab[word] for word in sentence if word in self.vocab]
        word_indices = [len(self.vocab)] + word_indices  # 添加 [CLS] token
        return torch.tensor(word_indices, dtype=torch.long), label


# 自定义 collate_fn
def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # 填充 0
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_inputs, labels


# 定义 Transformer 模型
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=4, num_layers=2):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1是为[CLS]添加一个索引
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embedding_dim, num_classes)  # 分类层

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        print(embedded.shape,x.shape)
        # 构造 mask，如果未传入，则根据 x 自动生成
        if mask is None:
            mask = (x != 0)  # (batch_size, seq_len)
        # print(mask,x)
        transformer_output = self.transformer(
            embedded, src_key_padding_mask=~mask
        )  # 注意，mask 直接使用 (batch_size, seq_len)
        cls_output = transformer_output[0]  # 取 [CLS] 的输出 (batch_size, embedding_dim)
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        return logits



# 准备数据
vocab = {'hello': 1, 'how': 2, 'are': 3, 'you': 4, 'weather': 5, 'today': 6}
sentences = [['hello', 'how', 'are', 'you'], ['weather', 'today']]
labels = [0, 1]  # 0: greeting, 1: query_weather
dataset = IntentDataset(sentences, labels, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 初始化模型
vocab_size = len(vocab)
embedding_dim = 300
num_classes = 2
model = IntentClassifier(vocab_size, embedding_dim, num_classes)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(5):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
