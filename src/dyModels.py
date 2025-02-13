import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class NormalizedModel(nn.Module):
    def __init__(self) -> None:
        super(NormalizedModel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # mean = input.mean(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        # std = input.std(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        mean = input.mean()
        std = input.std()
        normalized_input = (input - mean)/std
        return normalized_input
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_len, hidden_size)
        attention_score = torch.matmul(lstm_out, self.attention_weights).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_score, dim=-1)  # (batch_size, seq_len)
        # 对 lstm_out 做加权求和
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)
        return context_vector
    
class LSTM_Attention_Model(nn.Module):
    def __init__(self,classes=10,input_size=12*2, z_dim=64,num_layers=3):
        super(LSTM_Attention_Model, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True)
        # Attention层
        self.attention = Attention(z_dim)
        # 全连接层
        self.fc = nn.Linear(z_dim, classes)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # 对变长序列进行填充
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # 将填充后的序列包装为PackedSequence格式
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # 通过LSTM进行处理
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # 通过 pad_packed_sequence 解包
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        attention_output = self.attention(output)  # 这里使用 Attention 层
        output = self.fc(attention_output)

        return output    
class LSTM_Model(nn.Module):
    def __init__(self,classes=10,input_size=12*2, z_dim=64,num_layers=3):
        super(LSTM_Attention_Model, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(z_dim, classes)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
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
class FCModel(nn.Module):
    def __init__(self, z_dim=54, classes=10):
        super().__init__()
        self.main_module = nn.Sequential(
            NormalizedModel(),
            nn.Linear(12*2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, input, length):
        segment = input
        out = self.main_module(segment[:,0,:])
        return out

    # def features(self, input):
    #     N = len(input)
    #     segment = input.view(N, -1) 
    #     out = self.main_module(segment).view(N, -1)
    #     return out

class Transformer_Model(nn.Module):
    def __init__(self, classes=10, input_size=24, z_dim=64, num_layers=2, num_heads=4):
        super(Transformer_Model, self).__init__()
        # Transformer的编码器层
        self.embedding = nn.Linear(input_size, z_dim)  # 将输入的每个时间步的特征映射到z_dim维
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=z_dim, nhead=num_heads, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )
        # 全连接层
        self.fc = nn.Linear(z_dim, classes)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # 1. 填充数据并转换为适应 Transformer 的格式
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        device = x.device 
        # 2. 使用 padding_mask
        # lengths: (batch_size,) -> 将其转换为 mask
        padding_mask = self.create_padding_mask(lengths, x_padded.size(1)).to(device) # (batch_size, seq_len)

        # 3. 输入到 Transformer 之前，将每个时间步的特征映射到 z_dim
        x_embedded = self.embedding(x_padded)  # (batch_size, seq_len, z_dim)

        # 4. Transformer 输入是 (seq_len, batch_size, z_dim)
        x_embedded = x_embedded.permute(1, 0, 2)  # (seq_len, batch_size, z_dim)

        # 5. Transformer 编码器
        transformer_output = self.transformer_encoder(x_embedded, src_key_padding_mask=padding_mask)
        
        # 6. 取 Transformer 输出的最后一个时间步的隐藏状态
        last_hidden_state = transformer_output[-1]  # (batch_size, z_dim)

        # 7. 分类输出
        output = self.fc(last_hidden_state)
        return output

    def create_padding_mask(self, lengths, max_len):
        """
        创建 padding mask，返回形状为 (batch_size, max_len)
        lengths: (batch_size,) - 有效序列长度
        max_len: 最大序列长度
        """
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return mask  # 返回 (batch_size, max_len)

class LSTM_MultiHeadAttention_Model(nn.Module):
    def __init__(self, classes=10, input_size=12*2, z_dim=64, num_layers=3, num_heads=8):
        super(LSTM_MultiHeadAttention_Model, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True)
        
        # Multi-Head Attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=z_dim, num_heads=num_heads, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(z_dim, classes)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # Padding the sequences for variable length inputs
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # Packing the padded sequence
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Unpacking the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Multi-Head Attention (output shape should be (batch_size, seq_len, z_dim))
        attention_output, _ = self.multihead_attention(output, output, output)
        
        # Pass through fully connected layer
        output = self.fc(attention_output[:, -1, :])  # Only take the last timestep for classification

        return output

class LSTM_FusionMultiheadAttention_Model(nn.Module):
    def __init__(self, classes=10, input_size=12*2, z_dim=64, num_layers=3, num_heads=8):
        super(LSTM_FusionMultiheadAttention_Model, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True)
        
        # Multi-Head Attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=z_dim, num_heads=num_heads, batch_first=True)
        
        # Fusion Attention layer: concatenate LSTM output and Multi-Head Attention output
        self.fc_fusion = nn.Linear(2 * z_dim, z_dim)  # After fusion, concatenate outputs of LSTM and Multi-Head Attention
        
        # Fully connected layer for final classification
        self.fc_out = nn.Linear(z_dim, classes)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # Padding the sequences for variable length inputs
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # Packing the padded sequence
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Unpacking the output
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Multi-Head Attention (output shape should be (batch_size, seq_len, z_dim))
        attention_output, _ = self.multihead_attention(lstm_output, lstm_output, lstm_output)
        
        # Fusion Attention: concatenate LSTM output and Attention output along the feature dimension
        fusion_output = torch.cat((lstm_output, attention_output), dim=-1)  # Concatenate along feature axis (z_dim)
        fusion_output = self.fc_fusion(fusion_output)  # Apply a fully connected layer after fusion
        
        # Final classification output: typically, use the last time step for prediction
        output = self.fc_out(fusion_output[:, -1, :])  # Take the last time step for classification

        return output

class BiLSTM_MultiHeadAttention_Model(nn.Module):
    def __init__(self, classes=10, input_size=12*2, z_dim=64, num_layers=3, num_heads=8):
        super(BiLSTM_MultiHeadAttention_Model, self).__init__()
        
        # BiLSTM layer (bidirectional=True makes it a BiLSTM)
        self.bilstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Multi-Head Attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=z_dim*2, num_heads=num_heads, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(z_dim*2, classes)  # Updated input size for BiLSTM (z_dim * 2 for bidirectional)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # Padding the sequences for variable length inputs
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # Packing the padded sequence
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through BiLSTM (bidirectional LSTM)
        packed_output, (hn, cn) = self.bilstm(packed_input)
        
        # Unpacking the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Multi-Head Attention (output shape should be (batch_size, seq_len, z_dim*2) for BiLSTM)
        attention_output, _ = self.multihead_attention(output, output, output)
        
        # Pass through fully connected layer
        output = self.fc(attention_output[:, -1, :])  # Only take the last timestep for classification

        return output

class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, classes=10, input_size=12*2, z_dim=64, num_layers=4, num_heads=8):
        super(BiLSTM_Attention_Model, self).__init__()
        
        # BiLSTM layer (bidirectional=True makes it a BiLSTM)
        self.bilstm = nn.LSTM(input_size, z_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Multi-Head Attention layer
        self.attention = Attention(z_dim*2)
        
        # Fully connected layer
        self.fc = nn.Linear(z_dim*2, classes)  # Updated input size for BiLSTM (z_dim * 2 for bidirectional)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size,)
        
        # Padding the sequences for variable length inputs
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        
        # Packing the padded sequence
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through BiLSTM (bidirectional LSTM)
        packed_output, (hn, cn) = self.bilstm(packed_input)
        
        # Unpacking the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        attention_output = self.attention(output)  # 这里使用 Attention 层
        output = self.fc(attention_output)

        return output