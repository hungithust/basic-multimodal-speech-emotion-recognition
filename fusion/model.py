import torch
from torch import nn
import torch.nn.functional as F

from typing import List

from audio.wav2vec2 import Wav2Vec2
from text.deberta import DebertaV3


class FusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        deberta_model: DebertaV3,
        wav2vec2_model: Wav2Vec2,
        hidden_layers: List[int] = None,
        freeze_weights: bool = True,
    ):
        super(FusionModel, self).__init__()
        if freeze_weights:
            for param in deberta_model.parameters():
                param.requires_grad = False
            for param in wav2vec2_model.parameters():
                param.requires_grad = False
        if hidden_layers is None:
            hidden_layers = [512, 128, 32]
        self.num_classes = num_classes
        self.text_model = deberta_model
        self.audio_model = wav2vec2_model

        self.mlp_head = nn.Sequential(
            # TODO: Remove + 4 because is from the old DeBERTa model
            nn.Linear(768 + 768 + num_classes * 2, hidden_layers[0])
        )
        for i in range(0, len(hidden_layers) - 1):
            self.mlp_head.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.cls_head = nn.Linear(hidden_layers[-1], num_classes + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        # audio_features = self.audio_model.flatten(
        #     self.audio_model.wav2vec2(audio).last_hidden_state
        # )
        # audio_classification = self.audio_model.softmax(
        #     self.audio_model.cls_head(self.audio_model.lm_head(audio_features))
        # )

        
        # Lấy audio features từ mean pooling (không dùng flatten nữa)  
        audio_sequence = self.audio_model.wav2vec2(audio).last_hidden_state  
        audio_features = audio_sequence.mean(dim=1)  # (batch_size, 768)  
          
        # Lấy audio classification  
        audio_classification = self.audio_model.softmax(  
            self.audio_model.cls_head(self.audio_model.lm_head(audio_features))  
        )

        # --- XỬ LÝ TEXT (Sửa lại phần này) ---
        # 1. Lấy toàn bộ trạng thái ẩn (Sequence Output)
        # Shape: (Batch_size, Sequence_length, Hidden_size)
        sequence_output = self.text_model.roberta(text).last_hidden_state

        # 2. Lấy feature đại diện cho câu (CLS token - vị trí số 0) để ghép vào Fusion layer
        # Shape: (Batch_size, Hidden_size)
        text_features = sequence_output[:, 0, :] 

        # 3. Lấy logits phân loại từ text (đưa toàn bộ sequence vào classifier)
        # Classifier của RoBERTa tự động lấy token [CLS] bên trong nó để xử lý
        text_classification = self.text_model.classifier(sequence_output)
        
        # Lưu ý: classifier đã có dropout bên trong, không cần self.text_model.dropout() ở ngoài nữa
        x = torch.cat(
            [audio_features, audio_classification, text_features, text_classification],
            dim=1,
        )
        x = self.mlp_head(x)
        x = self.cls_head(x)
        x = self.softmax(x)
        return x


class EarlyFusionModel(nn.Module):
    """
    Kết hợp text và audio ở lớp fully connected thứ nhất (lớp gộp có kích thước 256)
    """
    def __init__(
        self,
        num_classes: int,
        deberta_model: DebertaV3,
        wav2vec2_model: Wav2Vec2,
        hidden_layers: List[int] = None,
        freeze_weights: bool = True,
    ):
        super(EarlyFusionModel, self).__init__()
        if freeze_weights:
            for param in deberta_model.parameters():
                param.requires_grad = False
            for param in wav2vec2_model.parameters():
                param.requires_grad = False
        if hidden_layers is None:
            hidden_layers = [128, 32]
        
        self.num_classes = num_classes
        self.text_model = deberta_model
        self.audio_model = wav2vec2_model
        self.hidden_size = 256  # Kích thước lớp gộp thứ nhất

        # Kết hợp ở lớp lm_head: (256 + 256)
        self.fusion_head = nn.Linear(256 + 256, self.hidden_size)
        
        # Các hidden layers tiếp theo
        self.mlp_head = nn.Sequential()
        for i in range(0, len(hidden_layers) - 1):
            self.mlp_head.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Classification head
        self.cls_head = nn.Linear(hidden_layers[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        # Audio processing
        audio_sequence = self.audio_model.wav2vec2(audio).last_hidden_state  # (batch_size, seq_len, 768)
        audio_sequence = audio_sequence.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        audio_features = self.audio_model.lm_head(audio_sequence)  # (batch_size, 256, seq_len)
        
        # Text processing
        text_sequence = self.text_model.deberta(text).last_hidden_state  # (batch_size, seq_len, 768)
        text_sequence = text_sequence.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        text_features = self.text_model.lm_head(text_sequence)  # (batch_size, 256, seq_len)
        
        # Pooling
        audio_features = audio_features.mean(dim=2)  # (batch_size, 256)
        text_features = text_features.mean(dim=2)  # (batch_size, 256)
        
        # Fusion ở lớp đầu tiên
        x = torch.cat([audio_features, text_features], dim=1)  # (batch_size, 512)
        x = self.fusion_head(x)  # (batch_size, 256)
        x = self.mlp_head(x)
        x = self.cls_head(x)
        x = self.softmax(x)
        return x


class LateFusionModel(nn.Module):
    """
    Kết hợp text và audio ở lớp fully connected thứ hai (lớp gộp có kích thước 128)
    """
    def __init__(
        self,
        num_classes: int,
        deberta_model: DebertaV3,
        wav2vec2_model: Wav2Vec2,
        hidden_layers: List[int] = None,
        freeze_weights: bool = True,
    ):
        super(LateFusionModel, self).__init__()
        if freeze_weights:
            for param in deberta_model.parameters():
                param.requires_grad = False
            for param in wav2vec2_model.parameters():
                param.requires_grad = False
        if hidden_layers is None:
            hidden_layers = [32]
        
        self.num_classes = num_classes
        self.text_model = deberta_model
        self.audio_model = wav2vec2_model
        self.hidden_size = 128  # Kích thước lớp gộp thứ hai

        # Kết hợp ở lớp lm_head_2: (128 + 128)
        self.fusion_head = nn.Linear(128 + 128, self.hidden_size)
        
        # Các hidden layers tiếp theo
        self.mlp_head = nn.Sequential()
        for i in range(0, len(hidden_layers) - 1):
            self.mlp_head.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Classification head
        self.cls_head = nn.Linear(hidden_layers[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        # Audio processing
        audio_sequence = self.audio_model.wav2vec2(audio).last_hidden_state  # (batch_size, seq_len, 768)
        audio_sequence = audio_sequence.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        audio_features = self.audio_model.lm_head(audio_sequence)  # (batch_size, 256, seq_len)
        audio_features = self.audio_model.lm_head_2(audio_features)  # (batch_size, 128, seq_len)
        
        # Text processing
        text_sequence = self.text_model.deberta(text).last_hidden_state  # (batch_size, seq_len, 768)
        text_sequence = text_sequence.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        text_features = self.text_model.lm_head(text_sequence)  # (batch_size, 256, seq_len)
        text_features = self.text_model.lm_head_2(text_features)  # (batch_size, 128, seq_len)
        
        # Pooling
        audio_features = audio_features.mean(dim=2)  # (batch_size, 128)
        text_features = text_features.mean(dim=2)  # (batch_size, 128)
        
        # Fusion ở lớp thứ hai
        x = torch.cat([audio_features, text_features], dim=1)  # (batch_size, 256)
        x = self.fusion_head(x)  # (batch_size, 128)
        x = self.mlp_head(x)
        x = self.cls_head(x)
        x = self.softmax(x)
        return x
