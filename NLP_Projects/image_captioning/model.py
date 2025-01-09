import torch
import torch.nn as nn
from torchvision import models

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, pretrained_embeddings):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings, dtype=torch.float))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)
