import torch
from torch import nn

from cnn_finetune import make_model


class CountryEmbModel(nn.Module):
    def __init__(self, cnn_finetune, num_country, country_emb_dim=10):
        super().__init__()
        num_classes = cnn_finetune['num_classes']
        cnn_finetune = make_model(**cnn_finetune)
        self.features = cnn_finetune._features
        self.pool = cnn_finetune.pool
        self.dropout = cnn_finetune.dropout

        img_in_features = cnn_finetune._classifier.in_features

        self.country_emb = nn.Embedding(num_country, country_emb_dim)

        self.fc_1 = nn.Linear(img_in_features + country_emb_dim, 512)
        self.fc_2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        img, country = x
        img = self.features(img)
        if self.pool is not None:
            img = self.pool(img)
        img = img.view(img.size(0), -1)

        country = self.country_emb(country)
        country = country.view(country.size(0), -1)

        x = torch.cat([img, country], dim=1)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_2(x)
        return x


class CountryRecEmbModel(nn.Module):
    def __init__(self, cnn_finetune, num_country, country_emb_dim=10, recogn_emb_dim=4):
        super().__init__()
        num_classes = cnn_finetune['num_classes']
        cnn_finetune = make_model(**cnn_finetune)
        self.features = cnn_finetune._features
        self.pool = cnn_finetune.pool
        self.dropout = cnn_finetune.dropout

        img_in_features = cnn_finetune._classifier.in_features

        self.country_emb = nn.Embedding(num_country, country_emb_dim)
        self.recogn_emb = nn.Embedding(2, recogn_emb_dim)

        self.fc_1 = nn.Linear(img_in_features + country_emb_dim + recogn_emb_dim + 1, 512)
        self.fc_2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        img, country, recogn, draw_len = x
        img = self.features(img)
        if self.pool is not None:
            img = self.pool(img)
        img = img.view(img.size(0), -1)

        country = self.country_emb(country)
        country = country.view(country.size(0), -1)

        recogn = self.recogn_emb(recogn)
        recogn = recogn.view(recogn.size(0), -1)

        draw_len = draw_len.view(draw_len.size(0), -1)

        x = torch.cat([img, country, recogn, draw_len], dim=1)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_2(x)
        return x
