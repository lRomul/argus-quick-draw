import torch
from torch import nn

from cnn_finetune import make_model


class PNAS(nn.Module):
    def __init__(self, pnas):
        super(PNAS, self).__init__()
        self.pnas =  pnas._features

    def forward(self, x):
        x_conv_0 = self.pnas.conv_0(x)
        x_stem_0 = self.pnas.cell_stem_0(x_conv_0)
        x_stem_1 = self.pnas.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.pnas.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.pnas.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.pnas.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.pnas.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.pnas.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.pnas.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.pnas.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.pnas.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.pnas.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.pnas.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.pnas.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.pnas.cell_11(x_cell_9, x_cell_10)
        return x_cell_11


class CountryEmbModel(nn.Module):
    def __init__(self, cnn_finetune, num_country, country_emb_dim=10):
        super().__init__()
        num_classes = cnn_finetune['num_classes']

        if cnn_finetune['model_name'] == 'pnasnet5large':
            self.features = PNAS(make_model(**cnn_finetune))
        else:
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
