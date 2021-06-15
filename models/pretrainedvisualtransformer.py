from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder
from .visualtransformermodel import VisualTransformer, get_gloabal_pos_embedding


class PreTrainedVisualTransformer(nn.Module):
    def __init__(self, args):
        super(PreTrainedVisualTransformer, self).__init__()
        self.image_size = 300
        self.detection_alg = args.detection_alg
        self.wo_location_enhancement = args.wo_location_enhancement

        # same layers as VisualTransformer visual representation learning part
        self.global_conv = nn.Conv2d(512, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)

        if self.detection_alg == 'detr' and not self.wo_location_enhancement:
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 249),
                nn.ReLU(),
            )
        elif self.detection_alg == 'detr' and self.wo_location_enhancement:
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 255),
                nn.ReLU(),
            )
        elif self.detection_alg == 'fasterrcnn':
            self.local_embedding = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 249),
                nn.ReLU(),
            )
        elif self.detection_alg == 'fasterrcnn_bottom':
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 249),
                nn.ReLU(),
            )

        self.visual_transformer = VisualTransformer(
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout_rate,
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_rate)
        )

        # pretraining network action predictor, should be used in Visual Transformer model
        self.pretrain_fc = nn.Linear(3136, 6)

    def forward(self, global_feature: torch.Tensor, local_feature: dict):
        batch_size = global_feature.shape[0]

        global_feature = global_feature.squeeze(dim=1)
        image_embedding = F.relu(self.global_conv(global_feature))
        image_embedding = image_embedding + self.global_pos_embedding.repeat([batch_size, 1, 1, 1]).cuda()
        image_embedding = image_embedding.reshape(batch_size, -1, 49)

        if self.wo_location_enhancement:
            detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
            local_input = torch.cat((detection_input_features, local_feature['indicator']), dim=2)
        else:
            detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
            local_input = torch.cat((
                detection_input_features,
                local_feature['labels'].unsqueeze(dim=2),
                local_feature['bboxes'] / self.image_size,
                local_feature['scores'].unsqueeze(dim=2),
                local_feature['indicator']
            ), dim=2)

        visual_representation, _ = self.visual_transformer(src=local_input, query_embed=image_embedding)

        visual_rep = self.visual_rep_embedding(visual_representation)
        visual_rep = visual_rep.reshape(batch_size, -1)

        action = self.pretrain_fc(visual_rep)

        return {
            'action': action,
            'fc_weights': self.pretrain_fc.weight,
            'visual_reps': visual_rep.reshape(batch_size, 64, 49)
        }
