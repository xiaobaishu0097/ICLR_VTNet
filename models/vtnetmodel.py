from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder


class VisualTransformer(Transformer):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(VisualTransformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed):
        bs, n, c = src.shape
        src = src.permute(1, 0, 2)
        query_embed = query_embed.permute(2, 0, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src)
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        return hs.transpose(0, 1), memory.permute(1, 2, 0).view(bs, c, n)



class VTNetModel(nn.Module):

    def __init__(self, args):
        super(VTNetModel, self).__init__()
        # visual representation part
        # the networks used to process local visual representation should be replaced with linear transformer
        self.num_cate = args.num_category
        self.image_size = 300
        self.action_embedding_before = args.action_embedding_before
        self.detection_alg = args.detection_alg
        self.wo_location_enhancement = args.wo_location_enhancement

        # global visual representation learning networks
        resnet_embedding_sz = 512
        hidden_state_sz = args.hidden_state_sz
        self.global_conv = nn.Conv2d(resnet_embedding_sz, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)

        # previous action embedding networks
        action_space = args.action_space
        if not self.action_embedding_before:
            self.embed_action = nn.Linear(action_space, 64)
        else:
            self.embed_action = nn.Linear(action_space, 256)

        # local visual representation learning networks
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
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_rate)
        )

        # ==================================================
        # navigation policy part
        # this part should be fixed in this model
        self.lstm_input_sz = 3200
        self.hidden_state_sz = hidden_state_sz

        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)

        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)

        # ==================================================
        # weights initialization
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.global_conv.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

    def embedding(self, state, detection_inputs, action_embedding_input):
        if self.wo_location_enhancement:
            detection_input_features = self.local_embedding(detection_inputs['features'].unsqueeze(dim=0)).squeeze(dim=0)
            detection_input = torch.cat((detection_input_features, detection_inputs['indicator']), dim=1).unsqueeze(dim=0)
        else:
            detection_input_features = self.local_embedding(detection_inputs['features'].unsqueeze(dim=0)).squeeze(dim=0)
            detection_input = torch.cat((
                detection_input_features,
                detection_inputs['labels'].unsqueeze(dim=1),
                detection_inputs['bboxes'] / self.image_size,
                detection_inputs['scores'].unsqueeze(dim=1),
                detection_inputs['indicator']
            ), dim=1).unsqueeze(dim=0)

        image_embedding = F.relu(self.global_conv(state))
        gpu_id = image_embedding.get_device()
        image_embedding = image_embedding + self.global_pos_embedding.cuda(gpu_id)
        image_embedding = image_embedding.reshape(1, -1, 49)

        if not self.action_embedding_before:
            visual_queries = image_embedding
            visual_representation, encoded_rep = self.visual_transformer(src=detection_input,
                                                                         query_embed=visual_queries)
            out = self.visual_rep_embedding(visual_representation)

            action_embedding = F.relu(self.embed_action(action_embedding_input)).unsqueeze(dim=1)
            out = torch.cat((out, action_embedding), dim=1)
        else:
            action_embedding = F.relu(self.embed_action(action_embedding_input)).unsqueeze(dim=2)
            visual_queries = torch.cat((image_embedding, action_embedding), dim=-1)
            visual_representation, encoded_rep = self.visual_transformer(src=detection_input,
                                                                         query_embed=visual_queries)
            out = self.visual_rep_embedding(visual_representation)

        out = out.reshape(1, -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):
        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        (hx, cx) = model_input.hidden

        detection_inputs = model_input.detection_inputs
        action_probs = model_input.action_probs

        x, image_embedding = self.embedding(state, detection_inputs, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )

def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos