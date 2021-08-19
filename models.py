import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import os
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pytorch_lightning.metrics.functional import accuracy

from torch.utils.data import Dataset
from torch.jit.annotations import Optional

import math
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

import cv2


class BasicPMP(pl.LightningModule):
    def __init__(self, lr=0.0001, drop_rate=0.5, n_channels=12, img_input_dim=(500, 500), img_out_dim=(28, 28)):
        super().__init__()
        self.lr = lr

        self.d_model = 256
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=16, num_encoder_layers=4, num_decoder_layers=2)

        # self.points_layer = nn.Linear(2, self.d_model-3)

        self.states_layer = nn.Linear(9, self.d_model)

        self.img_input_dim = img_input_dim
        self.img_out_dim = img_out_dim
        self.img_out_len = self.img_out_dim[0] * self.img_out_dim[1]

        self.img_tgt_input_dim = (500, 500)
        tgt_pool_len = list(math.ceil(input_ / out_) for input_, out_ in zip(self.img_tgt_input_dim, self.img_out_dim))
        tgt_pad_len = list(((out_ * tgt_ - input_) // 2) for out_, tgt_, input_ in zip(self.img_out_dim, tgt_pool_len, self.img_tgt_input_dim))
        self.tgt_pool = nn.MaxPool2d(kernel_size=tgt_pool_len, stride=tgt_pool_len, padding=tgt_pad_len)

        # [9, 500, 500] --> [256, 28, 28]
        self.map_encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=tgt_pool_len, stride=tgt_pool_len, padding=tgt_pad_len),
            nn.Dropout2d(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(self.d_model, self.img_out_len*2)
        self.dropout = nn.Dropout(p=drop_rate)

        self.ref_training_images = None
        self.ref_validation_images = None

    def forward(self, batch):
        src_maps = batch['src_maps'].float()  # torch.FloatTensor: [B, C, 500, 500]
        agent_points = batch['agent_points'].float()  # torch.FloatTensor: [B, S_tgt, 4]
        states_feat = batch['states_feat']  # torch.FloatTensor: [B, S_tgt, 5]

        # att
        bs = src_maps.size(0)
        tgt_seq_len = agent_points.size(1)
        n, m = self.img_out_dim
        # encoder inputs
        imgs_enc = self.map_encoder(src_maps)  # [B, 9, 500, 500] --> [B, 256, 28, 28]
        imgs_enc = imgs_enc.view(bs, self.d_model, n*m).permute(2, 0, 1)  # --> [B, 256, 28*28=784] --> [784, B, 256] == (S_src, B, E)
        enc_inputs = self.dropout(imgs_enc)  # (S_src, B, E)
        # decoder inputs
        dec_inputs = torch.cat((agent_points, states_feat), dim=-1)  # [B, S, 9]
        dec_inputs = self.states_layer(dec_inputs.view(-1, 9)).view(bs, -1, self.d_model).transpose(0,1)  # [S, B, E]

        # points_enc = self.points_layer(agent_points.permute(1, 0, 2).reshape(-1, 2)).view(tgt_seq_len, bs, self.d_model-3)  # [S, B, 2] -> [S*B, 2] -> [S*B, E-3] -> [S, B, E-3]
        # dec_inputs = torch.cat((points_enc, velocities.unsqueeze(0).repeat(tgt_seq_len, 1, 1), accelerations.unsqueeze(0).repeat(tgt_seq_len, 1, 1), 
        #     heading_change_rates.unsqueeze(0).repeat(tgt_seq_len, 1, 1)), dim=-1)  # [S, B, E]
        
        # decode
        dec_outputs = self.transformer(enc_inputs, dec_inputs, tgt_key_padding_mask=batch['tgt_key_padding_mask'])  # --> [S_tgt, B, 256] == [S_tgt, B, E]
        dec_outputs = self.dropout(dec_outputs.view(-1, self.d_model))  # --> [S_tgt*B, E]
        # classify
        cls_outs = self.out_layer(dec_outputs).view(-1, n*m, 2)  # [S_tgt*B, E] --> [S_tgt*B, 784*2]
        soft_outs = F.softmax(cls_outs, dim=-1)  # [S_tgt*B, 784, 2]
        soft_outs = soft_outs.view(tgt_seq_len, bs, n, m, 2)  # [S_tgt, B, 28, 28, 2]
        
        return soft_outs.transpose(1, 0)  # [B, S_tgt, 28, 28, 2]

    def preprocess_batch(self, batch):
        tgt_maps = batch['tgt_maps']  # torch.FloatTensor: [B, S, N, M]
        tgt_maps = self.tgt_pool(tgt_maps)  # torch.FloatTensor: [B, S, 28, 28]
        tgt_min, tgt_max = tgt_maps.min(), tgt_maps.max()
        batch['tgt_maps'] = (tgt_maps - tgt_min) / (tgt_max - tgt_min + 1e-9)  # normalize to [0, 1]
        return batch

    def training_step(self, batch, batch_idx):
        num_agents = batch['num_agents'].astype(np.int0)  # np.array: [B, ]
        num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)  # np.array: [B+1, ]
        # torch.FloatTensor: [[A1, A2, ...], ...] -> [B, A_tot, ...]
        # batch['tgt_maps'] = pad_sequence([batch['tgt_maps'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_maps'].unsqueeze_(1)
        batch['agent_points'] = pad_sequence([batch['agent_points'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['states_feat'] = pad_sequence([batch['states_feat'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_key_padding_mask'] = torch.stack(
            [torch.tensor([0.] * ls + [1.] * (batch['agent_points'].size(1) - ls)) for ls in num_agents]).bool().type_as(batch['tgt_maps'])

        batch = self.preprocess_batch(batch)
        tgt_maps = batch['tgt_maps']  # [A_tot, 1, 28, 28]
        soft_outs = self(batch)  # [B, S, 28, 28, 2]
        outs_, targets_ = torch.cat([soft_outs[b, :na].reshape(-1, 2) for b, na in zip(range(len(num_agents)), num_agents)], dim=0), tgt_maps.reshape(-1).round().long()
        # outs_, targets_ = soft_outs.reshape(-1, 2), tgt_maps.reshape(-1).round().long()
        loss, acc, f1 = self.cal_loss(outs_, targets_), self.cal_acc(outs_, targets_), self.cal_f1(outs_, targets_)
        self.log_dict({'train_loss': loss, 'train_acc': acc, 'train_f1': f1}, sync_dist=True)

        if 'viz' in batch:
            # self.ref_training_images = batch['viz'].mean(1).unsqueeze(1).repeat(1,3,1,1)  # [B, 3, H, W]
            # self.ref_training_images = batch['viz'].mean(1) * 0.8 + 0.2 * 255 * F.interpolate(soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).squeeze(1)
            # self.ref_training_images = self.ref_training_images.unsqueeze(1).repeat(1,3,1,1)

            self.ref_training_images = 0.5*batch['viz'] + F.interpolate(150*soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)
            # self.ref_training_images = torch.cat((batch['viz'], F.interpolate(255*soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)), dim=2)
            # self.ref_training_images = batch['viz']*0.5 + \
            #     batch['viz']*0.5*F.interpolate(soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)

        return loss

    def validation_step(self, batch, batch_idx):
        num_agents = batch['num_agents'].astype(np.int0)  # np.array: [B, ]
        num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)  # np.array: [B+1, ]
        # torch.FloatTensor: [[A1, A2, ...], ...] -> [B, A_tot, ...]
        # batch['tgt_maps'] = pad_sequence([batch['tgt_maps'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_maps'].unsqueeze_(1)
        batch['agent_points'] = pad_sequence([batch['agent_points'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['states_feat'] = pad_sequence([batch['states_feat'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_key_padding_mask'] = torch.stack(
            [torch.tensor([0.] * ls + [1.] * (batch['agent_points'].size(1) - ls)) for ls in num_agents]).bool().type_as(batch['tgt_maps'])

        batch = self.preprocess_batch(batch)
        tgt_maps = batch['tgt_maps']  # [B, S, 28, 28]
        soft_outs = self(batch)  # [B, S, 28, 28, 2]
        outs_, targets_ = torch.cat([soft_outs[b, :na].reshape(-1, 2) for b, na in zip(range(len(num_agents)), num_agents)], dim=0), tgt_maps.reshape(-1).round().long()
        # outs_, targets_ = soft_outs.reshape(-1, 2), tgt_maps.reshape(-1).round().long()
        loss, acc, f1 = self.cal_loss(outs_, targets_), self.cal_acc(outs_, targets_), self.cal_f1(outs_, targets_)
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_f1': f1}, sync_dist=True)

        if 'viz' in batch:
            # viz - torch.tensor: [B, C, H, W], soft_outs - torch.tensor: [B, H, W]
            # sample_imgs = self.get_viz_img(batch['viz'], soft_outs[:, 0, :, :, 1])  # np.array: [B, H, W, C]
            # self.logger.experiment.add_images('val_sources', batch['viz'] * soft_outs[:, 0, :, :, 1], self.global_step, dataformats="NCHW")
            # self.logger.experiment.add_images('val_predicted', sample_imgs, self.global_step, dataformats="NHWC")
            # self.ref_validation_images = batch['viz'].mean(1).unsqueeze(1).repeat(1,3,1,1)  # [B, 3, H, W]
            # o_ = F.interpolate(soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).squeeze(1)
            # self.ref_validation_images = batch['viz'] * 0.2 + (o_ / (o_.max() + 0.00001)) * 255. * 0.8
            # self.ref_validation_images = self.ref_validation_images.unsqueeze(1).repeat(1,3,1,1)


            self.ref_validation_images = 0.5*batch['viz'] + F.interpolate(150*soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)
            # self.ref_validation_images = torch.cat((batch['viz'], F.interpolate(255*soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)), dim=2)
            # self.ref_training_images += 0.5 * 255 * F.interpolate(soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).squeeze(1)  # [B, W, H]
            # self.ref_validation_images[:, 1, :, :].clamp_(max=255)

            # self.ref_validation_images = batch['viz']*0.5 + \
            #     batch['viz']*0.5*F.interpolate(soft_outs[:, 0, :, :, 1].unsqueeze(1), size=batch['viz'].size()[-2:]).repeat(1,3,1,1)

        return loss

    def training_epoch_end(self, outputs):
        if self.ref_training_images is not None:
            self.logger.experiment.add_images('train_predicted', self.ref_training_images, self.global_step, dataformats="NCHW")

    def validation_epoch_end(self, outputs):
        if self.ref_validation_images is not None:
            self.logger.experiment.add_images('val_predicted', self.ref_validation_images, self.global_step, dataformats="NCHW")
    
    def test_step(self, batch, batch_idx):
        num_agents = batch['num_agents'].astype(np.int0)  # np.array: [B, ]
        num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)  # np.array: [B+1, ]
        # torch.FloatTensor: [[A1, A2, ...], ...] -> [B, A_tot, ...]
        # batch['tgt_maps'] = pad_sequence([batch['tgt_maps'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_maps'].unsqueeze_(1)
        batch['agent_points'] = pad_sequence([batch['agent_points'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['states_feat'] = pad_sequence([batch['states_feat'][num_agents_accum[i]:num_agents_accum[i+1]] for i in range(len(num_agents_accum)-1)], batch_first=True)
        batch['tgt_key_padding_mask'] = torch.stack(
            [torch.tensor([0.] * ls + [1.] * (batch['agent_points'].size(1) - ls)) for ls in num_agents]).bool().type_as(batch['tgt_maps'])

        batch = self.preprocess_batch(batch)
        tgt_maps = batch['tgt_maps']  # [B, S, 28, 28]
        soft_outs = self(batch)  # [B, S, 28, 28, 2]
        outs_, targets_ = torch.cat([soft_outs[b, :na].reshape(-1, 2) for b, na in zip(range(len(num_agents)), num_agents)], dim=0), tgt_maps.reshape(-1).round().long()
        # outs_, targets_ = soft_outs.reshape(-1, 2), tgt_maps.reshape(-1).round().long()
        loss, acc, f1 = self.cal_loss(outs_, targets_), self.cal_acc(outs_, targets_), self.cal_f1(outs_, targets_)
        self.log_dict({'test_loss': loss, 'test_acc': acc, 'test_f1': f1}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)

    @staticmethod
    # outputs - [B, 2] / targets - [B]
    def cal_loss(outputs, targets):
        true_mask = (targets == 1)
        false_mask = torch.logical_not(true_mask)

        true_ids = true_mask.nonzero(as_tuple=True)[0]
        false_ids = false_mask.nonzero(as_tuple=True)[0]
        min_len = min(len(true_ids), len(false_ids))

        true_ids = true_ids[torch.randint(len(true_ids), size=(min_len,))]
        false_ids = false_ids[torch.randint(len(false_ids), size=(min_len,))]

        loss = F.cross_entropy(outputs[true_ids], targets[true_ids]) \
            + F.cross_entropy(outputs[false_ids], targets[false_ids])

        # loss = F.cross_entropy(outputs[true_mask], targets[true_mask]) \
        #     + F.cross_entropy(outputs[false_mask], targets[false_mask]) * 4
        return loss

    @staticmethod
    # outputs - [B, 2] / targets - [B]
    def cal_acc(outputs, targets):
        pred = outputs.argmax(1)
        n = pred.size(0)
        tp = ((pred == 1) & (targets == 1)).sum()
        tn = ((pred == 0) & (targets == 0)).sum()
        acc = (tp + tn) / n
        return acc

    @staticmethod
    # outputs - [B, 2] / targets - [B]
    def cal_f1(outputs, targets):
        pred = outputs.argmax(1)
        tp = ((pred == 1) & (targets == 1)).sum()
        fp = ((pred == 1) & (targets == 0)).sum()
        fn = ((pred == 0) & (targets == 1)).sum()
        f1 = (2*tp) / (2*tp + fp + fn) if tp > 0 else 0.
        return f1

    @staticmethod
    # viz - [B, C, H, W], soft_outs - [B, H, W]
    def get_viz_img(viz, soft_outs):
        viz_ = viz.permute(0, 2, 3, 1).cpu().numpy()  # np.array: [B, H, W, C]
        soft_outs_ = soft_outs.cpu().numpy()  # np.array: [B, H, W]
        viz_imgs = [(vz / vz.max()) * 50. + cv2.resize(out_, vz.shape[:2])[..., np.newaxis] * 200. for vz, out_ in zip(viz_, soft_outs_)]  # B * [H, W, C]
        return np.stack(viz_imgs)  # [B, H, W, C]

    @staticmethod
    def cross_entropy(soft_pred, soft_targets):
        return (-soft_targets * torch.log(soft_pred)).sum(dim=1).mean()



class NewModel(BasicPMP):
    def __init__(self, *args):
        # init BasicPMP if you want
        # super(self, BasicPMP).__init__()
        pass  # ...

    def forward(self, batch):
        pass  # ...
        soft_outs = NotImplemented
        return soft_outs  # [B, S_tgt, 28, 28, 2]







if __name__ == "__main__":
    test5()
else:
    pass

