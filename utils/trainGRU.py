import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import time
from PIL import Image
import json
import os
import numpy as np
from glob import glob
import sys
import random
import argparse
import glob
from torchvision.models import mobilenet_v3_small


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set it as False to make sure the reproduce of result


# Metric for rotation loss
# L2-based loss with ±q double-cover handling and scale normalization -> relative quaternion error
class RelativeQuaternionFrobeniusLoss(nn.Module):
    """
    Relative Frobenius norm loss for quaternions with double cover handling (q ≡ -q)
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: tuple = None
    ) -> torch.Tensor:
        """
        Compute the relative Frobenius norm loss for quaternions, accounting for double cover

        Args:
            pred (torch.Tensor): Predicted quaternions (B, 4)
            target (torch.Tensor): Ground truth quaternions (B, 4)
            bbox (tuple): Bounding box tensor

        Returns:
            torch.Tensor: Relative Frobenius norm loss
        """
        # Compute Frobenius norms for both q and -q representations
        diff_frobenius_q = torch.norm(pred - target, p="fro", dim=1)
        diff_frobenius_neg_q = torch.norm(pred + target, p="fro", dim=1)

        # Choose the smaller difference (closer quaternion)
        min_diff_frobenius = torch.min(diff_frobenius_q, diff_frobenius_neg_q)

        # Compute target norm (same for q and -q)
        target_frobenius = torch.norm(target, p="fro", dim=1)

        # Compute relative errors
        relative_errors = min_diff_frobenius / (target_frobenius + self.epsilon)
        return relative_errors.mean()


# Metric for translation loss
class RelativeTranslationLoss(nn.Module):
    """
    Relative translation loss: norm2(t_groundtruth - t_prediction) / norm2(t_groundtruth)
    Based on the SPEED+ evaluation metric
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the relative translation loss

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor (unused)

        Returns:
            torch.Tensor: Relative translation loss
        """
        diff = target - pred
        # Use optimized vector norm operations for better performance
        diff_norms = torch.linalg.vector_norm(diff, dim=1)
        target_norms = torch.linalg.vector_norm(target, dim=1)
        relative_errors = diff_norms / target_norms
        return relative_errors.mean()


# Dataset format for train set
class TemporalImageDataset(Dataset):
    # Item in dataset = a sequence of images [T,C,H,W];
    # A batch = multiple sequences [B,T,C,H,W]
    def __init__(self, image_seq_list, image_type, init_type, transform=None):
        """
        image_seq_list: list of image sequence
        """
        self.seq = image_seq_list
        self.transform = transform or (lambda x: x)
        self.image_type = image_type
        self.init_type = init_type

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # img_seq: Tensor [T, C, H, W]
        # state_seq: Tensor [T, 8]  (quat(4), xyz(3), delta_t(1))
        # target_next_state: Tensor [7]  (quat(4), xyz(3))  (ground truth for next time)
        item = self.seq[idx]
        img_paths = item["img_paths"]
        imgs = []
        for path in img_paths:
            path = path.replace('\\', '/')
            img = Image.open(path)
            img = img.resize((512, 512))
            img = np.array(img, dtype=np.float32)
            if self.image_type=="spad4bit":
                max_val = 15
                img = torch.tensor(img/max_val, dtype=torch.float32).unsqueeze(0)
            elif self.image_type=="spad2bit":
                max_val = 3
                img = np.floor(img / 4).astype(np.float32)
                img = torch.tensor(img/max_val, dtype=torch.float32).unsqueeze(0)
            elif self.image_type=="spad1bit":
                max_val = 15
                threshold = max_val / 2.0
                img = (img > threshold).astype(np.float32)
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError(f"Unknown image_type: {self.image_type}")
            imgs.append(img)

        img_seq = torch.stack(imgs, dim=0)  # [T, C, H, W]

        state_seq = item["state_seq"].clone()      # [T, 8]
        if self.init_type == "g":
            pass
        elif self.init_type == "z":
            state_seq[0, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=state_seq.dtype)
            state_seq[0, 4:7] = 0.0
        elif self.init_type == "l":
            rot_std = 0.1
            trans_std = 0.05

            q = state_seq[0, :4]  # [0:3] quat
            noise_q = q + torch.randn_like(q) * rot_std
            noise_q = noise_q / noise_q.norm()
            r = state_seq[0, 4:7]  # [4:6] xyz
            noise_r = r + torch.randn_like(r) * trans_std
            state_seq[0, :4] = noise_q
            state_seq[0, 4:7] = noise_r
        elif self.init_type == "h":
            rot_std = 1.0
            trans_std = 0.5

            q = state_seq[0, :4]  # [0:3] quat
            noise_q = q + torch.randn_like(q) * rot_std
            noise_q = noise_q / noise_q.norm()
            r = state_seq[0, 4:7]  # [4:6] xyz
            noise_r = r + torch.randn_like(r) * trans_std
            state_seq[0, :4] = noise_q
            state_seq[0, 4:7] = noise_r
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")

        # target next state
        target_state = item["target_state"]  # [7]
        gt_init = item["state_seq"][0].clone()
        return img_seq, state_seq, target_state, gt_init


# Dataset format for val/test set
class SequenceDataset(Dataset):
    """
    Sequence-wise dataset for validation/test.
    Each item = full sequence of images [T, C, H, W] + delta_ts [T].
    state_seq is dynamically generated using GRU predictions.
    """
    def __init__(self, sequence_list, image_type, init_type, transform=None):
        self.sequences = sequence_list
        self.transform = transform or (lambda x: x)
        self.image_type = image_type
        self.init_type = init_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        img_paths = item["img_paths"]
        delta_ts = item["delta_ts"]
        target_states = item["target_states"]  # [T,7]
        imgs = []
        for path in img_paths:
            path = path.replace('\\', '/')
            img = Image.open(path)
            img = img.resize((512, 512))
            img = np.array(img, dtype=np.float32)
            if self.image_type=="spad4bit":
                max_val = 15
                img = torch.tensor(img/max_val, dtype=torch.float32).unsqueeze(0)
            elif self.image_type=="spad2bit":
                max_val = 3
                img = np.floor(img / 4).astype(np.float32)
                img = torch.tensor(img/max_val, dtype=torch.float32).unsqueeze(0)
            elif self.image_type=="spad1bit":
                max_val = 15
                threshold = max_val / 2.0
                img = (img > threshold).astype(np.float32)
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError(f"Unknown image_type: {self.image_type}")
            imgs.append(img)
        img_seq = torch.stack(imgs, dim=0)
        delta_ts = torch.tensor(delta_ts, dtype=torch.float32).unsqueeze(-1)
        # Reads all images into a tensor, and initializes state_seq to zeros.
        T = img_seq.size(0)
        state_seq = torch.zeros(T, 8, dtype=torch.float32)
        state_seq[0:, -1] = delta_ts[0:,0]
        state_seq[0, :7] = item['init_qr'].clone()

        if self.init_type == "g":
            pass
        elif self.init_type == "z":
            state_seq[0, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=state_seq.dtype)
            state_seq[0, 4:7] = 0.0
        elif self.init_type == "l":
            desired_angle_rad = 0.3
            desired_rel_t = 0.065

            q = state_seq[0, :4]
            r = state_seq[0, 4:7]

            # -------------------
            axis = torch.randn(3)
            axis = axis / axis.norm()
            half_angle = desired_angle_rad / 2
            q_offset = torch.cat([torch.tensor([math.cos(half_angle)], dtype=q.dtype),
                                  axis * math.sin(half_angle)])
            w0, x0, y0, z0 = q
            w1, x1, y1, z1 = q_offset
            noise_q = torch.tensor([
                w0*w1 - x0*x1 - y0*y1 - z0*z1,
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1
            ], dtype=q.dtype)
            noise_q = noise_q / noise_q.norm()

            # -------------------
            rand_dir = torch.randn(3)
            rand_dir = rand_dir / rand_dir.norm()
            r_norm = r.norm()
            noise_r = r + rand_dir * r_norm * desired_rel_t
            # -------------------
            state_seq[0, :4] = noise_q
            state_seq[0, 4:7] = noise_r
        elif self.init_type == "h":
            desired_angle_rad = 1.83
            desired_rel_t = 0.65

            q = state_seq[0, :4]
            r = state_seq[0, 4:7]

            # -------------------
            axis = torch.randn(3)
            axis = axis / axis.norm()
            half_angle = desired_angle_rad / 2
            q_offset = torch.cat([torch.tensor([math.cos(half_angle)], dtype=q.dtype),
                                  axis * math.sin(half_angle)])
            w0, x0, y0, z0 = q
            w1, x1, y1, z1 = q_offset
            noise_q = torch.tensor([
                w0*w1 - x0*x1 - y0*y1 - z0*z1,
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1
            ], dtype=q.dtype)
            noise_q = noise_q / noise_q.norm()

            # -------------------
            rand_dir = torch.randn(3)
            rand_dir = rand_dir / rand_dir.norm()
            r_norm = r.norm()
            noise_r = r + rand_dir * r_norm * desired_rel_t
            # -------------------
            state_seq[0, :4] = noise_q
            state_seq[0, 4:7] = noise_r
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")
        return img_seq, state_seq, target_states


# Convert tensor format into Dataset format
def prepare_dataloader(train_list, val_list, image_type, init_type, batch_size=8):
    train_dataset = TemporalImageDataset(train_list, image_type, init_type)
    val_dataset = SequenceDataset(val_list, image_type, init_type)
    # data already shuffled in the .pt files, no need to shuffle again here
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Use batch_size=1 to process one full sequence at a time, and num_workers=2 to reduce overhead during validation.
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return dataloader_train, dataloader_val


# *** CNN ***
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_dim=128):
        super().__init__()
        # Using stride instead of max/avg pool to improve efficiency
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1)  # /2
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # /4

        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)  # /8
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0) # /16

        self.conv5 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # /32   16*16
        #self.reduce = nn.Conv2d(128, 8, kernel_size=1)
        self.reduce = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),         # 32*8*8
            nn.ReLU(),
        )
        # Compress each channel into a scalar to keep a lightweight FC, nd independent of the input size.
        # Fully connected layer aggregates channel information into a fused representation
        self.fc = nn.Linear(32 * 8 * 8, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [N, C, H, W], where N = B*T
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # [N, out_dim]
        return x


# ResNet18
'''
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_dim=128, pretrained=True):
        super().__init__()
        assert in_channels in [1, 3], "ResNet18 easiest with 1 or 3 channels."

        # 1) Load resnet18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        backbone = models.resnet18(weights=weights)

        # 2) Adapt first conv for 1-channel input (if needed)
        if in_channels == 1:
            old_conv = backbone.conv1  # Conv2d(3,64,7,stride=2,padding=3,bias=False)
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if pretrained:
                    # Map RGB weights -> Gray weights
                    # shape: [64,3,7,7] -> [64,1,7,7]
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
            backbone.conv1 = new_conv

        # 3) Remove the classification head, keep feature extractor
        # resnet18: output feature dim is 512 after avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # until avgpool included -> [N,512,1,1]
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: [N, C, H, W]
        feat = self.backbone(x)            # [N, 512, 1, 1]
        feat = torch.flatten(feat, 1)      # [N, 512]
        out = self.proj(feat)              # [N, out_dim]
        return out
'''





# Pose predictor with GRU
# GRU automatically computes gates based on the input fused and hidden state hidden
# Two-layer GRU for capturing more complex and long-term temporal dependencies
class PosePredictor(nn.Module):
    def __init__(self,
                 img_channels=1,
                 cnn_out_dim=128,
                 state_dim=8,
                 fusion_dim=256,
                 gru_hidden=256,
                 gru_layers=2,
                 out_dim=7):  # 4 quat + 3 trans
        super().__init__()
        # CNN
        self.cnn = CNN(in_channels=img_channels, out_dim=cnn_out_dim)

        self.state_encoder = StateEncoder(in_dim=state_dim, out_dim=128)

        # fusion: CNN feature + q,r + delta_t
        self.fusion_fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 128, fusion_dim),
            nn.ReLU()
        )
        # GRU: input is fusion_dim
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=gru_hidden,
                          num_layers=gru_layers, batch_first=True)
        # output MLP
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden, device=device)


class StateEncoder(nn.Module):
    def __init__(self, in_dim=8, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def add_bad_frames(
    imgs,
    occ_prob=0.1,
    occ_ratio=0.2,
    blackout_prob=0.05,
    brightness_prob=0.1,
    brightness_sigma=0.3
):
    """
    imgs: [B, T, C, H, W]
    return: imgs_noisy same shape
    """
    B, T, C, H, W = imgs.shape
    imgs = imgs.clone()

    # ========== 1) Random occlusion ==========
    for b in range(B):
        for t in range(T):
            if torch.rand(1).item() < occ_prob:
                h = int(H * occ_ratio)
                w = int(W * occ_ratio)
                y0 = torch.randint(0, H - h + 1, (1,))
                x0 = torch.randint(0, W - w + 1, (1,))
                imgs[b, t, :, y0:y0+h, x0:x0+w] = 0.0

    # ========== 2) random blackout ==========
    for b in range(B):
        for t in range(T):
            if torch.rand(1).item() < blackout_prob:
                if torch.rand(1).item() < 0.5:
                    imgs[b, t] *= 0
                else:
                    imgs[b, t] = 1.0

    # ========== 3) brightness jitter ==========
    for b in range(B):
        for t in range(T):
            if torch.rand(1).item() < brightness_prob:
                factor = float(1.0 + brightness_sigma * torch.randn(1))
                imgs[b, t] = torch.clamp(imgs[b, t] * factor, 0.0, 1.0)

    return imgs


# Auto-regressive training loop
def train_epoch_autoreg(model, dataloader, optimizer, device, alpha=1.0, beta=1.0, scaler=None, dataset_type=None):
    """
    Autoregressive training within each window.
    alpha: weight for quaternion loss
    beta: weight for translation loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    print_every = 500

    rotation_loss_fn = RelativeQuaternionFrobeniusLoss().to(device)
    translation_loss_fn = RelativeTranslationLoss().to(device)

    for batch_idx, (imgs, states, target_next, _) in enumerate(dataloader):
        if (not torch.isfinite(imgs).all()) or (not torch.isfinite(states).all()) or (not torch.isfinite(target_next).all()):
            print("Found NaN or inf in training batch inputs! Skipping this batch.")
            continue

        imgs = imgs.to(device).float()       # [B, T, C, H, W]

        if dataset_type == "spad1bit":
            imgs = add_bad_frames(
                imgs,
                occ_prob=0.1,
                occ_ratio=0.2,
                blackout_prob=0.05,
                brightness_prob=0.0,
                brightness_sigma=0.0
            )
        else:
            imgs = add_bad_frames(
                imgs,
                occ_prob=0.1,
                occ_ratio=0.2,
                blackout_prob=0.05,
                brightness_prob=0.1,
                brightness_sigma=0.3
            )

        states = states.to(device).float()   # [B, T, 8]  (q, r, delta_t)
        target_next = target_next.to(device).float()  # [B, 7]
        B, T = imgs.shape[0], imgs.shape[1]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            # --- CNN feature extraction ---
            imgs_flat = imgs.view(B * T, imgs.size(2), imgs.size(3), imgs.size(4))
            img_feats_flat = model.cnn(imgs_flat)                  # [B*T, feat_dim]
            img_feats = img_feats_flat.view(B, T, -1)              # [B, T, feat_dim]

            # --- Initialize hidden state ---
            hidden = model.init_hidden(B, device)

            # --- Initial input state (first frame) ---
            state_t = states[:, 0, :]  # [B, 8]
            state_feat_t = model.state_encoder(state_t)
            fused_t = torch.cat([img_feats[:, 0, :], state_feat_t], dim=-1)
            fused_t = model.fusion_fc(fused_t).unsqueeze(1)

            pred_seq = torch.zeros(B, T, 7, device=device)

            # --- Autoregressive rollout inside this window ---
            for t in range(T):
                gru_out, hidden = model.gru(fused_t, hidden)
                out_t = model.head(gru_out[:, -1, :])  # [B, 7]

                # --- Normalize quaternion part ---
                q_normalized = F.normalize(out_t[:, :4], p=2, dim=1, eps=1e-8)
                out_t = torch.cat([q_normalized, out_t[:, 4:]], dim=1)

                pred_seq[:, t, :] = out_t

                if t < T - 1:
                    next_state = states[:, t + 1, :].clone()
                    next_state[:, :7] = out_t  # Replace ground-truth pose (q,r) with model's prediction for autoregressive input
                    state_feat_next = model.state_encoder(next_state)
                    fused_next = torch.cat([img_feats[:, t + 1, :], state_feat_next], dim=-1)
                    fused_t = model.fusion_fc(fused_next).unsqueeze(1)

            # --- Compute loss over the entire window ---
            gt_seq = torch.cat([states[:, 1:, :7], target_next.unsqueeze(1)], dim=1)  # Ground truth q,r for all steps

            pred_q_flat = pred_seq[:, :, :4].reshape(B*T, 4)
            gt_q_flat   = gt_seq[:, :, :4].reshape(B*T, 4)
            pred_t_flat = pred_seq[:, :, 4:7].reshape(B*T, 3)
            gt_t_flat   = gt_seq[:, :, 4:7].reshape(B*T, 3)

            loss_q = rotation_loss_fn(pred_q_flat, gt_q_flat)
            loss_t = translation_loss_fn(pred_t_flat, gt_t_flat)

            loss = alpha * loss_q + beta * loss_t

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B

        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1 == num_batches) or (batch_idx == 4):
            avg_loss = total_loss / ((batch_idx + 1) * B)
            print(f"  [Batch {batch_idx+1}/{num_batches}]  Current Batch Loss={loss.item():.6f}  (Cumulative Avg Loss so far={avg_loss:.6f})")

    return total_loss / len(dataloader.dataset)


class CUDAGraphPoseStepRunner:
    def __init__(self, model: PosePredictor, device, img_shape, state_dim=8):
        self.model = model
        self.device = device
        self.state_dim = state_dim

        self.C, self.H, self.W = img_shape
        B = 1

        self.img_buf = torch.empty(
            (B, self.C, self.H, self.W),
            device=device
        ).to(memory_format=torch.channels_last)

        self.state_buf  = torch.empty(B, state_dim, device=device)
        self.hidden_buf = model.init_hidden(B, device)  # [num_layers, B, hidden_dim]
        self.pose_buf   = torch.empty(B, 7, device=device)

        self.graph = torch.cuda.CUDAGraph()

        self.img_buf.zero_()
        self.state_buf.zero_()
        self.hidden_buf.zero_()

        torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            with torch.amp.autocast(device_type='cuda'):
                img_feat = self.model.cnn(self.img_buf)
                state_feat = self.model.state_encoder(self.state_buf)
                fused_t = torch.cat([img_feat, state_feat], dim=-1)
                fused_t = self.model.fusion_fc(fused_t).unsqueeze(1)  # [B,1,fusion_dim]

                gru_out, hidden_next = self.model.gru(fused_t, self.hidden_buf)  # [B,1,H]

                out_t = self.model.head(gru_out[:, -1, :])  # [B,7]

                q_normalized = F.normalize(out_t[:, :4], p=2, dim=1, eps=1e-8)
                out_t = torch.cat([q_normalized, out_t[:, 4:]], dim=1)

                self.pose_buf.copy_(out_t)
                self.hidden_buf.copy_(hidden_next)

    def reset_hidden(self):
        self.hidden_buf.zero_()

    def step(self, img_t, state_t):
        self.img_buf.copy_(img_t.to(memory_format=torch.channels_last))
        self.state_buf.copy_(state_t)

        self.graph.replay()

        pose_t = self.pose_buf.clone()
        return pose_t


def eval_estimator(model, dataloader, device, r_mean, r_std, alpha=1.0, beta=1.0,
                   noise=False, occ_prob=0.1, occ_ratio=0.2, blackout_prob=0.05,
                   brightness_prob=0.1, brightness_sigma=0.3,
                   use_cudagraph=True):
    model.eval()
    total_loss = 0.0
    all_t_pred_real = []
    all_t_true_real = []
    all_q_pred = []
    all_q_true = []

    rotation_loss_fn = RelativeQuaternionFrobeniusLoss().to(device)
    translation_loss_fn = RelativeTranslationLoss().to(device)

    cg_runner = None

    with torch.inference_mode():
        for imgs, state_seq, target_states in dataloader:
            imgs = imgs.to(device).float()          # [B, T, C, H, W]，B=1
            state_seq = state_seq.to(device).float()
            target_states = target_states.to(device).float()
            B, T, C, H, W = imgs.shape
            if use_cudagraph and cg_runner is None:
                cg_runner = CUDAGraphPoseStepRunner(
                    model=model,
                    device=device,
                    img_shape=(C, H, W),
                    state_dim=state_seq.shape[-1]
                )

            pred_seq = torch.zeros(B, T, 7, device=device)

            if noise:
                imgs = add_bad_frames(
                    imgs,
                    occ_prob=occ_prob,
                    occ_ratio=occ_ratio,
                    blackout_prob=blackout_prob,
                    brightness_prob=brightness_prob,
                    brightness_sigma=brightness_sigma
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seq_start = time.perf_counter()

            if use_cudagraph and cg_runner is not None:
                cg_runner.reset_hidden()

                next_state = None
                for t in range(T):
                    if t == 0:
                        cur_state = state_seq[:, 0, :]          # [1,8]
                    else:
                        cur_state = next_state                   # [1,8]

                    img_t = imgs[:, t, :, :, :]                 # [1,C,H,W]

                    pose_t = cg_runner.step(img_t, cur_state)   # [1,7]
                    pred_seq[:, t, :] = pose_t

                    if t < T - 1:
                        next_state = state_seq[:, t+1, :].clone()
                        next_state[:, :7] = pose_t

            else:
                hidden = model.init_hidden(B, device)

                with torch.amp.autocast(device_type='cuda'):
                    img_feat = model.cnn(imgs[:, 0, :, :, :].to(memory_format=torch.channels_last))
                    state_feat_0 = model.state_encoder(state_seq[:, 0, :])
                    fused_t = torch.cat([img_feat, state_feat_0], dim=-1)
                    fused_t = model.fusion_fc(fused_t).unsqueeze(1)  # [B,1,fusion_dim]

                    for t in range(T):
                        gru_out, hidden = model.gru(fused_t, hidden)
                        out_t = model.head(gru_out[:, -1, :])

                        q_normalized = F.normalize(out_t[:, :4], p=2, dim=1, eps=1e-8)
                        out_t = torch.cat([q_normalized, out_t[:, 4:]], dim=1)
                        pred_seq[:, t, :] = out_t

                        if t < T - 1:
                            next_state = state_seq[:, t + 1, :].clone()
                            next_state[:, :7] = out_t
                            state_feat_next = model.state_encoder(next_state)
                            img_feat_next = model.cnn(imgs[:, t + 1, :, :, :].to(memory_format=torch.channels_last))
                            fused_next = torch.cat([img_feat_next, state_feat_next], dim=-1)
                            fused_t = model.fusion_fc(fused_next).unsqueeze(1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seq_end = time.perf_counter()

            pred_q_flat = pred_seq[:, :, :4].reshape(B*T, 4)
            gt_q_flat   = target_states[:, :, :4].reshape(B*T, 4)
            pred_t_flat = pred_seq[:, :, 4:7].reshape(B*T, 3)
            gt_t_flat   = target_states[:, :, 4:7].reshape(B*T, 3)

            loss_q = rotation_loss_fn(pred_q_flat, gt_q_flat)
            loss_t = translation_loss_fn(pred_t_flat, gt_t_flat)
            loss = alpha * loss_q + beta * loss_t
            total_loss += loss.item() * B

            # ---- De-standardize translation ----
            t_pred_real = pred_seq[:, :, 4:7] * r_std + r_mean
            t_true_real = target_states[:, :, 4:7] * r_std + r_mean
            t_pred_real = t_pred_real.reshape(-1, 3)
            t_true_real = t_true_real.reshape(-1, 3)
            all_t_pred_real.append(t_pred_real)
            all_t_true_real.append(t_true_real)

            q_pred = pred_seq[:, :, :4].reshape(-1, 4)
            q_true = target_states[:, :, :4].reshape(-1, 4)
            all_q_pred.append(q_pred)
            all_q_true.append(q_true)

    all_t_pred_real = torch.cat(all_t_pred_real, dim=0)
    all_t_true_real = torch.cat(all_t_true_real, dim=0)
    all_q_pred = torch.cat(all_q_pred, dim=0)
    all_q_true = torch.cat(all_q_true, dim=0)

    seq_time = seq_end - seq_start

    return total_loss / len(dataloader.dataset), all_t_pred_real, all_t_true_real, all_q_pred, all_q_true, seq_time


# Main function
def main():
    args = parse_args()
    dataset_type = args.dataset_type
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    alpha = args.alpha
    beta = args.beta
    init_type = args.init_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Load .pt dataset =====
    train_path = os.path.join("pt324_4bit", "spad4bit_train.pt")
    val_path = os.path.join("pt324_4bit", "spad4bit_val.pt")

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(f"Dataset files not found for {dataset_type}")

    print(f"Loading {dataset_type} datasets from disk...")
    train_dict = torch.load(train_path)
    train_list = train_dict["data"]
    r_mean = train_dict["r_mean"].to(device)
    r_std = train_dict["r_std"].to(device)
    val_list = torch.load(val_path)  # list of list of dicts

    img_channels = 1

    dataloader_train, dataloader_val = prepare_dataloader(train_list, val_list, dataset_type, init_type, batch_size=batch_size)

    model = PosePredictor(img_channels=img_channels,
                          cnn_out_dim=128,
                          state_dim=8,
                          fusion_dim=256,
                          gru_hidden=256,
                          gru_layers=1,
                          out_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Automatically adjusts the learning rate during training
    scaler = torch.cuda.amp.GradScaler()   # Mixed Precision Training

    best_val_loss = float("inf")
    best_model_path = None

    # Early Stopping
    if dataset_type == "spad1bit":
        patience = 100
    else:
        patience = 3
    no_improve_count = 0

    for ep in range(epochs):
        t0 = time.time()

        # The first epoch uses mixed
        train_loss = train_epoch_autoreg(model, dataloader_train, optimizer, device, alpha=alpha, beta=beta, scaler=scaler, dataset_type=dataset_type)

        val_loss, _, _, _, _, _ = eval_estimator(model, dataloader_val, device, r_mean, r_std, alpha=alpha, beta=beta, noise=False)
        t1 = time.time()
        print(f"Epoch {ep+1}/{epochs} | Train Loss (per window)={train_loss:.6f} | Val Loss (per seq)={val_loss:.6f} | Time={(t1-t0):.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            no_improve_count = 0
            os.makedirs(f"bestModel_{script_name}_{dataset_type}_{init_type}", exist_ok=True)

            old_models = glob.glob(f"bestModel_{script_name}_{dataset_type}_{init_type}/*.pth")
            '''for m in old_models:
                os.remove(m)'''  # eenable this if you only want to save the best checkpoint

            best_val_loss = val_loss
            save_timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"bestModel_{script_name}_{dataset_type}_{init_type}"
            best_model_path = os.path.join(save_dir, f"{dataset_type}_best_epoch{ep+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated: {best_model_path} (val_loss={val_loss:.6f})")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"No improvement for {patience} epochs, early stopping...")
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Train PosePredictor on image sequences")
    parser.add_argument("dataset_type", type=str,
                        choices=["spad1bit", "spad2bit", "spad4bit"],
                        help="Which dataset to train on")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for quaternion loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for translation loss")
    parser.add_argument("--init_type", type=str, required=True,
                        choices=["g", "z", "l", "h"],
                        help="Initial state type for sequences: 'gt', 'zero', 'light', or 'heavy'")
    return parser.parse_args()


# Usage Example: python train_CNN_GRU.py orbbec --batch_size 8 --epochs 3 --lr 1e-3 --alpha 1.0 --beta 1.0
if __name__ == "__main__":
    main()
