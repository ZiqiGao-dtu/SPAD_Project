import torch
import torch.nn.functional as F
import os
import time
import argparse
from torch.utils.data import DataLoader
from train_GRU import PosePredictor, SequenceDataset, eval_estimator, RelativeTranslationLoss
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import json
import wandb


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = False  # True: reproduce, False: Allow cuDNN to select the fastest algorithm
torch.backends.cudnn.benchmark = True  # set it as False to make sure the reproduce of result, True: faster CNN


# map between idx and sequence name
dataset_idx_to_name = {
    "spad4bit": [
        "dark_exp1",
        "ambient_full",
        "sun_SE_short_exp2"
    ]
}

spad_4bit_test = [
    (r"dark_runs\dark_exp1\spad\4bit",
     r"dark_runs\dark_exp1\Take 2024-09-20 12.09.01 PM_spad_aligned_cleaned_4bit.json"),
    (r"ambient_runs\ambient_full\spad\4bit",
     r"ambient_runs\ambient_full\Take 2024-09-20 11.23.20 AM_spad_aligned_cleaned_4bit.json"),
    (r"sun_SE_short_exp2\spad\4bit",
     r"sun_SE_short_exp2\Take 2024-09-20 11.50.48 AM_spad_aligned_cleaned_4bit.json")
]


def get_sequence_name(dataset_type, idx):
    if dataset_type not in dataset_idx_to_name:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    name_list = dataset_idx_to_name[dataset_type]
    if idx < 0 or idx >= len(name_list):
        raise IndexError(f"Index out of range. {dataset_type} has {len(name_list)} sequences.")
    name = name_list[idx]
    if name is None:
        raise ValueError(f"Sequence index {idx} does not exist for dataset {dataset_type}")
    return name


# ===================== Argument Parser =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Sequence-wise Test PosePredictor")
    '''parser.add_argument("--th_t", type=float, default=0.01,
                        help="Translation threshold for convergence")
    parser.add_argument("--th_r", type=float, default=0.1,
                        help="Rotation threshold (rad) for convergence")'''
    parser.add_argument("dataset_type", type=str, choices=["spad1bit", "spad2bit", "spad4bit"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--init_type", type=str, required=True,
                        choices=["g", "z", "l", "h"],
                        help="Initial state type for sequences: 'gt', 'zero', 'light', or 'heavy'")

    parser.add_argument("--noise", action="store_true",
                        help="Enable noise during evaluation (default: False)")
    parser.add_argument("--occ_prob", type=float, default=0,
                        help="Occlusion probability (default: 0.1)")

    parser.add_argument("--occ_ratio", type=float, default=0,
                        help="Occlusion ratio (default: 0.2)")

    parser.add_argument("--blackout_prob", type=float, default=0,
                        help="Full blackout/whiteout probability (default: 0.05)")

    parser.add_argument("--brightness_prob", type=float, default=0,
                        help="Brightness jitter probability (default: 0.1)")

    parser.add_argument("--brightness_sigma", type=float, default=0,
                        help="Brightness jitter magnitude (default: 0.3)")

    return parser.parse_args()


# ===================== Main =====================
def main():

    args = parse_args()
    dataset_type = args.dataset_type
    model_path = args.model_path
    init_type = args.init_type
    alpha, beta = args.alpha, args.beta

    noise = args.noise
    occ_prob = args.occ_prob
    occ_ratio = args.occ_ratio
    blackout_prob = args.blackout_prob
    brightness_prob = args.brightness_prob
    brightness_sigma = args.brightness_sigma

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.login(key="d3d0b6fd0e3f36b1519a5ed617cd94bb9ac308c8")
    filename = os.path.basename(__file__).replace(".py", "")
    wandb_run = wandb.init(
        project="pose_predictor",
        name=f"GRU_{dataset_type}",
        config={
            "model_name": "GRU",
        }
    )
    wandb.define_metric("occlusion_percent")
    wandb.define_metric(f"{dataset_type}occlusion/*", step_metric="occlusion_percent")
    wandb.define_metric("blackout_percent")
    wandb.define_metric(f"{dataset_type}blackout/*", step_metric="blackout_percent")
    wandb.define_metric("brightness_jitter_percent")
    wandb.define_metric(f"{dataset_type}brightness_jitter/*", step_metric="brightness_jitter_percent")


    # === Load test dataset ===
    test_path = os.path.join("pt324_4bit", "spad4bit_test.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test_list_all = torch.load(test_path)

    train_path = os.path.join("pt324_4bit", f"spad4bit_train.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset files not found for {dataset_type}")
    print(f"Loading {dataset_type} datasets from disk...")
    train_dict = torch.load(train_path)
    r_mean = train_dict["r_mean"].to(device)
    r_std = train_dict["r_std"].to(device)

    # ================= Load model =================
    img_channels = 3 if dataset_type == "orbbec" else 1
    model = PosePredictor(img_channels=img_channels,
                          cnn_out_dim=128,
                          state_dim=8,
                          fusion_dim=256,
                          gru_hidden=256,
                          gru_layers=1,
                          out_dim=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(memory_format=torch.channels_last)

    # ================= warm-up =================
    model.eval()
    C = 3 if dataset_type == "orbbec" else 1
    H, W = 512, 512
    with torch.inference_mode():
        dummy_img = torch.rand(1, img_channels, H, W, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            _ = model.cnn(dummy_img.to(memory_format=torch.channels_last))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    warm_seq = test_list_all[0]
    warm_dataset = SequenceDataset([warm_seq], dataset_type, init_type)
    warm_loader = DataLoader(warm_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    with torch.inference_mode():
        for imgs, state_seq, target in warm_loader:
            imgs = imgs.to(device)
            state_seq = state_seq.to(device)
            target = target.to(device)
            B, T_small = 1, 30
            hidden = model.init_hidden(B, device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                img_feat = model.cnn(imgs[:, 0].to(memory_format=torch.channels_last))
                state_feat = model.state_encoder(state_seq[:, 0])
                fused_t = model.fusion_fc(torch.cat([img_feat, state_feat], dim=-1)).unsqueeze(1)
                for t in range(T_small):
                    gru_out, hidden = model.gru(fused_t, hidden)
                    out_t = model.head(gru_out[:, -1, :])
                    q_norm = F.normalize(out_t[:, :4], dim=1)
                    out_t = torch.cat([q_norm, out_t[:, 4:]], dim=1)
                    if t < T_small - 1:
                        next_state = state_seq[:, t+1].clone()
                        next_state[:, :7] = out_t
                        fused_t = model.fusion_fc(
                            torch.cat([
                                model.cnn(imgs[:, t+1].to(memory_format=torch.channels_last)),
                                model.state_encoder(next_state)
                            ], dim=-1)
                        ).unsqueeze(1)
            break

    total_time_all = 0.0
    total_frames_all = 0
    min_avg_time = float("inf")
    max_avg_time = 0.0

    # ================= Test all sequences individually =================
    if init_type == "g" or init_type == "z" or init_type == "h" or init_type == "l":
        if noise:
            occ_prob_list = [i * 0.1 for i in range(0, 9)] + [0.9, 0.95, 0.98, 1.0]
        else:
            occ_prob_list = [occ_prob]
        all_frame_times = []
        for i, seq_data in enumerate(test_list_all):
            seq_dataset = SequenceDataset([seq_data], dataset_type, init_type)
            seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
            try:
                seq_name = get_sequence_name(dataset_type, i)
            except ValueError:
                seq_name = f"Seq_{i}"  # fallback

            for occ_p in occ_prob_list:
                loss, t_pred, t_true, q_pred, q_true, seq_time = eval_estimator(model, seq_loader, device, r_mean, r_std, alpha=alpha, beta=beta,
                                                                                noise=noise, occ_prob=occ_p, occ_ratio=0.5,
                                                                                blackout_prob=0.0, brightness_prob=0.0,
                                                                                brightness_sigma=0.0, use_cudagraph=True)
                num_frames = len(seq_data["img_paths"])
                avg_time_per_frame = seq_time / num_frames
                all_frame_times.append(avg_time_per_frame)
                translation_loss_fn = RelativeTranslationLoss()
                trans_err = translation_loss_fn(t_pred, t_true).item()
                cos = torch.clamp(torch.abs(torch.sum(q_pred * q_true, dim=-1)), -1.0, 1.0)
                rot_deg = (2 * torch.acos(cos)).mean().item()
                total_time_all   += seq_time
                total_frames_all += num_frames
                min_avg_time = min(min_avg_time, avg_time_per_frame)
                max_avg_time = max(max_avg_time, avg_time_per_frame)
                wandb.log({
                    "occlusion_percent": occ_p * 100.0,
                    f"{dataset_type}occlusion/{seq_name}_translation_error": trans_err,
                    f"{dataset_type}occlusion/{seq_name}_rotation_error": rot_deg,
                })

                print("========================================================================")
                print(f"[Test Result] Dataset={dataset_type} | Seq={seq_name} | Loss for each frame={loss:.6f}")
                print(f"Relative translation error per frame = {trans_err:.8f} | Rotation angular error per frame = {rot_deg:.8f} rad")
                print(f"Pose Estimation time for each frame: {avg_time_per_frame*1000:.3f} ms")

    # ================= Test all sequences individually =================
    if init_type == "g" or init_type == "z" or init_type == "h" or init_type == "l":
        if noise:
            blackout_prob_list = [i * 0.1 for i in range(0, 9)] + [0.9, 0.95, 0.98, 1.0]
        else:
            blackout_prob_list = [blackout_prob]
        all_frame_times = []
        for i, seq_data in enumerate(test_list_all):
            seq_dataset = SequenceDataset([seq_data], dataset_type, init_type)
            seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
            try:
                seq_name = get_sequence_name(dataset_type, i)
            except ValueError:
                seq_name = f"Seq_{i}"  # fallback

            for blackout_p in blackout_prob_list:
                loss, t_pred, t_true, q_pred, q_true, seq_time = eval_estimator(model, seq_loader, device, r_mean, r_std, alpha=alpha, beta=beta,
                                                                                noise=noise, occ_prob=0.0, occ_ratio=0.0,
                                                                                blackout_prob=blackout_p, brightness_prob=0.0,
                                                                                brightness_sigma=0.0, use_cudagraph=True)
                num_frames = len(seq_data["img_paths"])
                avg_time_per_frame = seq_time / num_frames
                all_frame_times.append(avg_time_per_frame)
                translation_loss_fn = RelativeTranslationLoss()
                trans_err = translation_loss_fn(t_pred, t_true).item()
                cos = torch.clamp(torch.abs(torch.sum(q_pred * q_true, dim=-1)), -1.0, 1.0)
                rot_deg = (2 * torch.acos(cos)).mean().item()
                total_time_all   += seq_time
                total_frames_all += num_frames
                min_avg_time = min(min_avg_time, avg_time_per_frame)
                max_avg_time = max(max_avg_time, avg_time_per_frame)
                wandb.log({
                    "blackout_percent": blackout_p * 100.0,
                    f"{dataset_type}blackout/{seq_name}_translation_error": trans_err,
                    f"{dataset_type}blackout/{seq_name}_rotation_error": rot_deg,
                })

                print("========================================================================")
                print(f"[Test Result] Dataset={dataset_type} | Seq={seq_name} | Loss for each frame={loss:.6f}")
                print(f"Relative translation error per frame = {trans_err:.8f} | Rotation angular error per frame = {rot_deg:.8f} rad")
                print(f"Pose Estimation time for each frame: {avg_time_per_frame*1000:.3f} ms")

    # ================= Test all sequences individually =================
    if dataset_type == "spad1bit":
        pass
    else:
        if init_type == "g" or init_type == "z" or init_type == "h" or init_type == "l":
            if noise:
                brightness_prob_list = [i * 0.1 for i in range(0, 9)] + [0.9, 0.95, 0.98, 1.0]
            else:
                brightness_prob_list = [brightness_prob]
            all_frame_times = []
            for i, seq_data in enumerate(test_list_all):
                seq_data = seq_data.copy()
                length = 300
                seq_data["img_paths"] = seq_data["img_paths"][:length]
                seq_data["delta_ts"] = seq_data["delta_ts"][:length]
                seq_data["target_states"] = seq_data["target_states"][:length]
                seq_data["init_qr"] = seq_data["init_qr"]
                seq_dataset = SequenceDataset([seq_data], dataset_type, init_type)
                seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
                try:
                    seq_name = get_sequence_name(dataset_type, i)
                except ValueError:
                    seq_name = f"Seq_{i}"  # fallback

                for brightness_p in brightness_prob_list:
                    loss, t_pred, t_true, q_pred, q_true, seq_time = eval_estimator(model, seq_loader, device, r_mean, r_std, alpha=alpha, beta=beta,
                                                                                    noise=noise, occ_prob=0.0, occ_ratio=0.0,
                                                                                    blackout_prob=0.0, brightness_prob=brightness_p,
                                                                                    brightness_sigma=0.5, use_cudagraph=True)
                    num_frames = len(seq_data["img_paths"])
                    avg_time_per_frame = seq_time / num_frames
                    all_frame_times.append(avg_time_per_frame)
                    translation_loss_fn = RelativeTranslationLoss()
                    trans_err = translation_loss_fn(t_pred, t_true).item()
                    cos = torch.clamp(torch.abs(torch.sum(q_pred * q_true, dim=-1)), -1.0, 1.0)
                    rot_deg = (2 * torch.acos(cos)).mean().item()
                    total_time_all   += seq_time
                    total_frames_all += num_frames
                    min_avg_time = min(min_avg_time, avg_time_per_frame)
                    max_avg_time = max(max_avg_time, avg_time_per_frame)
                    wandb.log({
                        "brightness_jitter_percent": brightness_p * 100.0,
                        f"{dataset_type}brightness_jitter/{seq_name}_translation_error": trans_err,
                        f"{dataset_type}brightness_jitter/{seq_name}_rotation_error": rot_deg,
                    })

                    print("========================================================================")
                    print(f"[Test Result] Dataset={dataset_type} | Seq={seq_name} | Loss for each frame={loss:.6f}")
                    print(f"Relative translation error per frame = {trans_err:.8f} | Rotation angular error per frame = {rot_deg:.8f} rad")
                    print(f"Pose Estimation time for each frame: {avg_time_per_frame*1000:.3f} ms")
            wandb_run.finish()
            overall_avg_time_per_frame = total_time_all / total_frames_all
            print("========================================================================")
            print(f"[Overall] Overall average inference time "
                  f"= {overall_avg_time_per_frame * 1000:.3f} ms")
            print(f"[Overall] Min average time per frame "
                  f"(fastest run) = {min_avg_time * 1000:.3f} ms")
            print(f"[Overall] Max average time per frame "
                  f"(slowest run) = {max_avg_time * 1000:.3f} ms")

if __name__ == "__main__":
    main()
