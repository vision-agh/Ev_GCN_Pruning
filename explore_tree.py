import lightning as L
import torch
import numpy as np
from omegaconf import OmegaConf

from models.recognition import LNRecognition
from data.mnist import MNIST
from utils.structured_pruning import structured_pruning
from utils.precompute_space import precompute_space


def main():
    # ───────────────────────────── baseline ─────────────────────────────
    L.seed_everything(42, workers=True)
    cfg = OmegaConf.load('configs/mnist.yaml')
    dm = MNIST(cfg)
    dm.setup()

    baseline_model = LNRecognition.load_from_checkpoint(
        'checkpoints/mnist-dvs_3.ckpt', cfg=cfg
    ).cuda()
    baseline_model.model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for batch in dm.test_dataloader():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            preds = torch.argmax(baseline_model(batch), dim=-1)
            correct += (preds == batch['label']).sum().item()
            total += batch['label'].size(0)
    print(f"Baseline accuracy: {correct/total:.4f}")

    # ───────────────────── prepare design space sorted by BRAM ─────────────────────
    precomputed_space = precompute_space(cfg, depth_start=0, depth_end=100)
    sorted_precomputed_space = []
    for layer in precomputed_space:
        sorted_idx = np.argsort(layer['bram_space'])[::-1]
        layer['pruning_space'] = np.array(layer['pruning_space'])[sorted_idx].tolist()
        layer['bit_space']     = np.array(layer['bit_space'])[sorted_idx].tolist()
        layer['bram_space']    = np.array(layer['bram_space'])[sorted_idx].tolist()
        sorted_precomputed_space.append(layer)

    # unpack
    c1_pruning, c2_pruning, c3_pruning, c4_pruning, c5_pruning = [sorted_precomputed_space[i]['pruning_space'] for i in range(5)]
    c1_bits,    c2_bits,    c3_bits,    c4_bits,    c5_bits    = [sorted_precomputed_space[i]['bit_space']     for i in range(5)]
    c1_bram,    c2_bram,    c3_bram,    c4_bram,    c5_bram    = [sorted_precomputed_space[i]['bram_space']    for i in range(5)]

    max_lens     = [len(c1_bits), len(c2_bits), len(c3_bits), len(c4_bits), len(c5_bits)]
    layer_names  = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    bram_lists   = [c1_bram, c2_bram, c3_bram, c4_bram, c5_bram]

    # ───────────────────── helper: evaluate given configuration ─────────────────────
    def evaluate_at(idx):
        """Return (accuracy, brams) for configuration described by 5‑element idx list."""
        cfg.conv1.num_bits = c1_bits[idx[0]]
        cfg.conv2.num_bits = c2_bits[idx[1]]
        cfg.conv3.num_bits = c3_bits[idx[2]]
        cfg.conv4.num_bits = c4_bits[idx[3]]
        cfg.conv5.num_bits = c5_bits[idx[4]]

        model = LNRecognition.load_from_checkpoint(
            'checkpoints/mnist-dvs_3.ckpt', cfg=cfg
        ).cuda()
        model.model.eval()
        model.model.calibrate()

        structured_pruning(model.model.conv1, (cfg.conv1.out_channels - c1_pruning[idx[0]])/cfg.conv1.out_channels)
        structured_pruning(model.model.conv2, (cfg.conv2.out_channels - c2_pruning[idx[1]])/cfg.conv2.out_channels)
        structured_pruning(model.model.conv3, (cfg.conv3.out_channels - c3_pruning[idx[2]])/cfg.conv3.out_channels)
        structured_pruning(model.model.conv4, (cfg.conv4.out_channels - c4_pruning[idx[3]])/cfg.conv4.out_channels)
        structured_pruning(model.model.conv5, (cfg.conv5.out_channels - c5_pruning[idx[4]])/cfg.conv5.out_channels)

        brams = (c1_bram[idx[0]] + c2_bram[idx[1]] + c3_bram[idx[2]]
                 + c4_bram[idx[3]] + c5_bram[idx[4]])

        with torch.no_grad():
            correct, total = 0, 0
            for batch in dm.test_dataloader():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                preds = torch.argmax(model(batch), dim=-1)
                correct += (preds == batch['label']).sum().item()
                total   += batch['label'].size(0)
        return correct/total, brams

    # additional helper – indices with equal BRAM as first next option
    def equal_bram_indices(layer_bram, start_idx):
        """Return all indices > start_idx that share BRAM with the immediate next configuration."""
        if start_idx + 1 >= len(layer_bram):
            return []
        target_bram = layer_bram[start_idx + 1]
        return [j for j in range(start_idx + 1, len(layer_bram)) if layer_bram[j] == target_bram]

    # ───────────────────── greedy search ─────────────────────
    current_idx   = [0, 0, 0, 0, 0]
    best_acc, best_bram = evaluate_at(current_idx)
    print(f"Start → idx={current_idx}, acc={best_acc:.4f}, bram={best_bram}")

    while True:
        candidates = []
        # gather candidates: +1 plus wszystkie o tym samym BRAM‑ie
        for i in range(1, 5):
            for idx_option in equal_bram_indices(bram_lists[i], current_idx[i]):
                new_idx      = current_idx.copy()
                new_idx[i]   = idx_option
                acc, bram    = evaluate_at(new_idx)
                candidates.append((acc, bram, i, new_idx))

                print(f"    {layer_names[i]} → idx {idx_option} | acc={acc:.4f} | bram={bram}")
                print(f"    conv1 {c1_pruning[new_idx[0]] } + {c1_bits[new_idx[0]]} bits | conv2 {c2_pruning[new_idx[1]]} + {c2_bits[new_idx[1]]} bits | conv3 {c3_pruning[new_idx[2]]} + {c3_bits[new_idx[2]]} bits | conv4 {c4_pruning[new_idx[3]]} + {c4_bits[new_idx[3]]} bits | conv5 {c5_pruning[new_idx[4]]} + {c5_bits[new_idx[4]]} bits")
                print( "    ───────────────────────────────────────────────────────────────")
        if not candidates:
            print("Wszystkie indeksy osiągnęły maksimum – koniec.")
            break

        acc_c, bram_c, layer_id, idx_c = max(candidates, key=lambda x: x[0])

        # if acc_c <= best_acc:
        #     print("Koniec – dalsze zwiększanie nie poprawia accuracy.")
        #     break

        current_idx = idx_c
        best_acc    = acc_c
        best_bram   = bram_c
        print(f"↑ {layer_names[layer_id]} → idx {current_idx[layer_id]}  | acc={best_acc:.4f} | bram={best_bram}")
        print(f"conv1 {c1_pruning[current_idx[0]]} + {c1_bits[current_idx[0]]} bits | conv2 {c2_pruning[current_idx[1]]} + {c2_bits[current_idx[1]]} bits | conv3 {c3_pruning[current_idx[2]]} + {c3_bits[current_idx[2]]} bits | conv4 {c4_pruning[current_idx[3]]} + {c4_bits[current_idx[3]]} bits | conv5 {c5_pruning[current_idx[4]]} + {c5_bits[current_idx[4]]} bits")

    print("\n──────────────  wynik końcowy  ──────────────")
    print(f"Najlepsza konfiguracja: idx={current_idx}")
    print(f"Accuracy: {best_acc:.4f} | BRAMs: {best_bram}")


if __name__ == "__main__":
    main()
