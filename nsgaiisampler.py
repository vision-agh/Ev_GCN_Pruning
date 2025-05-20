# optimise.py
import copy
import time
import optuna
import math
from optuna.samplers import NSGAIISampler, NSGAIIISampler
from optuna.pruners  import SuccessiveHalvingPruner       # wczesne zatrzymanie

import torch
import lightning as L
from omegaconf import OmegaConf

from models.recognition import LNRecognition
from data.mnist         import MNIST
from utils.structured_pruning import structured_pruning

# ──────────────────────────────────────────────────────────────────────────────
#  1. Stałe – baza kanałów oraz pomocnicze funkcje
# ──────────────────────────────────────────────────────────────────────────────
BASE_CFG_PATH   = "configs/mnist.yaml"
CKPT_PATH       = "checkpoints/mnist-dvs_3.ckpt"

# Jeśli liczba BRAM = channels_after_pruning × bits – zmień tu,
# jeżeli Twój FPGA liczy inaczej (np. BLOCK_RAM_WIDTH).
def layer_bram(ch_after_prune: int, bits: int) -> int:
    return math.ceil( (ch_after_prune * bits + 18) / 18 ) / 2

# ──────────────────────────────────────────────────────────────────────────────
#  2. Dane, model FP32 – ładujemy raz, by dataloader nie powtarzał się w każdym trialu
# ──────────────────────────────────────────────────────────────────────────────
L.seed_everything(42, workers=True)
base_cfg = OmegaConf.load(BASE_CFG_PATH)
dm = MNIST(base_cfg)
dm.setup()

test_loader = dm.test_dataloader()          # ↔ 1 iterator dla wszystkich triali

# ──────────────────────────────────────────────────────────────────────────────
#  3. Funkcja celu dla Optuny
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial):

    # --- 3.1 Wylosuj parametry ------------------------------------------------
    cfg = copy.deepcopy(base_cfg)  # NIE zmieniamy globalnego cfg

    bram_total = 0
    pruning_plan = []   # (mod, keep_ratio) – przyda się za chwilę

    for idx, layer_name in enumerate(["conv1", "conv2", "conv3", "conv4", "conv5"]):
        layer_cfg   = cfg[layer_name]
        base_ch     = layer_cfg.out_channels

        if layer_name == "conv1":
            # dla pierwszej warstwy nie ma pruningu
            base_ch     = layer_cfg.out_channels
            bits        = trial.suggest_categorical(f"{layer_name}_bits", [6, 8])
            setattr(layer_cfg, "num_bits", bits)
            bram_total += layer_bram(base_ch, bits)
            continue

        layer_cfg   = cfg[layer_name]
        base_ch     = layer_cfg.out_channels

        bits        = trial.suggest_categorical(f"{layer_name}_bits", [6, 8])
        step        = 3 if bits == 6 else 9
        # K-praunuj liczby kanałów; Optuna zwraca liczbę PRUNED-out, tak jak w Twoim kodzie
        pruned_out  = trial.suggest_int(f"{layer_name}_pruning",
                                        low= step * ((base_ch // step) // 2), high=base_ch, step=step)

        keep_ratio  = (base_ch - pruned_out) / base_ch
        pruning_plan.append((layer_name, keep_ratio))

        setattr(layer_cfg, "num_bits", bits)  # wpisujemy do konfigu
        bram_total += layer_bram(pruned_out, bits)

    # --- 3.2 Załaduj model, przytnij kanały, skalibruj (jak u Ciebie) --------
    model = LNRecognition.load_from_checkpoint(CKPT_PATH, cfg=cfg).cuda()
    model.model.eval()
    model.model.calibrate()

    # structured_pruning dla każdej warstwy
    for layer_name, keep_ratio in pruning_plan:
        structured_pruning(getattr(model.model, layer_name), keep_ratio)

    # --- 3.3 Oceń accuracy  ---------------------------------------------------
    correct, seen = 0, 0
    for batch in test_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        y_hat  = model(batch).argmax(dim=-1)
        y_true = batch["label"]

        correct += (y_hat == y_true).sum().item()
        seen    += y_true.size(0)
    acc = correct / seen

    # ZWRACAMY krotkę (accuracy, bram) – kolejność odpowiada study.directions
    return acc, bram_total

# ──────────────────────────────────────────────────────────────────────────────
#  4. Konfiguracja i uruchomienie eksperymentu
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    L.seed_everything(42, workers=True)


    # sampler = optuna.integration.BoTorchSampler(
    #     n_startup_trials=10,
    #     seed=42,
    # )

    # study = optuna.create_study(
    #     directions=["maximize", "minimize"],
    #     sampler=sampler,
    # )

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=NSGAIISampler(population_size=40, seed=42),
    )

    # n_trials ≈ 5 × population_size × pokolenia; zacznij od 500 –1000
    study.optimize(objective, n_trials=800, timeout=None, show_progress_bar=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  5. Zapis Pareto-frontu
    # ──────────────────────────────────────────────────────────────────────────
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    print("Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")


    # import pandas as pd
    # pareto = optuna.study._multiobjective.multi_objective.optimize._get_pareto_front_trials(study)
    # df = pd.DataFrame([
    #     {**t.params, "accuracy": t.values[0], "brams": t.values[1]} for t in pareto
    # ])
    # df.to_csv("pareto_front.csv", index=False)

    # print("✔ Done – Pareto front zapisany w pareto_front.csv")
