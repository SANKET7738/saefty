import torch
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict

from saefty.models.sae.base import BaseSAE
from saefty.models.sae.activation_store import ActivationStore


class TrainerConfig(BaseModel):
    lr: float = 2e-4
    warmup_steps: int = 1000
    log_every: int = 100
    checkpoint_every: int = 5000
    output_dir: str = "results/train_sae"


class SAETrainer:
    def __init__(self, sae: BaseSAE, store: ActivationStore, config: TrainerConfig):
        self.sae = sae
        self.store = store
        self.config = config
        self.history: List[Dict[str, float]] = []


    def _get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.lr * (step + 1) / self.config.warmup_steps
        return self.config.lr


    def train(self) -> List[Dict[str, float]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = self.sae.to(device)

        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.config.lr)

        print(f"starting training: lr={self.config.lr}, warmup={self.config.warmup_steps}")
        print(f"sae: {self.sae.d_model} → {self.sae.d_sae} features")

        for step, batch in enumerate(self.store):
            # warmup lr
            lr = self._get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = batch.to(device)
            x_hat, features = self.sae(batch)
            losses = self.sae.loss(batch, x_hat, features)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            self.sae.normalize_decoder()

            # compute L0 and dead fraction
            l0 = (features > 0).float().sum(dim=-1).mean().item()
            dead_fraction = (features.sum(dim=0) == 0).float().mean().item()

            record = {
                "step": step,
                "lr": lr,
                "l0": l0,
                "dead_fraction": dead_fraction,
            }
            for k, v in losses.items():
                record[k] = v.item()
            self.history.append(record)

            if step % self.config.log_every == 0:
                print(f"step {step:>6d} | loss={losses['loss']:.4f} | "
                      f"L0={l0:.1f} | dead={dead_fraction:.3f} | lr={lr:.2e}")

            if self.config.checkpoint_every and (step + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)

        print(f"training complete: {len(self.history)} steps")
        return self.history


    def _save_checkpoint(self, step: int):
        out = Path(self.config.output_dir) / ".cache"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"sae_step_{step}.pt"
        torch.save(self.sae.state_dict(), path)
        print(f"checkpoint saved: {path}")


    def save_results(self, extra_metrics: Optional[Dict] = None):
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # training log
        log_path = out / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"training log saved: {log_path}")

        # final checkpoint
        cache = out / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae.state_dict(), cache / "sae_final.pt")
        print(f"final weights saved: {cache / 'sae_final.pt'}")

        # summary metrics
        if self.history:
            last = self.history[-1]
            metrics = {
                "final_loss": last.get("loss"),
                "final_reconstruction_loss": last.get("reconstruction_loss"),
                "final_l0": last.get("l0"),
                "final_dead_fraction": last.get("dead_fraction"),
                "total_steps": len(self.history),
            }
            if extra_metrics:
                metrics.update(extra_metrics)
            metrics_path = out / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"metrics saved: {metrics_path}")
