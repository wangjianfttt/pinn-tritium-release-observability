#!/usr/bin/env python3
"""Multi-condition joint inverse PINN for Li ceramic release parameters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from li_ceramic_bed_1d import Li2TiO3Params, Li4SiO4Params, SharedParams
from li_ceramic_inverse_pinn import MultiplierSet
from li_ceramic_pinn import NeuralNetwork, grad_column, update_adaptive_weights
from multi_condition_pinn import COND_SCALES, load_dataset, plot_prediction, sample_rows, sample_x


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def multiplier_physics_loss(material: str, model: nn.Module, x_phys: torch.Tensor, scale: torch.Tensor, z_max: float, multipliers):
    x = x_phys.clone().detach().requires_grad_(True)
    scale_flat = scale.reshape(-1)
    y = model(x) * scale_flat.reshape(1, -1)
    cols = [y[:, i : i + 1] for i in range(y.shape[1])]
    shared = SharedParams()
    eps = shared.porosity
    d_ax = shared.axial_dispersion_m2_s
    gen = shared.source_rate_mol_m3_s
    humidity = x[:, 2:3] * COND_SCALES["humidity"]
    h2_fraction = x[:, 3:4] * COND_SCALES["h2_fraction"]
    u = x[:, 4:5] * COND_SCALES["superficial_velocity_m_s"]
    t_final = torch.clamp(x[:, 5:6] * COND_SCALES["t_final_s"], min=1.0)

    def dt(v):
        return grad_column(v, x, 1) / t_final

    def dz(v):
        return grad_column(v, x, 0) / z_max

    def dzz(v):
        return grad_column(grad_column(v, x, 0), x, 0) / (z_max**2)

    c_ht, c_hto = cols[0], cols[1]
    source_norm = max(gen, 1.0e-12)
    mean_t_final = float(torch.mean(t_final.detach()).cpu())
    gas_norm_ht = max(float(scale_flat[0].detach().cpu()) / mean_t_final, source_norm)
    gas_norm_hto = max(float(scale_flat[1].detach().cpu()) / mean_t_final, source_norm)

    if material == "li2tio3":
        grain, trap, open_pore, closed = cols[2], cols[3], cols[4], cols[5]
        mat = Li2TiO3Params()
        h2_boost = 0.25 * h2_fraction / (h2_fraction + 1.0e-3)
        humidity_penalty = 0.25 * humidity / (humidity + 1.0e-4)
        f_ht = torch.clamp(mat.base_ht_fraction + h2_boost - humidity_penalty, 0.05, 0.95)
        k_open = mat.k_open_release_s_1 * multipliers["k_open_release"]
        k_closed = mat.k_closed_release_s_1 * multipliers["k_closed_release"]
        release = k_open * open_pore
        s_ht = f_ht * release
        s_hto = (1.0 - f_ht) * release
        r_grain = dt(grain) - (
            gen
            - mat.k_diff_s_1 * grain
            - mat.k_trap_s_1 * grain
            + mat.k_detrap_s_1 * trap
            - mat.k_closed_in_s_1 * grain
        )
        r_trap = dt(trap) - (mat.k_trap_s_1 * grain - mat.k_detrap_s_1 * trap)
        r_open = dt(open_pore) - (mat.k_diff_s_1 * grain + k_closed * closed - release)
        r_closed = dt(closed) - (mat.k_closed_in_s_1 * grain - k_closed * closed)
        mat_res = [r_grain, r_trap, r_open, r_closed]
        mat_norms = [max(float(s.detach().cpu()) / mean_t_final, source_norm) for s in scale_flat[2:]]
    else:
        grain, surface, water = cols[2], cols[3], cols[4]
        mat = Li4SiO4Params()
        h2_boost = 1.0 + 6.0 * h2_fraction / (h2_fraction + 1.0e-3)
        water_boost = 1.0 + 3.0 * water / (water + 0.2)
        k_ht = mat.k_ex_ht_s_1 * multipliers["k_ex_ht"] * h2_boost
        k_hto = mat.k_ex_hto_s_1 * multipliers["k_ex_hto"] * water_boost
        k_water_desorption = mat.water_desorption_s_1 * multipliers["water_desorption"]
        s_ht = k_ht * surface
        s_hto = k_hto * surface
        r_grain = dt(grain) - (gen - mat.k_diff_s_1 * grain)
        r_surface = dt(surface) - (mat.k_diff_s_1 * grain - s_ht - s_hto)
        r_water = dt(water) - (
            mat.water_adsorption_s_1 * humidity
            - k_water_desorption * water
            - mat.water_exchange_s_1 * water
        )
        mat_res = [r_grain, r_surface, r_water]
        mat_norms = [max(float(s.detach().cpu()) / mean_t_final, source_norm) for s in scale_flat[2:]]

    r_ht = eps * dt(c_ht) + u * dz(c_ht) - eps * d_ax * dzz(c_ht) - (1.0 - eps) * s_ht
    r_hto = eps * dt(c_hto) + u * dz(c_hto) - eps * d_ax * dzz(c_hto) - (1.0 - eps) * s_hto
    gas_loss = torch.mean((r_ht / gas_norm_ht) ** 2) + torch.mean((r_hto / gas_norm_hto) ** 2)
    mat_loss = sum(torch.mean((res / norm) ** 2) for res, norm in zip(mat_res, mat_norms))
    return gas_loss, mat_loss


def load_forward_if_requested(model: nn.Module, init_run: str | None, device: torch.device):
    if not init_run:
        return None
    run_dir = Path(init_run)
    state_path = run_dir / "model.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def apply_detector_response(y: torch.Tensor, args) -> torch.Tensor:
    """Map true HT/HTO outlet values into detector-observation space."""
    ht = y[:, 0:1]
    hto = y[:, 1:2]
    measured_ht = args.detector_gain_ht * ((1.0 - args.detector_ht_loss) * ht + args.detector_hto_to_ht * hto)
    measured_hto = args.detector_gain_hto * (args.detector_ht_to_hto * ht + (1.0 - args.detector_hto_loss) * hto)
    return torch.cat([measured_ht, measured_hto], dim=1)


class TrainableDetectorResponse(nn.Module):
    """Small calibrated detector-response block for semi-unknown HT/HTO measurements."""

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.gain_ht = nn.Parameter(torch.tensor(float(args.detector_init_gain_ht), dtype=torch.float32, device=device))
        self.gain_hto = nn.Parameter(torch.tensor(float(args.detector_init_gain_hto), dtype=torch.float32, device=device))
        self.ht_to_hto = nn.Parameter(torch.tensor(float(args.detector_init_ht_to_hto), dtype=torch.float32, device=device))
        self.hto_to_ht = nn.Parameter(torch.tensor(float(args.detector_init_hto_to_ht), dtype=torch.float32, device=device))
        self.ht_loss = nn.Parameter(torch.tensor(float(args.detector_init_ht_loss), dtype=torch.float32, device=device))
        self.hto_loss = nn.Parameter(torch.tensor(float(args.detector_init_hto_loss), dtype=torch.float32, device=device))
        self.register_buffer(
            "prior",
            torch.tensor(
                [
                    args.detector_prior_gain_ht,
                    args.detector_prior_gain_hto,
                    args.detector_prior_ht_to_hto,
                    args.detector_prior_hto_to_ht,
                    args.detector_prior_ht_loss,
                    args.detector_prior_hto_loss,
                ],
                dtype=torch.float32,
                device=device,
            ),
        )

    def constrained(self) -> dict[str, torch.Tensor]:
        return {
            "gain_ht": torch.clamp(self.gain_ht, 0.7, 1.3),
            "gain_hto": torch.clamp(self.gain_hto, 0.7, 1.3),
            "ht_to_hto": torch.clamp(self.ht_to_hto, 0.0, 0.25),
            "hto_to_ht": torch.clamp(self.hto_to_ht, 0.0, 0.25),
            "ht_loss": torch.clamp(self.ht_loss, 0.0, 0.25),
            "hto_loss": torch.clamp(self.hto_loss, 0.0, 0.25),
        }

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        vals = self.constrained()
        ht = y[:, 0:1]
        hto = y[:, 1:2]
        measured_ht = vals["gain_ht"] * ((1.0 - vals["ht_loss"]) * ht + vals["hto_to_ht"] * hto)
        measured_hto = vals["gain_hto"] * (vals["ht_to_hto"] * ht + (1.0 - vals["hto_loss"]) * hto)
        return torch.cat([measured_ht, measured_hto], dim=1)

    def prior_loss(self) -> torch.Tensor:
        vals = self.constrained()
        current = torch.stack(
            [
                vals["gain_ht"],
                vals["gain_hto"],
                vals["ht_to_hto"],
                vals["hto_to_ht"],
                vals["ht_loss"],
                vals["hto_loss"],
            ]
        )
        scale = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.03, 0.03], dtype=current.dtype, device=current.device)
        return torch.mean(((current - self.prior) / scale) ** 2)

    def as_float(self) -> dict[str, float]:
        return {key: float(value.detach().cpu()) for key, value in self.constrained().items()}


def add_outlet_noise(y_out: torch.Tensor, noise_percent: float, seed: int, rho: float = 0.0) -> dict:
    """Add reproducible relative Gaussian or AR(1)-correlated noise to outlet observations."""
    if noise_percent <= 0.0:
        return {"enabled": False, "noise_percent": 0.0, "seed": seed, "rho": rho}
    generator = torch.Generator(device=y_out.device)
    generator.manual_seed(seed)
    scales = torch.clamp(torch.max(torch.abs(y_out), dim=0).values, min=1.0e-12)
    sigma = noise_percent / 100.0 * scales
    raw = torch.randn(y_out.shape, generator=generator, device=y_out.device, dtype=y_out.dtype)
    rho = float(max(0.0, min(0.999, rho)))
    if rho > 0.0 and y_out.shape[0] > 1:
        raw_corr = torch.empty_like(raw)
        raw_corr[0] = raw[0]
        coeff = float((1.0 - rho**2) ** 0.5)
        for i in range(1, y_out.shape[0]):
            raw_corr[i] = rho * raw_corr[i - 1] + coeff * raw[i]
        raw = raw_corr
    noise = raw * sigma.reshape(1, -1)
    y_out.add_(noise)
    return {
        "enabled": True,
        "noise_percent": noise_percent,
        "seed": seed,
        "rho": rho,
        "sigma_by_output": [float(v) for v in sigma.detach().cpu().numpy()],
    }


def outlet_channel_loss(pred: torch.Tensor, target: torch.Tensor, channel_weights: torch.Tensor) -> torch.Tensor:
    weights = channel_weights.reshape(1, -1)
    active = torch.clamp(torch.sum(weights > 0.0), min=1)
    return torch.sum(((pred - target) * weights) ** 2) / (pred.shape[0] * active)


def li4_initial_water_loss(model: nn.Module, x_ic: torch.Tensor, scale: torch.Tensor, multipliers) -> torch.Tensor:
    pred_water = model(x_ic)[:, 4:5]
    condition_initial_water = x_ic[:, 6:7] * COND_SCALES["initial_surface_water"]
    target = condition_initial_water * multipliers["initial_surface_water"] / scale.reshape(-1)[4]
    return torch.mean((pred_water - target) ** 2)


def plot_trace(out_dir: Path, rows: list[list[float]], names: list[str], material: str) -> None:
    arr = np.asarray(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    for idx, label in enumerate(["total", "outlet", "ic", "initial_water", "inlet", "gas", "material"], start=1):
        axes[0].semilogy(arr[:, 0], arr[:, idx], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{material} multi-condition inverse")
    axes[0].legend(frameon=False, fontsize=7)
    for i, name in enumerate(names):
        axes[1].plot(arr[:, 0], arr[:, 9 + i], label=name)
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--", label="truth")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("multiplier")
    axes[1].set_title("release-rate multipliers")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, color="#DDDDDD", linewidth=0.6)
    fig.savefig(out_dir / "joint_inverse_trace.png", dpi=300)
    fig.savefig(out_dir / "joint_inverse_trace.pdf")
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = load_dataset(args.material, Path(args.reference_root))
    tensors = {
        key: torch.tensor(dataset[key], dtype=torch.float32, device=device)
        for key in ["x_ic", "y_ic", "x_out", "y_out", "x_phys"]
    }
    tensors["y_out"] = apply_detector_response(tensors["y_out"], args)
    outlet_noise_meta = add_outlet_noise(tensors["y_out"], args.outlet_noise_percent, args.noise_seed, args.detector_noise_rho)
    outlet_channel_weights = torch.tensor([args.w_outlet_ht, args.w_outlet_hto], dtype=torch.float32, device=device)
    detector_response_meta = {
        "gain_ht": args.detector_gain_ht,
        "gain_hto": args.detector_gain_hto,
        "ht_to_hto": args.detector_ht_to_hto,
        "hto_to_ht": args.detector_hto_to_ht,
        "ht_loss": args.detector_ht_loss,
        "hto_loss": args.detector_hto_loss,
        "noise_rho": args.detector_noise_rho,
        "space": "detector_observation",
    }
    outlet_observation_meta = {
        "channels": ["HT", "HTO"],
        "weights": [args.w_outlet_ht, args.w_outlet_hto],
        "active_channels": [
            name
            for name, weight in zip(["HT", "HTO"], [args.w_outlet_ht, args.w_outlet_hto])
            if weight > 0.0
        ],
    }
    y_in = torch.zeros((1, 2), dtype=torch.float32, device=device)
    scale_t = torch.tensor(dataset["scale"], dtype=torch.float32, device=device).reshape(1, -1)
    z_max = max(float(z.max()) for z, _, _, _ in dataset["raw"])
    input_dim = int(dataset["x_ic"].shape[1])
    model = NeuralNetwork(input_dim, [args.hidden_size] * args.hidden_layers, len(dataset["names"]), args.activation).to(device)
    init_meta = load_forward_if_requested(model, args.init_forward_run, device)
    multipliers = MultiplierSet(args.material, args.initial_multiplier, args.low, args.high).to(device)
    detector_model = TrainableDetectorResponse(args, device) if args.train_detector_response else None
    opt_groups = [
        {"params": model.parameters(), "lr": args.model_lr},
        {"params": multipliers.parameters(), "lr": args.param_lr},
    ]
    if detector_model is not None:
        opt_groups.append({"params": detector_model.parameters(), "lr": args.detector_lr})
    opt = torch.optim.Adam(
        opt_groups
    )
    mse = nn.MSELoss()
    rng = np.random.default_rng(args.seed + 29)
    weights = {"outlet": args.w_outlet, "ic": args.w_ic, "inlet": args.w_inlet, "gas": args.w_gas, "material": args.w_material}
    log_rows = []
    weight_rows = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        freeze_model = epoch <= args.freeze_model_epochs
        for param in model.parameters():
            param.requires_grad_(not freeze_model)
        opt.zero_grad()
        xout, yout = sample_rows(rng, tensors["x_out"], tensors["y_out"], args.out_batch_size)
        xic, yic = sample_rows(rng, tensors["x_ic"], tensors["y_ic"], args.ic_batch_size)
        xin = sample_x(rng, tensors["x_out"], args.inlet_batch_size).clone()
        xin[:, 0] = 0.0
        yin = y_in.repeat(xin.shape[0], 1)
        xphys = sample_x(rng, tensors["x_phys"], args.phys_batch_size)
        true_pred_out = model(xout)[:, :2]
        pred_out = detector_model(true_pred_out) if detector_model is not None else apply_detector_response(true_pred_out, args)
        l_out = outlet_channel_loss(pred_out, yout, outlet_channel_weights)
        l_detector_prior = (
            detector_model.prior_loss() if detector_model is not None else torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        l_ic = mse(model(xic), yic)
        if args.material == "li4sio4":
            l_initial_water = li4_initial_water_loss(model, xic, scale_t, multipliers.values())
        else:
            l_initial_water = torch.tensor(0.0, dtype=torch.float32, device=device)
        l_in = mse(model(xin)[:, :2], yin)
        l_gas, l_material = multiplier_physics_loss(args.material, model, xphys, scale_t, z_max, multipliers.values())
        l_ic_total = l_ic + args.w_initial_water_ic * l_initial_water
        loss_terms = {"outlet": l_out, "ic": l_ic_total, "inlet": l_in, "gas": l_gas, "material": l_material}
        if args.adaptive_weights and not freeze_model and (epoch == 1 or epoch % args.adapt_every == 0):
            weights, grad_norms = update_adaptive_weights(loss_terms, weights, model, args.adapt_alpha)
        else:
            grad_norms = {name: 0.0 for name in weights}
        total = weights["outlet"] * l_out + weights["ic"] * l_ic_total + weights["inlet"] * l_in + args.w_phys * (
            weights["gas"] * l_gas + weights["material"] * l_material
        ) + args.detector_prior_weight * l_detector_prior
        total.backward()
        opt.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            vals = multipliers.as_float()
            detector_vals = detector_model.as_float() if detector_model is not None else {}
            log_rows.append(
                [
                    epoch,
                    float(total.detach().cpu()),
                    float(l_out.detach().cpu()),
                    float(l_ic.detach().cpu()),
                    float(l_initial_water.detach().cpu()),
                    float(l_in.detach().cpu()),
                    float(l_gas.detach().cpu()),
                    float(l_material.detach().cpu()),
                    float(l_detector_prior.detach().cpu()),
                    *[vals[name] for name in multipliers.names],
                    *[detector_vals[name] for name in detector_vals],
                ]
            )
            weight_rows.append(
                [
                    epoch,
                    weights["outlet"],
                    weights["ic"],
                    weights["inlet"],
                    weights["gas"],
                    weights["material"],
                    grad_norms["outlet"],
                    grad_norms["ic"],
                    grad_norms["inlet"],
                    grad_norms["gas"],
                    grad_norms["material"],
                ]
            )

    out_dir = Path(args.out) if args.out else RESULTS / f"multi_condition_joint_inverse_{args.material}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    detector_names = list(detector_model.as_float()) if detector_model is not None else []
    header = "epoch,total,outlet,ic,initial_water,inlet,gas,material,detector_prior," + ",".join(
        [*multipliers.names, *detector_names]
    )
    np.savetxt(out_dir / "joint_inverse_log.csv", np.asarray(log_rows), delimiter=",", header=header, comments="")
    np.savetxt(
        out_dir / "weights.csv",
        np.asarray(weight_rows),
        delimiter=",",
        header="epoch,w_outlet,w_ic,w_inlet,w_gas,w_material,gn_outlet,gn_ic,gn_inlet,gn_gas,gn_material",
        comments="",
    )
    rel_rmse = plot_prediction(out_dir, dataset, model, device)
    plot_trace(out_dir, log_rows, multipliers.names, args.material)
    meta = {
        "material": args.material,
        "reference_root": args.reference_root,
        "condition_names": [cond["name"] for cond in dataset["manifest"]["conditions"]],
        "input_dim": input_dim,
        "init_forward_run": args.init_forward_run,
        "init_forward_meta": init_meta,
        "true_multiplier": 1.0,
        "estimated_multipliers": multipliers.as_float(),
        "estimated_detector_response": detector_model.as_float() if detector_model is not None else None,
        "outlet_noise": outlet_noise_meta,
        "detector_response": detector_response_meta,
        "outlet_observation": outlet_observation_meta,
        "relative_rmse": rel_rmse,
        "final_weights": weights,
        "args": vars(args),
        "elapsed_seconds": time.time() - start,
        "final_log_row": log_rows[-1],
        "note": "Multi-condition joint inverse PINN trained from outlet HT/HTO, IC, inlet BC, and physics residuals.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=["li2tio3", "li4sio4"], required=True)
    parser.add_argument("--reference-root", default=str(RESULTS / "multi_condition_references"))
    parser.add_argument("--init-forward-run", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--activation", choices=["tanh", "sigmoid"], default="tanh")
    parser.add_argument("--model-lr", type=float, default=2.0e-4)
    parser.add_argument("--param-lr", type=float, default=5.0e-2)
    parser.add_argument("--detector-lr", type=float, default=1.0e-2)
    parser.add_argument("--freeze-model-epochs", type=int, default=300)
    parser.add_argument("--out-batch-size", type=int, default=512)
    parser.add_argument("--ic-batch-size", type=int, default=512)
    parser.add_argument("--inlet-batch-size", type=int, default=512)
    parser.add_argument("--phys-batch-size", type=int, default=1024)
    parser.add_argument("--initial-multiplier", type=float, default=0.6)
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=3.0)
    parser.add_argument("--w-outlet", type=float, default=1.0)
    parser.add_argument("--w-outlet-ht", type=float, default=1.0)
    parser.add_argument("--w-outlet-hto", type=float, default=1.0)
    parser.add_argument("--w-ic", type=float, default=0.3)
    parser.add_argument("--w-initial-water-ic", type=float, default=1.0)
    parser.add_argument("--w-inlet", type=float, default=0.3)
    parser.add_argument("--w-phys", type=float, default=0.01)
    parser.add_argument("--w-gas", type=float, default=1.0)
    parser.add_argument("--w-material", type=float, default=1.0)
    parser.add_argument("--adaptive-weights", action="store_true")
    parser.add_argument("--adapt-every", type=int, default=50)
    parser.add_argument("--adapt-alpha", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=6064)
    parser.add_argument("--outlet-noise-percent", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=20260610)
    parser.add_argument("--detector-gain-ht", type=float, default=1.0)
    parser.add_argument("--detector-gain-hto", type=float, default=1.0)
    parser.add_argument("--detector-ht-to-hto", type=float, default=0.0)
    parser.add_argument("--detector-hto-to-ht", type=float, default=0.0)
    parser.add_argument("--detector-ht-loss", type=float, default=0.0)
    parser.add_argument("--detector-hto-loss", type=float, default=0.0)
    parser.add_argument("--detector-noise-rho", type=float, default=0.0)
    parser.add_argument("--train-detector-response", action="store_true")
    parser.add_argument("--detector-prior-weight", type=float, default=1.0e-4)
    parser.add_argument("--detector-init-gain-ht", type=float, default=1.0)
    parser.add_argument("--detector-init-gain-hto", type=float, default=1.0)
    parser.add_argument("--detector-init-ht-to-hto", type=float, default=0.0)
    parser.add_argument("--detector-init-hto-to-ht", type=float, default=0.0)
    parser.add_argument("--detector-init-ht-loss", type=float, default=0.0)
    parser.add_argument("--detector-init-hto-loss", type=float, default=0.0)
    parser.add_argument("--detector-prior-gain-ht", type=float, default=1.0)
    parser.add_argument("--detector-prior-gain-hto", type=float, default=1.0)
    parser.add_argument("--detector-prior-ht-to-hto", type=float, default=0.0)
    parser.add_argument("--detector-prior-hto-to-ht", type=float, default=0.0)
    parser.add_argument("--detector-prior-ht-loss", type=float, default=0.0)
    parser.add_argument("--detector-prior-hto-loss", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
