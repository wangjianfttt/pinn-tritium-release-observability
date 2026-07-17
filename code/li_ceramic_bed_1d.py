#!/usr/bin/env python3
"""1D Li2TiO3 / Li4SiO4 ceramic breeder pebble-bed benchmark.

This finite-volume/material-compartment model is the reference target for the
later PINN implementation.  It is deliberately reduced, but it preserves the
main material-specific mechanisms identified in the literature review:

- Li2TiO3: grain, trap, open-pore, and closed-pore inventories.
- Li4SiO4: grain, surface, and surface-water inventories.
- Shared gas transport for HT and HTO in purge gas.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


@dataclass(frozen=True)
class SharedParams:
    length_m: float = 0.30
    porosity: float = 0.36
    superficial_velocity_m_s: float = 2.0e-3
    axial_dispersion_m2_s: float = 1.0e-5
    source_rate_mol_m3_s: float = 1.0e-7
    temperature_k: float = 773.15
    h2_fraction: float = 1.0e-3
    humidity: float = 1.0e-4


@dataclass(frozen=True)
class Li2TiO3Params:
    k_diff_s_1: float = 3.0e-4
    k_trap_s_1: float = 1.2e-4
    k_detrap_s_1: float = 2.0e-5
    k_closed_in_s_1: float = 5.0e-5
    k_closed_release_s_1: float = 1.0e-5
    k_open_release_s_1: float = 8.0e-4
    base_ht_fraction: float = 0.55


@dataclass(frozen=True)
class Li4SiO4Params:
    k_diff_s_1: float = 2.0e-4
    k_ex_ht_s_1: float = 3.0e-4
    k_ex_hto_s_1: float = 7.0e-4
    water_adsorption_s_1: float = 2.0e-5
    water_desorption_s_1: float = 4.0e-6
    water_exchange_s_1: float = 1.0e-5
    initial_surface_water: float = 1.0


def ht_fraction_li2tio3(params: SharedParams, mat: Li2TiO3Params) -> float:
    h2_boost = 0.25 * params.h2_fraction / (params.h2_fraction + 1.0e-3)
    humidity_penalty = 0.25 * params.humidity / (params.humidity + 1.0e-4)
    return float(np.clip(mat.base_ht_fraction + h2_boost - humidity_penalty, 0.05, 0.95))


def li4_exchange_rates(params: SharedParams, mat: Li4SiO4Params, water_surface: np.ndarray):
    h2_boost = 1.0 + 6.0 * params.h2_fraction / (params.h2_fraction + 1.0e-3)
    water_boost = 1.0 + 3.0 * water_surface / (water_surface + 0.2)
    k_ht = mat.k_ex_ht_s_1 * h2_boost
    k_hto = mat.k_ex_hto_s_1 * water_boost
    return k_ht, k_hto


def rhs_factory(material: str, shared: SharedParams, mat, n: int):
    dz = shared.length_m / n
    eps = shared.porosity
    u = shared.superficial_velocity_m_s
    d_ax = shared.axial_dispersion_m2_s
    gen = shared.source_rate_mol_m3_s

    def gas_transport(c: np.ndarray, source: np.ndarray) -> np.ndarray:
        c_left = np.empty_like(c)
        c_left[0] = 0.0
        c_left[1:] = c[:-1]
        c_right = np.empty_like(c)
        c_right[:-1] = c[1:]
        c_right[-1] = c[-1]
        adv = -(u / eps) * (c - c_left) / dz
        diff = d_ax * (c_right - 2.0 * c + c_left) / dz**2
        return adv + diff + ((1.0 - eps) / eps) * source

    if material == "li2tio3":
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            c_ht, c_hto, c_grain, c_trap, c_open, c_closed = np.split(y, 6)
            f_ht = ht_fraction_li2tio3(shared, mat)
            release = mat.k_open_release_s_1 * c_open
            s_ht = f_ht * release
            s_hto = (1.0 - f_ht) * release

            dc_grain = (
                gen
                - mat.k_diff_s_1 * c_grain
                - mat.k_trap_s_1 * c_grain
                + mat.k_detrap_s_1 * c_trap
                - mat.k_closed_in_s_1 * c_grain
            )
            dc_trap = mat.k_trap_s_1 * c_grain - mat.k_detrap_s_1 * c_trap
            dc_open = mat.k_diff_s_1 * c_grain + mat.k_closed_release_s_1 * c_closed - release
            dc_closed = mat.k_closed_in_s_1 * c_grain - mat.k_closed_release_s_1 * c_closed
            return np.concatenate(
                [
                    gas_transport(c_ht, s_ht),
                    gas_transport(c_hto, s_hto),
                    dc_grain,
                    dc_trap,
                    dc_open,
                    dc_closed,
                ]
            )

        return rhs

    if material == "li4sio4":
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            c_ht, c_hto, c_grain, c_surface, water_surface = np.split(y, 5)
            k_ht, k_hto = li4_exchange_rates(shared, mat, water_surface)
            s_ht = k_ht * c_surface
            s_hto = k_hto * c_surface
            dc_grain = gen - mat.k_diff_s_1 * c_grain
            dc_surface = mat.k_diff_s_1 * c_grain - s_ht - s_hto
            dwater = (
                mat.water_adsorption_s_1 * shared.humidity
                - mat.water_desorption_s_1 * water_surface
                - mat.water_exchange_s_1 * water_surface
            )
            return np.concatenate(
                [
                    gas_transport(c_ht, s_ht),
                    gas_transport(c_hto, s_hto),
                    dc_grain,
                    dc_surface,
                    dwater,
                ]
            )

        return rhs

    raise ValueError(f"unknown material: {material}")


def initial_state(material: str, mat, n: int) -> np.ndarray:
    if material == "li2tio3":
        return np.zeros(6 * n)
    if material == "li4sio4":
        y0 = np.zeros(5 * n)
        y0[4 * n : 5 * n] = mat.initial_surface_water
        return y0
    raise ValueError(f"unknown material: {material}")


def simulate(material: str, shared: SharedParams, n: int, t_final: float, n_times: int, mat=None):
    if mat is None:
        mat = Li2TiO3Params() if material == "li2tio3" else Li4SiO4Params()
    z = np.linspace(0.5 * shared.length_m / n, shared.length_m - 0.5 * shared.length_m / n, n)
    t_eval = np.linspace(0.0, t_final, n_times)
    sol = solve_ivp(
        rhs_factory(material, shared, mat, n),
        (0.0, t_final),
        initial_state(material, mat, n),
        t_eval=t_eval,
        method="BDF",
        rtol=1.0e-7,
        atol=1.0e-11,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return mat, z, sol.t, sol.y.T


def split_state(material: str, y: np.ndarray):
    if material == "li2tio3":
        return dict(zip(["c_ht", "c_hto", "grain", "trap", "open", "closed"], np.split(y, 6, axis=1)))
    return dict(zip(["c_ht", "c_hto", "grain", "surface", "water"], np.split(y, 5, axis=1)))


def integrate_field(field: np.ndarray, dz: float, solid_factor: float = 1.0) -> np.ndarray:
    return solid_factor * np.sum(field, axis=1) * dz


def diagnostics(material: str, shared: SharedParams, z: np.ndarray, t: np.ndarray, y: np.ndarray):
    fields = split_state(material, y)
    dz = shared.length_m / len(z)
    eps = shared.porosity
    solid_factor = 1.0 - eps
    c_ht = fields["c_ht"]
    c_hto = fields["c_hto"]
    ht_out = c_ht[:, -1]
    hto_out = c_hto[:, -1]
    ht_flux = shared.superficial_velocity_m_s * ht_out
    hto_flux = shared.superficial_velocity_m_s * hto_out
    ht_extracted = np.concatenate([[0.0], np.cumsum(0.5 * (ht_flux[1:] + ht_flux[:-1]) * np.diff(t))])
    hto_extracted = np.concatenate([[0.0], np.cumsum(0.5 * (hto_flux[1:] + hto_flux[:-1]) * np.diff(t))])
    generated = solid_factor * shared.source_rate_mol_m3_s * shared.length_m * t
    gas_inventory = eps * integrate_field(c_ht + c_hto, dz)

    solid_names = [name for name in fields if name not in ("c_ht", "c_hto", "water")]
    compartment_inventory = {
        name: integrate_field(fields[name], dz, solid_factor=solid_factor) for name in solid_names
    }
    solid_inventory = sum(compartment_inventory.values())
    total_inventory = solid_inventory + gas_inventory
    extracted = ht_extracted + hto_extracted
    balance_error = generated - extracted - total_inventory
    ht_fraction_out = np.divide(ht_out, ht_out + hto_out, out=np.zeros_like(ht_out), where=(ht_out + hto_out) > 0)
    return {
        "fields": fields,
        "ht_out": ht_out,
        "hto_out": hto_out,
        "ht_extracted": ht_extracted,
        "hto_extracted": hto_extracted,
        "total_extracted": extracted,
        "generated": generated,
        "gas_inventory": gas_inventory,
        "compartment_inventory": compartment_inventory,
        "solid_inventory": solid_inventory,
        "total_inventory": total_inventory,
        "mass_balance_error": balance_error,
        "ht_fraction_out": ht_fraction_out,
    }


def write_outputs(out_dir: Path, material: str, shared: SharedParams, mat, z, t, y):
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = diagnostics(material, shared, z, t, y)
    fields = diag["fields"]
    np.savetxt(out_dir / "z.csv", z, delimiter=",", header="z_m", comments="")
    np.savetxt(out_dir / "time.csv", t, delimiter=",", header="time_s", comments="")
    for name, values in fields.items():
        np.savetxt(out_dir / f"{name}.csv", values, delimiter=",")

    table = np.column_stack(
        [
            t,
            diag["ht_out"],
            diag["hto_out"],
            diag["ht_fraction_out"],
            diag["generated"],
            diag["ht_extracted"],
            diag["hto_extracted"],
            diag["total_extracted"],
            diag["solid_inventory"],
            diag["gas_inventory"],
            diag["mass_balance_error"],
        ]
    )
    np.savetxt(
        out_dir / "diagnostics.csv",
        table,
        delimiter=",",
        header=(
            "time_s,ht_out_mol_m3,hto_out_mol_m3,ht_fraction_out,"
            "generated_mol_m2,ht_extracted_mol_m2,hto_extracted_mol_m2,total_extracted_mol_m2,"
            "solid_inventory_mol_m2,gas_inventory_mol_m2,mass_balance_error_mol_m2"
        ),
        comments="",
    )
    final_generated = float(diag["generated"][-1])
    final_error = float(diag["mass_balance_error"][-1])
    metrics = {
        "material": material,
        "shared_params": asdict(shared),
        "material_params": asdict(mat),
        "grid_cells": int(len(z)),
        "time_samples": int(len(t)),
        "final_time_s": float(t[-1]),
        "final_ht_out_mol_m3": float(diag["ht_out"][-1]),
        "final_hto_out_mol_m3": float(diag["hto_out"][-1]),
        "final_ht_fraction_out": float(diag["ht_fraction_out"][-1]),
        "final_generated_mol_m2": final_generated,
        "final_total_extracted_mol_m2": float(diag["total_extracted"][-1]),
        "final_solid_inventory_mol_m2": float(diag["solid_inventory"][-1]),
        "final_gas_inventory_mol_m2": float(diag["gas_inventory"][-1]),
        "final_compartment_inventory_mol_m2": {
            key: float(value[-1]) for key, value in diag["compartment_inventory"].items()
        },
        "relative_mass_balance_error": abs(final_error) / final_generated if final_generated > 0 else 0.0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return diag, metrics


def plot_outputs(fig_path: Path, material: str, t: np.ndarray, diag):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    th = t / 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.9), constrained_layout=True)
    axes[0, 0].plot(th, diag["ht_out"], label="HT", color="#3366AA")
    axes[0, 0].plot(th, diag["hto_out"], label="HTO", color="#CC6677")
    axes[0, 0].set_xlabel("time (h)")
    axes[0, 0].set_ylabel("outlet concentration (mol/m3)")
    axes[0, 0].set_title(f"{material} outlet species")
    axes[0, 0].legend(frameon=False)

    for name, values in diag["compartment_inventory"].items():
        axes[0, 1].plot(th, values, label=name)
    axes[0, 1].set_xlabel("time (h)")
    axes[0, 1].set_ylabel("inventory (mol/m2)")
    axes[0, 1].set_title("solid compartments")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(th, diag["generated"], label="generated", color="#999933")
    axes[1, 0].plot(th, diag["total_extracted"], label="extracted", color="#AA4499")
    axes[1, 0].plot(th, diag["solid_inventory"], label="solid", color="#44AA99")
    axes[1, 0].set_xlabel("time (h)")
    axes[1, 0].set_ylabel("amount (mol/m2)")
    axes[1, 0].set_title("balance components")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(th, diag["ht_fraction_out"], color="#228833")
    axes[1, 1].set_xlabel("time (h)")
    axes[1, 1].set_ylabel("HT / (HT + HTO)")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("outlet molecular fraction")
    for ax in axes.ravel():
        ax.grid(True, color="#DDDDDD", linewidth=0.6)
    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=["li2tio3", "li4sio4"], required=True)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--t-final", type=float, default=8.0 * 3600.0)
    parser.add_argument("--n-times", type=int, default=401)
    parser.add_argument("--temperature", type=float, default=SharedParams.temperature_k)
    parser.add_argument("--h2-fraction", type=float, default=SharedParams.h2_fraction)
    parser.add_argument("--humidity", type=float, default=SharedParams.humidity)
    parser.add_argument("--velocity", type=float, default=SharedParams.superficial_velocity_m_s)
    parser.add_argument("--porosity", type=float, default=SharedParams.porosity)
    parser.add_argument("--source-rate", type=float, default=SharedParams.source_rate_mol_m3_s)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    shared = SharedParams(
        porosity=args.porosity,
        superficial_velocity_m_s=args.velocity,
        source_rate_mol_m3_s=args.source_rate,
        temperature_k=args.temperature,
        h2_fraction=args.h2_fraction,
        humidity=args.humidity,
    )
    mat, z, t, y = simulate(args.material, shared, args.n, args.t_final, args.n_times)
    out_dir = Path(args.out) if args.out else RESULTS / f"li_ceramic_{args.material}_baseline"
    diag, metrics = write_outputs(out_dir, args.material, shared, mat, z, t, y)
    plot_outputs(FIGURES / f"{out_dir.name}.png", args.material, t, diag)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
