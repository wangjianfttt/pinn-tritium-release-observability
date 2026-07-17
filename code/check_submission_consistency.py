#!/usr/bin/env python3
"""Check the compact Nuclear Fusion submission line.

This check intentionally follows the current paper, not the older project-wide
working notes. It verifies that the submission source stays focused on the
scientific argument: Li2TiO3 delayed-release identifiability, 3D packed-bed
transport, PINN source estimation, HT/HTO chemistry, detector response and
thermal identity.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
NOTES = ROOT / "notes"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_spaces(text: str) -> str:
    return " ".join(text.split())


def expand_tex_inputs(tex_text: str) -> str:
    """Inline simple manuscript-level \\input files for text-presence checks."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        path = MANUSCRIPT / name
        if not path.exists():
            return match.group(0)
        return read(path)

    return re.sub(r"\\input\{([^}]+)\}", replace, tex_text)


def add(rows: list[dict[str, str]], check: str, ok: bool, detail: str) -> None:
    rows.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})


def exists_all(paths: list[Path]) -> tuple[bool, str]:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    return not missing, "all present" if not missing else "; ".join(missing)


def close(value: float, target: float, tol: float = 0.01) -> bool:
    return abs(float(value) - target) <= tol


def main() -> int:
    rows: list[dict[str, str]] = []

    main_tex = MANUSCRIPT / "manuscript_iop_submission_closure_en.tex"
    si_tex = MANUSCRIPT / "supplementary_information_submission_en.tex"
    spine_cn = MANUSCRIPT / "MANUSCRIPT_SCIENTIFIC_SPINE_CN.md"
    readme = ROOT / "README.md"
    reproduce_script = ROOT / "scripts" / "reproduce_submission_science.sh"

    current_sources = [main_tex, si_tex, spine_cn, readme]
    main_text = read(main_tex)
    si_text = read(si_tex)
    main_text_expanded = expand_tex_inputs(main_text)
    si_text_expanded = expand_tex_inputs(si_text)
    text = "\n".join(read(path) for path in current_sources)
    readme_text = read(readme)
    reproduce_text = read(reproduce_script) if reproduce_script.exists() else ""
    pdf_build_text = read(ROOT / "scripts" / "build_submission_pdf_local.sh")

    current_title = (
        "Measurement-constrained PINN source recovery for delayed tritium release "
        "in solid-breeder pebble beds"
    )
    current_title_compact = compact_spaces(current_title)
    add(
        rows,
        "current_title_synchronized",
        all(
            current_title_compact in compact_spaces(source)
            for source in [main_text, si_text, read(spine_cn), readme_text]
        ),
        "title synchronized across main manuscript, Supplement, Chinese reader and README",
    )

    si_paths = re.findall(r"\\path\{([^}]+)\}", si_text + "\n" + si_text_expanded)
    missing_si_paths = [path for path in si_paths if not (ROOT / path).exists()]
    add(
        rows,
        "supplementary_reproducibility_paths_exist",
        not missing_si_paths,
        "all Supplement path entries exist"
        if not missing_si_paths
        else "; ".join(missing_si_paths),
    )

    pdfs = [
        MANUSCRIPT / "manuscript_iop_submission_closure_en.pdf",
        MANUSCRIPT / "supplementary_information_submission_en.pdf",
        MANUSCRIPT / "manuscript_iop_submission_closure_with_appendix_en.pdf",
    ]
    ok, detail = exists_all(pdfs)
    add(rows, "current_pdf_artifacts", ok, detail)

    render_dir = MANUSCRIPT / "_render_check_current_main_figures"
    render_contact = render_dir / "current_main_figure_pages_contact.png"
    render_pages = sorted(render_dir.glob("page-*.png"))
    pdf_mtime = pdfs[0].stat().st_mtime if pdfs[0].exists() else 0.0
    render_ok = (
        render_contact.exists()
        and len(render_pages) == 19
        and render_contact.stat().st_mtime >= pdf_mtime
        and all(page.stat().st_mtime >= pdf_mtime for page in render_pages)
    )
    add(
        rows,
        "final_pdf_figure_pages_rendered_current",
        render_ok,
        "final PDF pages 3--21 were rendered after the current manuscript PDF"
        if render_ok
        else (
            f"contact_exists={render_contact.exists()}; n_pages={len(render_pages)}; "
            f"render_after_pdf={render_contact.exists() and render_contact.stat().st_mtime >= pdf_mtime}"
        ),
    )

    figures = [
        FIGURES / "methodology_overview_figure.pdf",
        FIGURES / "release_history_main_figure.pdf",
        FIGURES / "packed_bed_transport_tests_current.pdf",
        FIGURES / "pinn_source_head_architecture.pdf",
        FIGURES / "pinn_training_release_evidence.pdf",
        FIGURES / "chemistry_thermal_identity_evidence.pdf",
        FIGURES / "whole_reactor_accountancy_main_figure.pdf",
    ]
    ok, detail = exists_all(figures)
    add(rows, "seven_main_science_figures", ok, detail)

    conceptual_style_specs = [
        (
            "fig1_workflow_style_lock",
            ROOT / "code" / "build_methodology_overview_figure.py",
            RESULTS / "methodology_overview_figure" / "methodology_overview_figure_summary.json",
        ),
        (
            "fig4_network_style_lock",
            ROOT / "code" / "build_pinn_source_head_architecture_figure.py",
            RESULTS / "pinn_source_head_architecture" / "pinn_source_head_architecture_summary.json",
        ),
    ]
    for check_name, script_path, summary_path in conceptual_style_specs:
        if not script_path.exists() or not summary_path.exists():
            add(rows, check_name, False, "missing controlled script or figure summary")
            continue
        script_text = read(script_path)
        summary = json.loads(read(summary_path))
        if "summary" in summary and isinstance(summary["summary"], dict):
            summary = summary["summary"]
        fixed_canvas_ok = 'bbox_inches="tight"' not in script_text and "bbox_inches='tight'" not in script_text
        summary_ok = any("fixed canvas" in str(note).lower() for note in summary.get("qa_notes", []))
        add(
            rows,
            check_name,
            fixed_canvas_ok and summary_ok,
            (
                "controlled script uses fixed canvas and records figure summary"
                if fixed_canvas_ok and summary_ok
                else f"fixed_canvas_ok={fixed_canvas_ok}; summary_ok={summary_ok}"
            ),
        )

    multipanel_style_specs = [
        (
            "fig2_release_history_style_lock",
            ROOT / "code" / "build_release_history_main_figure.py",
            RESULTS / "release_history_main_figure" / "release_history_main_figure_summary.json",
        ),
        (
            "fig3_packed_bed_style_lock",
            ROOT / "code" / "build_packed_bed_transport_tests_current.py",
            RESULTS / "packed_bed_transport_tests_current_summary.json",
        ),
        (
            "fig5_pinn_training_style_lock",
            ROOT / "code" / "build_pinn_training_release_evidence.py",
            RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_summary.json",
        ),
        (
            "fig6_chemistry_thermal_style_lock",
            ROOT / "code" / "build_chemistry_thermal_identity_evidence.py",
            RESULTS / "chemistry_thermal_identity_evidence" / "chemistry_thermal_identity_evidence_summary.json",
        ),
        (
            "fig7_accountancy_style_lock",
            ROOT / "code" / "build_whole_reactor_accountancy_main_figure.py",
            RESULTS
            / "whole_reactor_dynamic_tritium_observability"
            / "whole_reactor_accountancy_main_figure_summary.json",
        ),
    ]
    for check_name, script_path, summary_path in multipanel_style_specs:
        if not script_path.exists() or not summary_path.exists():
            add(rows, check_name, False, "missing script or figure summary")
            continue
        script_text = read(script_path)
        summary = json.loads(read(summary_path))
        if "summary" in summary and isinstance(summary["summary"], dict):
            summary = summary["summary"]
        layout_ok = "3x2" in str(summary.get("figure_layout", ""))
        aspect = float(summary.get("figure_aspect_width_over_height", 0.0))
        aspect_ok = abs(aspect - (7.36 / 9.20)) <= 0.035
        title_ok = summary.get("panel_titles_removed") is True
        source_title_ok = not re.search(r"\.set_title\((?![\"']{2})", script_text)
        no_panel_title_args = not re.search(
            r"(?:def\s+(?:panel|panel_label)\([^)]*title|(?:panel|panel_label)\(ax,\s*['\"][a-z]['\"],\s*['\"])",
            script_text,
        )
        legacy_layout_ok = (
            "plt.subplots(2, 2" not in script_text
            and "plt.subplots(1, 6" not in script_text
            and "plt.subplots(2, 3" not in script_text
            and "figsize=(7.40, 6.25)" not in script_text
        )
        fixed_canvas_ok = 'bbox_inches="tight"' not in script_text and "bbox_inches='tight'" not in script_text
        add(
            rows,
            check_name,
            layout_ok
            and aspect_ok
            and title_ok
            and source_title_ok
            and no_panel_title_args
            and legacy_layout_ok
            and fixed_canvas_ok,
            (
                "3x2 compact layout, no panel titles, fixed canvas save, no legacy layout block, 7.36x9.20 canvas aspect locked"
                if layout_ok
                and aspect_ok
                and title_ok
                and source_title_ok
                and no_panel_title_args
                and legacy_layout_ok
                and fixed_canvas_ok
                else f"layout={summary.get('figure_layout')}; aspect={aspect:.3f}; "
                f"panel_titles_removed={summary.get('panel_titles_removed')}; "
                f"source_title_ok={source_title_ok}; no_panel_title_args={no_panel_title_args}; "
                f"legacy_layout_ok={legacy_layout_ok}; fixed_canvas_ok={fixed_canvas_ok}"
            ),
        )

    style_contract_ok = True
    style_contract_notes: list[str] = []
    expected_aspect = 7.36 / 9.20
    expected_script_labels = {
        "fig2_release_history_style_lock": [
            "open-path coordinate",
            "delayed-path coordinate",
            "Kobayashi and Oya [11]",
            "Zhu et al. [10]",
        ],
        "fig3_transport_style_lock": [
            "mixed outlet, 1D",
            "4 outlets, matched 3D",
            "4 outlets, 3D",
            "quadrant envelope",
            "Ergun shape error",
        ],
        "fig5_pinn_training_style_lock": [
            "pooled",
            "target",
            "calibrated 3D HT/HTO",
            "PDE field diff.",
            "source-term layer",
        ],
        "fig6_chemistry_thermal_style_lock": [
            "wet He",
            "wet H2",
            "dry He",
            "dry H2",
            "pure pair",
            "five mixture",
            "nine mixture",
        ],
        "fig7_accountancy_style_lock": [
            "mixed outlet",
            "central q1",
            "central q4",
            "sector release: 96.0%",
            "system total: 16.7%",
        ],
    }
    for check_name, script_path, summary_path in multipanel_style_specs:
        if not script_path.exists() or not summary_path.exists():
            style_contract_ok = False
            style_contract_notes.append(f"{check_name}: missing source or summary")
            continue
        script_text = read(script_path)
        summary = json.loads(read(summary_path))
        if "summary" in summary and isinstance(summary["summary"], dict):
            summary = summary["summary"]
        layout = str(summary.get("figure_layout", ""))
        aspect = float(summary.get("figure_aspect_width_over_height", 0.0))
        local_ok = (
            "3x2" in layout
            and bool(re.search(r"plt\.subplots\(\s*3\s*,\s*2\b", script_text))
            and abs(aspect - expected_aspect) <= 0.035
            and summary.get("panel_titles_removed") is True
            and "finish_prl_axes" in script_text
            and not re.search(r"\.set_title\((?![\"']{2})", script_text)
            and not re.search(
                r"(?:def\s+(?:panel|panel_label)\([^)]*title|(?:panel|panel_label)\(ax,\s*['\"][a-z]['\"],\s*['\"])",
                script_text,
            )
            and 'bbox_inches="tight"' not in script_text
            and "bbox_inches='tight'" not in script_text
            and "plt.subplots(2, 2" not in script_text
            and "plt.subplots(2, 3" not in script_text
            and "plt.subplots(1, 6" not in script_text
            and "figsize=(7.40, 6.25)" not in script_text
            and all(label in script_text for label in expected_script_labels.get(check_name, []))
        )
        if not local_ok:
            style_contract_ok = False
            tight_bbox_present = 'bbox_inches="tight"' in script_text or "bbox_inches='tight'" in script_text
            uses_3x2_subplots = bool(re.search(r"plt\.subplots\(\s*3\s*,\s*2\b", script_text))
            legacy_layout_present = (
                "plt.subplots(2, 2" in script_text
                or "plt.subplots(2, 3" in script_text
                or "plt.subplots(1, 6" in script_text
                or "figsize=(7.40, 6.25)" in script_text
            )
            missing_labels = [
                label for label in expected_script_labels.get(check_name, []) if label not in script_text
            ]
            panel_title_args_present = bool(
                re.search(
                    r"(?:def\s+(?:panel|panel_label)\([^)]*title|(?:panel|panel_label)\(ax,\s*['\"][a-z]['\"],\s*['\"])",
                    script_text,
                )
            )
            style_contract_notes.append(
                f"{check_name}: layout={layout}; aspect={aspect:.3f}; "
                f"panel_titles_removed={summary.get('panel_titles_removed')}; "
                f"finish_prl_axes={'finish_prl_axes' in script_text}; "
                f"uses_3x2_subplots={uses_3x2_subplots}; "
                f"panel_title_args={panel_title_args_present}; "
                f"legacy_layout_present={legacy_layout_present}; "
                f"missing_labels={missing_labels}; "
                f"tight_bbox_present={tight_bbox_present}"
            )
    add(
        rows,
        "main_multipanel_figure_style_contract",
        style_contract_ok,
        "Fig.2/3/5/6/7 keep 3x2 compact PRL-like framed style with no panel titles and a 7.36x9.20 canvas"
        if style_contract_ok
        else "; ".join(style_contract_notes),
    )

    fig2_summary_path = RESULTS / "release_history_main_figure" / "release_history_main_figure_summary.json"
    fig2_script = ROOT / "code" / "build_release_history_main_figure.py"
    if fig2_summary_path.exists() and fig2_script.exists():
        fig2_summary = json.loads(read(fig2_summary_path))
        tds_labels = fig2_summary.get("tds_legend_labels", [])
        fig2_text = read(fig2_script)
        fig2_ref_ok = (
            "Kobayashi and Oya [11]" in tds_labels
            and "Zhu et al. [10]" in tds_labels
            and 'label="Kobayashi and Oya [11]"' in fig2_text
            and 'label="Zhu et al. [10]"' in fig2_text
        )
        detail = "Fig. 2 TDS legend includes source names and reference numbers" if fig2_ref_ok else str(tds_labels)
    else:
        fig2_ref_ok = False
        detail = "missing Fig. 2 summary or source script"
    add(rows, "fig2_tds_legend_has_reference_numbers", fig2_ref_ok, detail)

    preview_refs = re.findall(r"figures/_[^}]+", main_text + "\n" + si_text)
    forbidden_figure_refs = re.findall(
        r"figures/[^}\s]*(?:preview|thumb|current_preview|admissibility_evidence|scale_consistency_law|accountancy_closure)[^}\s]*",
        main_text + "\n" + si_text,
    )
    add(
        rows,
        "manuscript_uses_current_figures_not_preview_exports",
        not preview_refs and not forbidden_figure_refs,
        "no preview, thumbnail, current-preview or legacy figure paths are included in manuscript or Supplement"
        if not preview_refs and not forbidden_figure_refs
        else "; ".join(preview_refs + forbidden_figure_refs),
    )

    main_figure_paths = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", main_text)
    expected_main_figure_paths = [
        "figures/methodology_overview_figure.pdf",
        "figures/release_history_main_figure.pdf",
        "figures/packed_bed_transport_tests_current.pdf",
        "figures/pinn_source_head_architecture.pdf",
        "figures/pinn_training_release_evidence.pdf",
        "figures/chemistry_thermal_identity_evidence.pdf",
        "figures/whole_reactor_accountancy_main_figure.pdf",
    ]
    add(
        rows,
        "main_manuscript_figure_paths_locked",
        main_figure_paths == expected_main_figure_paths,
        "main text uses only the locked seven current figure files"
        if main_figure_paths == expected_main_figure_paths
        else "found=" + "; ".join(main_figure_paths),
    )

    readme_figure_markers = [
        "The seven main figures now follow the same story line",
        "Fig. 1 shows the method sequence",
        "Fig. 2 shows why delayed release needs release-tail",
        "Fig. 3 shows how 3D packed-bed transport",
        "Fig. 4 shows how the PINN connects measured histories",
        "Fig. 5 judges PINN source estimation",
        "Fig. 6 separates HT/HTO chemistry",
        "Fig. 7 carries the local source estimate",
    ]
    missing = [marker for marker in readme_figure_markers if marker not in readme_text]
    add(
        rows,
        "readme_seven_figure_storyline",
        not missing,
        "README figure storyline aligns with the main text" if not missing else "; ".join(missing),
    )

    body_before_refs = main_text.split(r"\begin{thebibliography}")[0]
    first_cited_keys: list[str] = []
    seen_cites: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", body_before_refs):
        for key in [item.strip() for item in match.group(1).split(",")]:
            if key and key not in seen_cites:
                seen_cites.add(key)
                first_cited_keys.append(key)
    bibitem_keys = re.findall(r"\\bibitem\{([^}]+)\}", main_text)
    expected_ref_keys = [f"ref{idx}" for idx in range(1, len(bibitem_keys) + 1)]
    citation_order_ok = (
        first_cited_keys == expected_ref_keys
        and bibitem_keys == expected_ref_keys
        and set(first_cited_keys) == set(bibitem_keys)
    )
    detail = (
        f"first citations and bibliography are in numeric order ref1--ref{len(bibitem_keys)}"
        if citation_order_ok
        else "first_cites="
        + ",".join(first_cited_keys)
        + "; bibitems="
        + ",".join(bibitem_keys)
    )
    add(rows, "reference_first_citation_order", citation_order_ok, detail)

    parameter_table_paths = [
        ROOT / "code" / "build_reduced_model_parameter_table_si.py",
        RESULTS / "reduced_model_parameter_table.csv",
        RESULTS / "reduced_model_parameter_table_si_summary.json",
        MANUSCRIPT / "generated_reduced_model_parameter_table.tex",
    ]
    ok, detail = exists_all(parameter_table_paths)
    if ok:
        parameter_summary = json.loads(read(parameter_table_paths[2]))
        parameter_table_text = read(parameter_table_paths[3])
        ok = (
            "build_reduced_model_parameter_table_si.py" in reproduce_text
            and r"\input{generated_reduced_model_parameter_table.tex}" in si_text
            and int(parameter_summary["n_parameters"]) == 37
            and int(parameter_summary["n_inverse_target_parameters"]) == 2
            and "Li$_2$TiO$_3$ release model" in parameter_table_text
            and "HT/HTO chemistry exchange" in parameter_table_text
            and "Detector response" in parameter_table_text
            and "Thermal response variables" in parameter_table_text
            and r"\eta_{\rm HT\to HTO}" in parameter_table_text
            and r"\Delta T_{\rm 3D}" in parameter_table_text
            and "inverse target multiplier" in parameter_table_text
            and "protocol-transfer multiplier" in parameter_table_text
        )
        detail = (
            "SI reduced-model parameter table is generated from tracked CSV"
            if ok
            else "expected 37 parameters, 2 inverse targets, SI input and generated table markers"
        )
    add(rows, "reduced_model_parameter_table", ok, detail)

    chemistry_thermal_paths = [
        ROOT / "code" / "build_chemistry_thermal_identity_evidence.py",
        FIGURES / "chemistry_thermal_identity_evidence.pdf",
        RESULTS / "chemistry_thermal_identity_evidence" / "chemistry_thermal_identity_evidence_summary.json",
    ]
    ok, detail = exists_all(chemistry_thermal_paths)
    if ok:
        chemistry_thermal = json.loads(read(chemistry_thermal_paths[2]))
        chemistry_thermal_ok = (
            "build_chemistry_thermal_identity_evidence.py" in reproduce_text
            and "build_chemistry_thermal_identity_evidence.py" in si_text
            and close(chemistry_thermal["wrong_chemistry_after_detector_calibration_pct"], 15.747541716937018)
            and close(chemistry_thermal["detector_matched_max_error_pct"], 0.0074654250901917685)
            and close(chemistry_thermal["quadrant_isothermal_closed_p95_pct"], 6.584460820823842)
            and close(chemistry_thermal["quadrant_wrong_hot_closed_p95_pct"], 20.372499606649814)
            and close(chemistry_thermal["quadrant_matched_closed_p95_pct"], 2.695137129880548)
            and int(chemistry_thermal["n_pass_checks"]) == int(chemistry_thermal["n_checks"]) == 7
        )
        ok = chemistry_thermal_ok
        detail = (
            "chemistry, detector and thermal-identity numbers are script-generated"
            if ok
            else "expected 15.75/0.0075/6.58/20.37/2.70 numbers and 7/7 checks"
        )
    add(rows, "chemistry_thermal_identity_rebuilt", ok, detail)

    table_paths = [
        MANUSCRIPT / "generated_observation_ambiguity_table.tex",
        RESULTS / "observation_ambiguity_table.csv",
    ]
    ok, detail = exists_all(table_paths)
    add(rows, "main_observation_ambiguity_table", ok, detail)
    table_text = read(table_paths[0]) if table_paths[0].exists() else ""
    table_csv_text = read(table_paths[1]) if table_paths[1].exists() else ""

    data_source_paths = [
        ROOT / "code" / "build_data_source_use_table.py",
        MANUSCRIPT / "generated_data_source_use_table.tex",
        RESULTS / "data_source_use_table.csv",
    ]
    ok, detail = exists_all(data_source_paths)
    data_source_text = read(data_source_paths[1]) if data_source_paths[1].exists() else ""
    if ok:
        data_source_markers = [
            "Data provenance and validation hierarchy for the source inverse",
            "fitting roles",
            "model-selection roles",
            "held-out checks",
            "computed reduced-equation histories",
            "computed 3D packed-bed response family",
            "digitized external experimental curve",
            "mechanism support without dense PINN labels",
            "computed generic 288-component reduced tritium-balance histories",
            "reactor-specific fields are required for blanket performance estimates",
        ]
        missing = [marker for marker in data_source_markers if marker not in data_source_text]
        ok = (
            not missing
            and r"\input{generated_data_source_use_table.tex}" in main_text
            and "build_data_source_use_table.py" in reproduce_text
        )
        detail = (
            "data source table aligns"
            if ok
            else "; ".join(missing)
            + ("; main input missing" if r"\input{generated_data_source_use_table.tex}" not in main_text else "")
            + ("; reproduce hook missing" if "build_data_source_use_table.py" not in reproduce_text else "")
        )
    add(rows, "data_provenance_validation_hierarchy_table", ok, detail)

    lit_obs_paths = [
        ROOT / "code" / "build_literature_observable_matrix_table.py",
        ROOT / "literature_data" / "observable_matrix.csv",
        MANUSCRIPT / "generated_literature_observable_matrix_table.tex",
        MANUSCRIPT / "generated_literature_observable_summary.tex",
    ]
    ok, detail = exists_all(lit_obs_paths)
    lit_obs_text = read(lit_obs_paths[2]) if lit_obs_paths[2].exists() else ""
    lit_obs_main_text = read(lit_obs_paths[3]) if lit_obs_paths[3].exists() else ""
    lit_obs_csv = read(lit_obs_paths[1]) if lit_obs_paths[1].exists() else ""
    if ok:
        lit_obs_markers = [
            "Experimental records from lithium-ceramic breeder-release literature",
            "Park/FEC 2025 [18]",
            "Zhu et al. 2013 [10]",
            "Kobayashi/Oya 2018 [11]",
            "Kinjyo et al. 2008 [9]",
            "Xiao et al. 2013 [12]",
            "Kang et al. 2015 [15]",
            "TRIO-01/LISA1 in-situ tests [4,5]",
            "Water-vapor and surface-water studies [6--8]",
            "EXOTIC-6 purge-gas study",
            "doi:10.1016/b978-0-444-89995-8.50276-9",
            "doi:10.1016/j.jnucmat.2010.12.130",
            "doi:10.1016/s0920-3796(01)00527-0",
            "water-bubbler collection",
            "TDS partial-pressure spectra",
            "HT/HTO fractions",
            "dense labels are generated only after the target-bed response family is calibrated",
            "Supplementary Table S1 groups 13 literature record rows",
            "including 7 Li$_2$TiO$_3$ and 3 Li$_4$SiO$_4$ rows",
            "dynamic release or trapped-release histories in 9 rows",
            "HT/HTO or water-collection records in 7",
            "purge-flow or gas-chemistry records in 9",
            "temperature or TDS programs in 6",
            "material-state records in 5",
            "recovery or extraction-system records in 3",
            "Existing lithium-ceramic release studies report several records around the purge-gas history",
            "The release curve gives the purge-gas time history",
            "The companion records constrain species chemistry, residence time",
            "uses these measurement families as physical inputs for source separation",
            "Computed PINN histories extend source amplitude, chemistry and temperature variation inside",
            "target-bed response family calibrated by flow",
            "Withheld cases change the release mechanism, 3D field, chemistry field, thermal field or balance history",
        ]
        missing = [
            marker
            for marker in lit_obs_markers
            if marker not in lit_obs_text and marker not in lit_obs_csv and marker not in lit_obs_main_text
        ]
        ok = (
            not missing
            and r"\input{generated_literature_observable_summary.tex}" in main_text
            and r"\input{generated_literature_observable_matrix_table.tex}" in si_text
            and "Lithium-ceramic release experiments report coupled records"
            in si_text
            and "build_literature_observable_matrix_table.py" in reproduce_text
        )
        detail = (
            "literature observable table aligns"
            if ok
            else "; ".join(missing)
            + (
                "; main summary input missing"
                if r"\input{generated_literature_observable_summary.tex}" not in main_text
                else ""
            )
            + (
                "; SI input missing"
                if r"\input{generated_literature_observable_matrix_table.tex}" not in si_text
                else ""
            )
            + (
                "; SI framing paragraph missing"
                if "Lithium-ceramic release experiments report coupled records"
                not in si_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_literature_observable_matrix_table.py" not in reproduce_text
                else ""
            )
        )
    add(rows, "literature_observable_measurement_table", ok, detail)

    observable_framing_markers = [
        "The TRIO-01 experiment measured in-situ tritium recovery",
        "LISA1 added an in-pile release record",
        "Surface-water studies connected adsorbed water to the release rate",
        "Water-adsorption experiments linked adsorbed water to the HTO response",
        r"\input{generated_literature_observable_summary.tex}",
        "These records make the outlet curve a dynamic signal with several physical inputs",
        "The companion records constrain species chemistry, residence time, thermal state, material state and tritium balance",
        "Computed PINN histories extend source amplitude, chemistry and temperature variation inside",
        "target-bed response family calibrated by flow",
        "This paper develops a measurement-constrained PINN source-recovery framework",
    ]
    observable_framing_text = text + "\n" + lit_obs_main_text
    missing = [
        marker
        for marker in observable_framing_markers
        if marker not in observable_framing_text
    ]
    add(
        rows,
        "observable_framing_not_single_outlet_premise",
        not missing,
        "observable framing keeps literature records and inverse-design question separated"
        if not missing
        else "; ".join(missing),
    )

    observable_necessity_paths = [
        ROOT / "code" / "build_observable_necessity_text.py",
        MANUSCRIPT / "generated_observable_necessity_summary.tex",
        MANUSCRIPT / "generated_observable_necessity_si.tex",
        RESULTS
        / "observable_nullspace_closure_synthesis"
        / "observable_nullspace_closure_summary.json",
        RESULTS
        / "observable_necessity_leave_one_out"
        / "observable_necessity_summary.json",
    ]
    ok, detail = exists_all(observable_necessity_paths)
    obs_main = read(observable_necessity_paths[1]) if observable_necessity_paths[1].exists() else ""
    obs_si = read(observable_necessity_paths[2]) if observable_necessity_paths[2].exists() else ""
    if ok:
        markers = [
            "Outlet-only histories leave 7 unresolved parameter combinations",
            "Using all 7 measurement types gives rank 10 and no unresolved linear direction",
            "smallest positive singular value is 10.18",
            "corresponding condition number is 13.3",
            "spatial source uncertainty by 16.8-fold",
            "thermal/species uncertainty by 16.9-fold",
            "transport uncertainty by 3.89-fold",
            "recovery uncertainty by 14.5-fold",
        ]
        missing = [marker for marker in markers if marker not in obs_main]
        ok = (
            not missing
            and r"\input{generated_observable_necessity_summary.tex}" in main_text
            and r"\input{generated_observable_necessity_si.tex}" in si_text
            and "build_observable_necessity_text.py" in reproduce_text
            and "observable_nullspace_closure_summary.json" in obs_si
        )
        detail = (
            "observable null-space and leave-one-measurement text aligns"
            if ok
            else "; ".join(missing)
            + (
                "; main input missing"
                if r"\input{generated_observable_necessity_summary.tex}" not in main_text
                else ""
            )
            + (
                "; SI input missing"
                if r"\input{generated_observable_necessity_si.tex}" not in si_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_observable_necessity_text.py" not in reproduce_text
                else ""
            )
            + (
                "; nullspace source missing"
                if "observable_nullspace_closure_summary.json" not in obs_si
                else ""
            )
        )
    add(rows, "observable_necessity_nullspace_text", ok, detail)

    core_number_pairs = [
        ("7.29\\%", "0.57\\%"),
        ("32.07\\%", "matched-bed error range"),
        ("9.95\\%", "matched-bed error range"),
    ]
    missing_pairs: list[str] = []
    for fail_value, pass_value in core_number_pairs:
        if fail_value not in main_text and fail_value not in table_text:
            missing_pairs.append(fail_value)
        if pass_value not in main_text and pass_value not in table_text:
            missing_pairs.append(pass_value)
    table_in_main_ok = "generated_observation_ambiguity_table.tex" in main_text
    table_core_ok = (
        "1D-on-3D closed p95 32.07\\%" in table_text
        and "matched-bed error range" in table_text
        and table_in_main_ok
    )
    add(
        rows,
        "core_number_chain_consistency",
        not missing_pairs and table_core_ok,
        "main observation table core numbers align"
        if not missing_pairs and table_core_ok
        else "; ".join(missing_pairs) + ("; main observation table or 3D chain missing" if not table_core_ok else ""),
    )

    external_curve_table_ok = (
        "interpolated dense closed error 28.02\\%" in table_text
        and "HT tau 158.5 min; HTO tau 123.3 min; matched 3D closed error 0.535\\%" in table_text
        and "real_curve_regularized_3d_inverse_summary.csv" in table_csv_text
    )
    add(
        rows,
        "external_curve_table_role",
        external_curve_table_ok,
        "external-curve table row separates harmful dense interpolation from weak time-scale use"
        if external_curve_table_ok
        else "external-curve table row or source artifact link missing",
    )

    tds_paths = [
        ROOT / "code" / "build_li2_tds_mechanism_summary.py",
        RESULTS
        / "li2_tds_closed_pathway_mechanism_closure"
        / "li2_tds_closed_pathway_mechanism_closure_summary.json",
        RESULTS
        / "li2_tds_closed_pathway_mechanism_closure"
        / "li2_tds_closed_pathway_mechanism_closure_checks.csv",
    ]
    ok, detail = exists_all(tds_paths)
    if ok:
        tds = json.loads(read(tds_paths[1]))["summary"]
        checks = list(csv.DictReader(tds_paths[2].open(encoding="utf-8")))
        tds_ok = (
            "build_li2_tds_mechanism_summary.py" in reproduce_text
            and "build_li2_tds_mechanism_summary.py" in si_text
            and int(tds["kobayashi_total_points"]) == 1105
            and close(tds["kobayashi_high_full_gain"], 11.715522163233674)
            and close(tds["kobayashi_high_low_gain"], 14.236720579968093)
            and close(tds["zhu_high_area_fraction_min"], 0.1869783352031837)
            and close(tds["zhu_high_area_fraction_max"], 0.3211015114794739)
            and close(tds["closed_response_spearman"], 1.0)
            and close(tds["closed_response_shape_rmse"], 0.18556891940885858)
            and all(row["status"] == "PASS" for row in checks)
        )
        detail = (
            "TDS mechanism numbers are script-generated and reproduced"
            if tds_ok
            else "expected TDS script, 1105 points, 11.7/14.2 gains and PASS checks"
        )
        ok = tds_ok
    add(rows, "li2_tds_mechanism", ok, detail)

    cfdem_table_paths = [
        MANUSCRIPT / "generated_cfdem_transport_check_table.tex",
        RESULTS / "cfdem_transport_check_table.csv",
    ]
    ok, detail = exists_all(cfdem_table_paths)
    add(rows, "cfdem_transport_check_table", ok, detail)
    cfdem_table_text = read(cfdem_table_paths[0]) if cfdem_table_paths[0].exists() else ""
    packed_bed_current_path = RESULTS / "packed_bed_transport_tests_current_summary.json"
    packed_bed_current = json.loads(read(packed_bed_current_path)) if packed_bed_current_path.exists() else {}
    pressure_noise_path = (
        RESULTS
        / "cfdem_pressure_anchor_noise_robustness"
        / "cfdem_pressure_anchor_noise_robustness_summary.json"
    )
    pressure_noise = json.loads(read(pressure_noise_path)) if pressure_noise_path.exists() else {}
    packed_bed_current_ok = (
        packed_bed_current
        and close(packed_bed_current["heterogeneous_bypass_uniform_mixed_closed_error_pct"], 23.658536585365848)
        and close(packed_bed_current["heterogeneous_bypass_matched_3d_four_outlet_closed_error_pct"], 0.2439024390243769)
        and close(packed_bed_current["mesh_uniform_mixed_max_closed_error_pct"], 23.658536585365848)
        and close(packed_bed_current["mesh_matched_3d_four_outlet_max_closed_error_pct"], 0.2439024390243769)
    )
    pressure_noise_ok = (
        pressure_noise
        and int(pressure_noise["n_trials"]) == 18000
        and close(pressure_noise["two_pct_pressure_noise_p95_source_error_pct"], 2.09903901118601, 0.0005)
        and close(pressure_noise["two_pct_pressure_noise_pass_rate_pct"], 100.0, 0.0005)
        and close(pressure_noise["five_pct_pressure_noise_p95_source_error_pct"], 6.454897591246361, 0.0005)
        and int(pressure_noise["n_pass_checks"]) == int(pressure_noise["n_checks"])
    )
    cfdem_markers = [
        "Ergun-form pressure fit",
        "max shape error below 0.1\\%",
        "mass-flow residual below 0.1\\%",
        "source scaling deviation below 0.1\\%",
        "t_{50}=0.815",
        "leave-one-velocity source error 28.21\\%",
        "pressure-calibrated source error returns to the pressure/RTD reference",
        "648--3000 cells",
        "matched max stays in the matched-bed error range",
        "computed histories 128",
        "hard sparse 5x p95 4.04\\%",
    ]
    missing = [marker for marker in cfdem_markers if marker not in cfdem_table_text]
    add(
        rows,
        "cfdem_physical_check_numbers",
        not missing and packed_bed_current_ok and pressure_noise_ok and "generated_cfdem_transport_check_table.tex" in si_text,
        "CFD-DEM transport checks align"
        if not missing and packed_bed_current_ok and pressure_noise_ok and "generated_cfdem_transport_check_table.tex" in si_text
        else "; ".join(missing)
        + ("; packed_bed_transport_tests_current_summary mismatch" if not packed_bed_current_ok else "")
        + ("; pressure-noise robustness mismatch" if not pressure_noise_ok else ""),
    )

    module_table_paths = [
        MANUSCRIPT / "generated_module_scale_interface_table.tex",
        RESULTS / "module_scale_interface_table.csv",
    ]
    ok, detail = exists_all(module_table_paths)
    add(rows, "module_scale_source_propagation_table", ok, detail)
    module_table_text = read(module_table_paths[0]) if module_table_paths[0].exists() else ""
    module_markers = [
        "module $t_{50}=1.10$ h",
        "80 cells",
        "uniform-total module bias 36.73\\%",
        "matched HT/HTO bias reaches the matched-chemistry result",
        "isothermal closed error 13.59\\%",
        "matched closed error reaches the calibrated-thermal result",
        "spatial/species radius 0.015",
    ]
    missing = [marker for marker in module_markers if marker not in module_table_text]
    add(
        rows,
        "module_scale_source_propagation_numbers",
        not missing and "generated_module_scale_interface_table.tex" in si_text,
        "module-scale source-propagation numbers align"
        if not missing and "generated_module_scale_interface_table.tex" in si_text
        else "; ".join(missing),
    )

    training_table_paths = [
        MANUSCRIPT / "generated_main_pinn_validation_table.tex",
        RESULTS / "main_pinn_validation_table.csv",
        MANUSCRIPT / "generated_pinn_training_validation_table.tex",
        RESULTS / "pinn_training_validation_separation.csv",
    ]
    ok, detail = exists_all(training_table_paths)
    main_split_table_included = (
        r"\input{generated_main_pinn_validation_table.tex}" in main_text
        or r"\input{generated_main_pinn_validation_table.tex}" in si_text
    )
    if ok and not main_split_table_included:
        ok = False
        detail = "submission sources do not include generated_main_pinn_validation_table.tex"
    add(rows, "pinn_training_validation_table", ok, detail)
    main_training_table = MANUSCRIPT / "generated_main_pinn_validation_table.tex"
    si_training_table = MANUSCRIPT / "generated_pinn_training_validation_table.tex"
    main_training_table_text = read(main_training_table) if main_training_table.exists() else ""
    training_table_text = read(si_training_table) if si_training_table.exists() else ""
    main_training_markers = [
        "PINN training, differentiable inverse and source-estimation calculations",
        "fixed-basis 3D release-layer row evaluates withheld curve prediction",
        "pooled worst p95 24.05\\%",
        "3D transport model selection",
        "selected closed/zone p95 3.65\\%/4.02\\%",
        "residual-only zone 95.07\\%",
        "pooled closed 25.40\\%",
        "New source-family calibration",
        "uncalibrated zone/closed p95 35.05\\%/14.29\\%",
        "K=16 zone/closed p95 3.01\\%/3.76\\%",
        "first source-zone below 5\\% at K=8",
        "Training-field coverage",
        "evaluation inside domain 98.98\\%",
        "severe high-error capture 98.38\\%",
        "in-domain severe closed/zone p95 3.91\\%/2.33\\%",
        "leave-one median/max 0.127/0.484",
        "matched 3D closed error 0.535\\%",
        "total-only RMSE 0.0628",
        "mismatched chemistry 11.57\\%",
        "interpolated dense labels 28.02\\%",
        "full experiment-visible set 3.63\\%; solid-inventory representation closes the representation error; prediction without that input 73.47\\%",
    ]
    missing = [marker for marker in main_training_markers if marker not in main_training_table_text]
    add(
        rows,
        "main_pinn_training_validation_table_numbers",
        not missing,
        "main PINN training/validation table aligns" if not missing else "; ".join(missing),
    )

    training_observability_paths = [
        ROOT / "code" / "build_training_data_observability_summary.py",
        MANUSCRIPT / "generated_training_data_observability_summary.tex",
        MANUSCRIPT / "generated_training_data_observability_si.tex",
    ]
    ok, detail = exists_all(training_observability_paths)
    train_obs_text = (
        read(training_observability_paths[1])
        if training_observability_paths[1].exists()
        else ""
    )
    if ok:
        markers = [
            "Using 1800 histories with the total-release measurement leaves a 9.99\\%",
            "full spatial HT/HTO and balance measurement set reaches 4.53\\%",
            "source extraction from the residual field leaves 38.34\\%",
            "explicit source-term estimate gives 3.25\\%",
        ]
        missing = [marker for marker in markers if marker not in train_obs_text]
        ok = (
            not missing
            and r"\input{generated_training_data_observability_summary.tex}" in main_text
            and "build_training_data_observability_summary.py" in reproduce_text
        )
        detail = (
            "training-data separation summary aligns"
            if ok
            else "; ".join(missing)
            + (
                "; main input missing"
                if r"\input{generated_training_data_observability_summary.tex}" not in main_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_training_data_observability_summary.py" not in reproduce_text
                else ""
            )
        )
    add(rows, "training_data_observability_summary", ok, detail)

    target_observability_paths = [
        ROOT / "code" / "build_target_observability_phase_diagram.py",
        RESULTS
        / "target_observability_phase_diagram"
        / "target_observability_phase_diagram.csv",
        RESULTS
        / "target_observability_phase_diagram"
        / "target_observability_phase_diagram_summary.json",
        MANUSCRIPT / "generated_target_observability_table.tex",
    ]
    ok, detail = exists_all(target_observability_paths)
    target_table_text = (
        read(target_observability_paths[3])
        if target_observability_paths[3].exists()
        else ""
    )
    if ok:
        target_summary = json.loads(read(target_observability_paths[2]))["summary"]
        markers = [
            "two-parameter 3D release multipliers",
            "thermal 3D source map",
            "component tritium-balance vector",
            "module uncertainty vector",
            "39.23\\% $\\rightarrow$ 0.26\\%",
            "130.00\\% $\\rightarrow$ 3.13\\%",
        ]
        missing = [marker for marker in markers if marker not in target_table_text]
        ok = (
            not missing
            and target_summary["n_targets"] == 6
            and target_summary["n_pass_checks"] == target_summary["n_checks"]
            and r"\input{generated_target_observability_table.tex}" in si_text
            and r"\input{generated_target_observability_table.tex}" in main_text
            and "build_target_observability_phase_diagram.py" in reproduce_text
            and "A two-multiplier 3D release target" in main_text
        )
        detail = (
            "measurement requirements by target table aligns"
            if ok
            else "; ".join(missing)
            + (
                "; SI input missing"
                if r"\input{generated_target_observability_table.tex}" not in si_text
                else ""
            )
            + (
                "; main input missing"
                if r"\input{generated_target_observability_table.tex}" not in main_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_target_observability_phase_diagram.py" not in reproduce_text
                else ""
            )
            + (
                "; main target paragraph missing"
                if "A two-multiplier 3D release target" not in main_text
                else ""
            )
        )
    add(rows, "target_dependent_observability_escalation", ok, detail)

    channel_integrity_paths = [
        ROOT / "code" / "build_pinn_channel_integrity_summary.py",
        MANUSCRIPT / "generated_pinn_channel_integrity_summary.tex",
        MANUSCRIPT / "generated_pinn_channel_integrity_si.tex",
        RESULTS
        / "component_3d_neural_information_scrambling"
        / "component_3d_neural_information_scrambling_summary.json",
        RESULTS
        / "component_3d_neural_sensor_degradation"
        / "component_3d_neural_sensor_degradation_summary.json",
    ]
    ok, detail = exists_all(channel_integrity_paths)
    channel_text = read(channel_integrity_paths[1]) if channel_integrity_paths[1].exists() else ""
    if ok:
        markers = [
            "worst closed-pathway p95 error is 4.02\\%",
            "92.16\\%, 93.32\\% and 86.33\\%",
            "A total-release plus balance vector remains at 14.52\\%",
            "total release amplitude alone is insufficient",
        ]
        missing = [marker for marker in markers if marker not in channel_text]
        ok = (
            not missing
            and r"\input{generated_pinn_channel_integrity_summary.tex}" in main_text
            and r"\input{generated_pinn_channel_integrity_si.tex}" in si_text
            and "build_pinn_channel_integrity_summary.py" in reproduce_text
        )
        detail = (
            "PINN channel-integrity summary aligns"
            if ok
            else "; ".join(missing)
            + (
                "; main input missing"
                if r"\input{generated_pinn_channel_integrity_summary.tex}" not in main_text
                else ""
            )
            + (
                "; SI input missing"
                if r"\input{generated_pinn_channel_integrity_si.tex}" not in si_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_pinn_channel_integrity_summary.py" not in reproduce_text
                else ""
            )
        )
    add(rows, "pinn_channel_integrity_summary", ok, detail)

    method_baseline_paths = [
        ROOT / "code" / "build_method_baseline_observability_table.py",
        MANUSCRIPT / "generated_method_baseline_observability_table.tex",
        RESULTS / "method_baseline_observability_table.csv",
        RESULTS / "inverse_baseline_comparison.csv",
        RESULTS / "observability_threshold_analysis" / "observability_threshold_summary.json",
    ]
    ok, detail = exists_all(method_baseline_paths)
    method_table_text = read(method_baseline_paths[1]) if method_baseline_paths[1].exists() else ""
    if ok:
        method_markers = [
            "Same-equation classical inverse",
            "exact FVM at matched reduced-equation reference; 1\\% noisy FVM 0.481\\%",
            "PINN same-equation comparison",
            "staged PINN closed error 0.57\\%; zero-physics control 40.00\\%",
            "Outlet-only profile likelihood",
            "Classical and neural 3D baselines",
            "full vector linear/neural 4.20\\%/3.34\\%; total-only 12.57\\%/10.48\\%",
            "3D source-estimation withheld split",
            "selected closed/zone 3.65\\%/4.02\\%; residual-only zone 95.07\\%",
            "Measurement-set source estimation",
            "zone uncertainty 25.82\\% $\\rightarrow$ 7.61\\% $\\rightarrow$ 3.74\\%",
            "System source estimation",
            "six-measurement set resolves source only; full measurement set resolves all listed quantities; TES uncertainty 4.65\\%",
        ]
        missing = [marker for marker in method_markers if marker not in method_table_text]
        ok = (
            not missing
            and r"\input{generated_method_baseline_observability_table.tex}" in main_text
            and "build_method_baseline_observability_table.py" in reproduce_text
        )
        detail = (
            "method baseline table aligns"
            if ok
            else "; ".join(missing)
            + ("; main input missing" if r"\input{generated_method_baseline_observability_table.tex}" not in main_text else "")
            + ("; reproduce hook missing" if "build_method_baseline_observability_table.py" not in reproduce_text else "")
        )
    add(rows, "method_baseline_observability_table", ok, detail)

    source_structure_paths = [
        ROOT / "code" / "build_source_structure_summary.py",
        MANUSCRIPT / "generated_source_structure_summary.tex",
        RESULTS
        / "fvm3d_source_uncertainty_profile_gate"
        / "fvm3d_source_uncertainty_profile_gate_summary.json",
    ]
    ok, detail = exists_all(source_structure_paths)
    source_structure_text = read(source_structure_paths[1]) if source_structure_paths[1].exists() else ""
    if ok:
        source_structure_markers = [
            "260 profile fits",
            "720 training histories",
            "360 severe-field histories",
            "2.22\\% full-source $L_1$ p95 error",
            "source-map interval width of 1.93\\%",
            "closest incompatible source-map family remains at 17.27\\% p95 error",
            "7.8 times the compatible case",
            "All incompatible source models exceed either the 5.00\\% point-error scale or the 10.00\\% interval-width scale",
        ]
        missing = [marker for marker in source_structure_markers if marker not in source_structure_text]
        ok = (
            not missing
            and r"\input{generated_source_structure_summary.tex}" in main_text
            and "build_source_structure_summary.py" in reproduce_text
            and "generated_source_structure_summary.tex" in pdf_build_text
        )
        detail = (
            "source-structure summary aligns"
            if ok
            else "; ".join(missing)
            + ("; main input missing" if r"\input{generated_source_structure_summary.tex}" not in main_text else "")
            + ("; reproduce hook missing" if "build_source_structure_summary.py" not in reproduce_text else "")
            + ("; PDF build copy missing" if "generated_source_structure_summary.tex" not in pdf_build_text else "")
        )
    add(rows, "source_structure_summary", ok, detail)

    threshold_sensitivity_paths = [
        ROOT / "code" / "build_source_error_threshold_sensitivity_table.py",
        MANUSCRIPT / "generated_source_error_threshold_sensitivity_table.tex",
        RESULTS / "source_error_threshold_sensitivity_table.csv",
        RESULTS / "source_error_threshold_sensitivity_summary.json",
    ]
    ok, detail = exists_all(threshold_sensitivity_paths)
    threshold_table_text = read(threshold_sensitivity_paths[1]) if threshold_sensitivity_paths[1].exists() else ""
    if ok:
        threshold_markers = [
            "Sensitivity of the 5\\% source-error reference",
            "2.00\\% & 0 & none",
            "5.00\\% & 1 & spatial HT/HTO + inventory + TES",
            "10.00\\% & 2 & spatial HT/HTO + inventory; spatial HT/HTO + inventory + TES",
        ]
        missing = [marker for marker in threshold_markers if marker not in threshold_table_text]
        ok = (
            not missing
            and r"\input{generated_source_error_threshold_sensitivity_table.tex}" in si_text
            and "build_source_error_threshold_sensitivity_table.py" in reproduce_text
            and "generated_source_error_threshold_sensitivity_table.tex" in pdf_build_text
            and "A 2\\%, 5\\% and 10\\% scale scan in the Supplement gives the same ordering" in main_text
        )
        detail = (
            "source-error threshold sensitivity table aligns"
            if ok
            else "; ".join(missing)
            + (
                "; SI input missing"
                if r"\input{generated_source_error_threshold_sensitivity_table.tex}" not in si_text
                else ""
            )
            + (
                "; reproduce hook missing"
                if "build_source_error_threshold_sensitivity_table.py" not in reproduce_text
                else ""
            )
            + (
                "; PDF build copy missing"
                if "generated_source_error_threshold_sensitivity_table.tex" not in pdf_build_text
                else ""
            )
            + (
                "; main text summary missing"
                if "A 2\\%, 5\\% and 10\\% scale scan in the Supplement gives the same ordering" not in main_text
                else ""
            )
        )
    add(rows, "source_error_threshold_sensitivity_table", ok, detail)

    uncertainty_paths = [
        ROOT / "code" / "build_uncertainty_operating_envelope_table.py",
        MANUSCRIPT / "generated_uncertainty_operating_envelope_table.tex",
        RESULTS / "uncertainty_operating_envelope_table.csv",
    ]
    ok, detail = exists_all(uncertainty_paths)
    uncertainty_table_text = read(uncertainty_paths[1]) if uncertainty_paths[1].exists() else ""
    if ok:
        uncertainty_markers = [
            "Outlet noise and fine-tuning",
            "staged max 2.87\\%; full fine-tune max 9.49\\%",
            "Velocity and pressure/RTD",
            "2\\% outlet-only source p95 5.42\\%; pressure/RTD source p95 2.32\\%",
            "Detector calibration",
            "finite calibration p95 6.24\\%; metadata-calibrated p95 4.48\\%",
            "Thermal-field identity",
            "mixed source p95 130.00\\%; four-quadrant source/closed 3.13\\%/2.73\\%",
            "External-curve digitization",
            "8 points; time-scale inventory reduction 16.08 percentage points; trend pass 3/4",
            "Source-structure uncertainty",
            "full source $L_1$ p95 2.22\\%; wrong-family p95 17.27\\%; incompatible families passing 0",
            "Module propagation",
            "within-limit fraction 67.5\\% $\\rightarrow$ 76.0\\%; $t_{90}$ half-width 2.21 h $\\rightarrow$ 0.95 h",
            "System-scale coverage",
            "covered transfer paths 1; outside-range paths 3",
        ]
        missing = [marker for marker in uncertainty_markers if marker not in uncertainty_table_text]
        ok = (
            not missing
            and r"\input{generated_uncertainty_operating_envelope_table.tex}" in si_text
            and "build_uncertainty_operating_envelope_table.py" in reproduce_text
        )
        detail = (
            "uncertainty and operating-envelope table aligns"
            if ok
            else "; ".join(missing)
            + ("; SI input missing" if r"\input{generated_uncertainty_operating_envelope_table.tex}" not in si_text else "")
            + ("; reproduce hook missing" if "build_uncertainty_operating_envelope_table.py" not in reproduce_text else "")
        )
    add(rows, "uncertainty_operating_envelope_table", ok, detail)

    error_metric_paths = [
        ROOT / "code" / "build_error_metric_definition_table.py",
        MANUSCRIPT / "generated_error_metric_definition_table.tex",
        RESULTS / "error_metric_definition_table.csv",
    ]
    ok, detail = exists_all(error_metric_paths)
    error_metric_text = read(error_metric_paths[1]) if error_metric_paths[1].exists() else ""
    if ok:
        markers = [
            "closed-pathway multiplier error",
            "source-zone p95",
            "full-source $L_1$ p95",
            "matched 3D-response result",
            "closed computed comparisons with the declared transport response",
        ]
        missing = [marker for marker in markers if marker not in error_metric_text]
        error_metric_table_included = (
            r"\input{generated_error_metric_definition_table.tex}" in main_text
            or r"\input{generated_error_metric_definition_table.tex}" in si_text
        )
        ok = (
            not missing
            and error_metric_table_included
            and "build_error_metric_definition_table.py" in reproduce_text
            and "build_error_metric_definition_table.py" in si_text
        )
        detail = (
            "error metric definition table is generated and included"
            if ok
            else "; ".join(missing)
            + ("; submission input missing" if not error_metric_table_included else "")
            + ("; reproduce hook missing" if "build_error_metric_definition_table.py" not in reproduce_text else "")
            + ("; SI path missing" if "build_error_metric_definition_table.py" not in si_text else "")
        )
    add(rows, "error_metric_definition_table", ok, detail)

    kernel_uncertainty_paths = [
        ROOT / "code" / "build_kernel_uncertainty_sensitivity_table.py",
        MANUSCRIPT / "generated_kernel_uncertainty_sensitivity_table.tex",
        RESULTS / "kernel_uncertainty_sensitivity_table.csv",
    ]
    ok, detail = exists_all(kernel_uncertainty_paths)
    kernel_text = read(kernel_uncertainty_paths[1]) if kernel_uncertainty_paths[1].exists() else ""
    if ok:
        markers = [
            "unmodelled slow kernel",
            "68.42\\%",
            "unmodelled fast kernel",
            "86.19\\%",
            "field-specific slow/fast kernels",
            "wall-bypass RTD shape",
            "velocity-support extrapolation",
        ]
        missing = [marker for marker in markers if marker not in kernel_text]
        ok = (
            not missing
            and r"\input{generated_kernel_uncertainty_sensitivity_table.tex}" in main_text
            and "build_kernel_uncertainty_sensitivity_table.py" in reproduce_text
            and "build_kernel_uncertainty_sensitivity_table.py" in si_text
        )
        detail = (
            "3D residence-time-response uncertainty table is generated and included"
            if ok
            else "; ".join(missing)
            + ("; main input missing" if r"\input{generated_kernel_uncertainty_sensitivity_table.tex}" not in main_text else "")
            + ("; reproduce hook missing" if "build_kernel_uncertainty_sensitivity_table.py" not in reproduce_text else "")
            + ("; SI path missing" if "build_kernel_uncertainty_sensitivity_table.py" not in si_text else "")
        )
    add(rows, "kernel_uncertainty_sensitivity_table", ok, detail)

    source_shape_paths = [
        ROOT / "code" / "build_neutronics_source_shape_table.py",
        MANUSCRIPT / "generated_neutronics_source_shape_table.tex",
        RESULTS / "neutronics_source_shape_table.csv",
        RESULTS
        / "neutronics_source_shape_accountancy"
        / "neutronics_source_shape_summary.json",
    ]
    ok, detail = exists_all(source_shape_paths)
    source_shape_text = read(source_shape_paths[1]) if source_shape_paths[1].exists() else ""
    if ok:
        source_shape_summary = json.loads(read(source_shape_paths[3]))["summary"]
        markers = [
            "fixed nominal source distribution",
            "51.00\\%",
            "242.62\\%",
            "source-map proxy only",
            "30.45\\%",
            "3.69\\%",
            "4.14\\%",
        ]
        missing = [marker for marker in markers if marker not in source_shape_text]
        ok = (
            not missing
            and close(source_shape_summary["wrong_nominal_source_l1_p95_pct"], 50.99561640348277)
            and close(source_shape_summary["wrong_nominal_closed_p95_pct"], 242.62414461409023)
            and close(source_shape_summary["full_closed_p95_pct"], 3.694921501345114)
            and r"\input{generated_neutronics_source_shape_table.tex}" in main_text
            and "build_neutronics_source_shape_table.py" in reproduce_text
            and "build_neutronics_source_shape_table.py" in si_text
        )
        detail = (
            "generated-tritium source-map table is generated and included"
            if ok
            else "; ".join(missing)
            + ("; source numbers drifted" if not close(source_shape_summary["full_closed_p95_pct"], 3.694921501345114) else "")
            + ("; main input missing" if r"\input{generated_neutronics_source_shape_table.tex}" not in main_text else "")
            + ("; reproduce hook missing" if "build_neutronics_source_shape_table.py" not in reproduce_text else "")
            + ("; SI path missing" if "build_neutronics_source_shape_table.py" not in si_text else "")
        )
    add(rows, "neutronics_source_shape_table", ok, detail)

    static_envelope_paths = [
        ROOT / "code" / "build_static_bed_operating_envelope_table.py",
        MANUSCRIPT / "generated_static_bed_operating_envelope_table.tex",
        RESULTS / "static_bed_operating_envelope_table.csv",
    ]
    ok, detail = exists_all(static_envelope_paths)
    static_envelope_text = read(static_envelope_paths[1]) if static_envelope_paths[1].exists() else ""
    if ok:
        markers = [
            "Operating envelope of the computed static-bed",
            "normalized inlet velocity 0.3, 1.0 and 2.0",
            "velocity scale 0.5--2.0",
            "Re/Pe/Damkohler reporting",
            "CAD/neutronics fields replace it",
        ]
        missing = [marker for marker in markers if marker not in static_envelope_text]
        ok = (
            not missing
            and r"\input{generated_static_bed_operating_envelope_table.tex}" in si_text
            and "build_static_bed_operating_envelope_table.py" in reproduce_text
            and "build_static_bed_operating_envelope_table.py" in si_text
        )
        detail = (
            "static-bed/component operating-envelope table is generated and included"
            if ok
            else "; ".join(missing)
            + ("; SI input missing" if r"\input{generated_static_bed_operating_envelope_table.tex}" not in si_text else "")
            + ("; reproduce hook missing" if "build_static_bed_operating_envelope_table.py" not in reproduce_text else "")
            + ("; SI path missing" if "build_static_bed_operating_envelope_table.py" not in si_text else "")
        )
    add(rows, "static_bed_operating_envelope_table", ok, detail)

    dimensionless_paths = [
        ROOT / "code" / "build_dimensionless_operating_table.py",
        MANUSCRIPT / "generated_dimensionless_operating_table.tex",
        RESULTS / "dimensionless_operating_table.csv",
        RESULTS / "dimensionless_operating_case_values.csv",
    ]
    ok, detail = exists_all(dimensionless_paths)
    dimensionless_text = read(dimensionless_paths[1]) if dimensionless_paths[1].exists() else ""
    if ok:
        markers = [
            "Dimensionless operating groups",
            "1D gas Peclet number",
            "60.0",
            "open/closed release time-scale ratio",
            "80.0",
            "3D gas residence time",
            "0.34--0.52 h",
            "3D HT/HTO Peclet numbers",
            "61.4--93.1/80.8--123",
            "HT to HTO exchange Damkohler number",
            "0.006--0.018",
            "HTO memory to 3D residence-time ratio",
            "0.86--1.31",
            "component max-cell",
        ]
        missing = [marker for marker in markers if marker not in dimensionless_text]
        main_markers = [
            r"\(Pe_L=60\)",
            "61.4--123",
            "0.34--0.52 h",
            "time-scale ratio is 80",
            "0.86--1.31 times the 3D residence time",
        ]
        missing_main = [marker for marker in main_markers if marker not in main_text]
        ok = (
            not missing
            and not missing_main
            and r"\input{generated_dimensionless_operating_table.tex}" in si_text
            and "build_dimensionless_operating_table.py" in reproduce_text
            and "dimensionless groups" in main_text
        )
        detail = (
            "dimensionless operating table is generated and included"
            if ok
            else "; ".join(missing + missing_main)
            + ("; SI input missing" if r"\input{generated_dimensionless_operating_table.tex}" not in si_text else "")
            + ("; reproduce hook missing" if "build_dimensionless_operating_table.py" not in reproduce_text else "")
            + ("; main pointer missing" if "dimensionless groups" not in main_text else "")
        )
    add(rows, "dimensionless_operating_table", ok, detail)

    add(
        rows,
        "training_operator_selection_in_reproduce_workflow",
        "build_fvm3d_training_data_selection.py" in reproduce_text,
        "training-operator selection rebuilt before main PINN table"
        if "build_fvm3d_training_data_selection.py" in reproduce_text
        else "reproduce workflow does not rebuild training-operator selection summary",
    )
    add(
        rows,
        "training_domain_applicability_in_reproduce_workflow",
        "build_fvm3d_training_domain_applicability.py" in reproduce_text,
        "training-domain applicability rebuilt before main PINN table"
        if "build_fvm3d_training_domain_applicability.py" in reproduce_text
        else "reproduce workflow does not rebuild training-domain applicability summary",
    )
    add(
        rows,
        "pde_mlp_leave_one_summary_in_reproduce_workflow",
        "build_fvm3d_trainable_pde_mlp_leave_one_case_summary.py" in reproduce_text,
        "leave-one-case PDE-MLP summary rebuilt before PINN validation table"
        if "build_fvm3d_trainable_pde_mlp_leave_one_case_summary.py" in reproduce_text
        else "reproduce workflow does not rebuild leave-one-case PDE-MLP summary",
    )
    add(
        rows,
        "real_curve_leave_one_summary_in_reproduce_workflow",
        "build_real_curve_leave_one_point_summary.py" in reproduce_text,
        "Park/FEC leave-one-point summary rebuilt before external-curve check"
        if "build_real_curve_leave_one_point_summary.py" in reproduce_text
        else "reproduce workflow does not rebuild Park/FEC leave-one-point summary",
    )
    add(
        rows,
        "external_timescale_system_summary_in_reproduce_workflow",
        "build_external_timescale_to_system_summary.py" in reproduce_text,
        "external time-scale to system-accountancy summary rebuilt before manuscript checks"
        if "build_external_timescale_to_system_summary.py" in reproduce_text
        else "reproduce workflow does not rebuild external time-scale system summary",
    )
    add(
        rows,
        "cfdem_pressure_noise_summary_in_reproduce_workflow",
        "build_cfdem_pressure_noise_summary.py" in reproduce_text,
        "CFD-DEM pressure-noise robustness summary rebuilt before manuscript checks"
        if "build_cfdem_pressure_noise_summary.py" in reproduce_text
        else "reproduce workflow does not rebuild CFD-DEM pressure-noise summary",
    )
    real_curve_lopo = (
        RESULTS
        / "real_curve_leave_one_point_predictive"
        / "real_curve_leave_one_point_summary.json"
    )
    ok = real_curve_lopo.exists()
    detail = "Park/FEC leave-one-point summary present"
    if ok:
        lopo = json.loads(read(real_curve_lopo))["summary"]
        ok = (
            int(lopo["n_holdout_points"]) == 6
            and int(lopo["n_bootstrap_per_holdout"]) == 500
            and close(lopo["median_point_abs_error_norm"], 0.1266576961501781, 0.0005)
            and close(lopo["max_point_abs_error_norm"], 0.4838670141634258, 0.0005)
            and close(lopo["expanded_interval_coverage"], 1.0 / 3.0, 0.0005)
            and int(lopo["n_pass_checks"]) == int(lopo["n_checks"])
        )
        detail = (
            "Park/FEC leave-one-point numbers align"
            if ok
            else "expected 6 holdouts, 500 bootstraps, median 0.127, max 0.484 and coverage 0.333"
        )
    add(rows, "real_curve_leave_one_summary_numbers", ok, detail)

    ext_system_path = (
        RESULTS
        / "external_timescale_to_system_accountancy"
        / "external_timescale_to_system_accountancy_summary.json"
    )
    ok = ext_system_path.exists()
    detail = "external time-scale system-accountancy summary present"
    if ok:
        ext_system = json.loads(read(ext_system_path))["summary"]
        ok = (
            int(ext_system["time_scale_guided_first_system_k"]) == 16
            and int(ext_system["random_first_system_k"]) == 64
            and close(ext_system["time_scale_guided_k16_joint_p95_pct"], 3.412820326616116, 0.0005)
            and close(ext_system["random_k16_joint_p95_worst_pct"], 7.677204053921591, 0.0005)
            and close(ext_system["random_k16_trial_pass_rate"], 0.5, 0.0005)
            and int(ext_system["n_pass_checks"]) == int(ext_system["n_checks"])
        )
        detail = (
            "external time-scale system propagation numbers align"
            if ok
            else "expected guided K16 3.41%, random K16 7.68%, random first K64 and PASS checks"
        )
    add(rows, "external_timescale_system_numbers", ok, detail)

    pde_mlp_path = (
        RESULTS
        / "fvm3d_trainable_pde_mlp_leave_one_case"
        / "fvm3d_trainable_pde_mlp_leave_one_case_summary.json"
    )
    ok = pde_mlp_path.exists()
    detail = "leave-one-case PDE-MLP summary present"
    if ok:
        pde_mlp = json.loads(read(pde_mlp_path))["summary"]
        ok = (
            int(pde_mlp["n_holdout_cases"]) == 3
            and close(pde_mlp["min_pde_residual_gain"], 1.0051786724738456, 0.0005)
            and close(pde_mlp["max_pde_residual_gain"], 1.1464401952391223, 0.0005)
            and close(pde_mlp["max_field_rmse_ratio"], 0.9195848122452148, 0.0005)
            and int(pde_mlp["n_pass_checks"]) == int(pde_mlp["n_checks"])
        )
        detail = (
            "leave-one-case PDE-MLP numbers align"
            if ok
            else "expected 3 cases, gain 1.005--1.146 and field-RMSE ratio 0.920"
        )
    add(rows, "pde_mlp_leave_one_summary_numbers", ok, detail)

    pde_mlp_input_path = (
        RESULTS
        / "fvm3d_trainable_pde_mlp_leave_one_case"
        / "fvm3d_trainable_pde_mlp_leave_one_case_input_check_summary.json"
    )
    ok = pde_mlp_input_path.exists()
    detail = "leave-one-case PDE-MLP input split checks present"
    if ok:
        pde_mlp_input = json.loads(read(pde_mlp_input_path))["summary"]
        ok = (
            int(pde_mlp_input["n_metrics_rows"]) == 18
            and int(pde_mlp_input["n_model_rows"]) == 9
            and int(pde_mlp_input["n_holdout_cases"]) == 3
            and close(pde_mlp_input["min_pde_residual_gain"], 1.0051786724738456, 0.0005)
            and close(pde_mlp_input["max_field_rmse_ratio"], 0.9195848122452148, 0.0005)
            and int(pde_mlp_input["n_pass_checks"]) == int(pde_mlp_input["n_checks"]) == 8
        )
        detail = (
            "leave-one-case PDE-MLP input split checks align"
            if ok
            else "expected 18 metric rows, 9 models, 3 holdout cases and 8/8 input checks"
        )
    add(rows, "pde_mlp_leave_one_input_split_checks", ok, detail)

    selection_path = (
        RESULTS
        / "fvm3d_training_data_selection_gate"
        / "fvm3d_training_data_selection_gate_summary.json"
    )
    ok = selection_path.exists()
    detail = "training-operator selection summary present"
    if ok:
        selection = json.loads(read(selection_path))["summary"]
        ok = (
            int(selection["n_holdout_field_source_splits"]) == 16
            and close(selection["admissible_selector_worst_closed_p95_pct"], 3.647898293944414)
            and close(selection["admissible_selector_worst_zone_p95_pct"], 4.015427476655634)
            and close(selection["residual_only_worst_zone_p95_pct"], 95.07407421427928)
            and close(selection["pooled_no_gate_worst_closed_p95_pct"], 25.40162517764801)
        )
        detail = (
            "training-operator selection numbers align"
            if ok
            else "expected 16 splits, 3.65/4.02 selected, 95.07 residual-only and 25.40 pooled"
        )
    add(rows, "pinn_training_operator_selection_numbers", ok, detail)
    training_table_markers = [
        "0 primary PINN training rows from external curves",
        "matched 3D closed error returns to the matched-bed error range",
        "PDE-derived source multiplier error 93.84\\%",
        "solid-inventory representation closes the representation error; prediction without that input 73.47\\%",
        "full vector: linear 4.20\\%, neural 3.34\\%",
        "total only: linear 12.57\\%, neural 10.48\\%",
    ]
    missing = [marker for marker in training_table_markers if marker not in training_table_text]
    add(
        rows,
        "pinn_training_split_numbers",
        not missing,
        "PINN training/validation numbers align" if not missing else "; ".join(missing),
    )

    data_volume_paths = [
        RESULTS / "fvm3d_data_volume_vs_field_physics" / "fvm3d_data_volume_vs_field_physics_summary.csv",
        RESULTS / "fvm3d_data_volume_vs_field_physics" / "fvm3d_data_volume_vs_field_physics_summary.json",
    ]
    ok, detail = exists_all(data_volume_paths)
    if ok:
        data_volume = json.loads(read(data_volume_paths[1]))["summary"]
        ok = (
            int(data_volume["pooled_max_train_size"]) == 2080
            and close(data_volume["pooled_best_worst_closed_p95_pct"], 24.0450381991563)
            and close(data_volume["pooled_best_worst_zone_l1_p95_pct"], 5.698817738994089)
            and int(data_volume["target_first_joint_pass_n_train"]) == 80
            and close(data_volume["target_520_worst_closed_p95_pct"], 3.527118365332331)
            and close(data_volume["pooled_to_target_520_closed_ratio"], 6.817190609618438)
            and data_volume["pooled_any_joint_pass"] is False
        )
        detail = (
            "data-volume/field-matched numbers align"
            if ok
            else "expected pooled 2080/24.05%, target 80 pass and 520/3.53%"
        )
    add(rows, "pinn_data_volume_target_field_numbers", ok, detail)

    whole_reactor_path = (
        RESULTS
        / "whole_reactor_dynamic_tritium_observability"
        / "whole_reactor_dynamic_accountancy_main_summary.json"
    )
    ok = whole_reactor_path.exists()
    detail = "288-component system dynamic summary present"
    if ok:
        wr = json.loads(read(whole_reactor_path))["summary"]["manuscript_numbers"]
        ok = (
            close(wr["aggregate_tail_p95_pct"], 5.76)
            and close(wr["aggregate_top_tail_accuracy_pct"], 16.7)
            and close(wr["sector_tail_p95_pct"], 0.63)
            and close(wr["sector_top_tail_accuracy_pct"], 96.0)
            and close(wr["full_neural_tail_p95_pct"], 0.20)
            and close(wr["full_neural_risk_p95_pct"], 0.91)
        )
        detail = (
            "288-component system dynamic numbers align"
            if ok
            else "expected 5.76/16.7/0.63/96.0/0.20/0.91"
        )
    add(rows, "whole_reactor_dynamic_numbers", ok, detail)

    whole_reactor_figure_path = (
        RESULTS
        / "whole_reactor_dynamic_tritium_observability"
        / "whole_reactor_accountancy_main_figure_summary.json"
    )
    ok = whole_reactor_figure_path.exists()
    detail = "288-component system figure summary present"
    if ok:
        wr_fig = json.loads(read(whole_reactor_figure_path))
        ok = (
            int(wr_fig["n_test_scenarios"]) == 300
            and close(wr_fig["aggregate_tail_p95_pct"], 5.76)
            and close(wr_fig["aggregate_top_tail_accuracy_pct"], 16.7)
            and close(wr_fig["sector_tail_p95_pct"], 0.63)
            and close(wr_fig["sector_top_tail_accuracy_pct"], 96.0)
            and close(wr_fig["full_neural_tail_p95_pct"], 0.20)
            and close(wr_fig["full_neural_risk_p95_pct"], 0.91)
        )
        detail = (
            "288-component system figure numbers align"
            if ok
            else "expected 300/5.76/16.7/0.63/96.0/0.20/0.91"
        )
    add(rows, "whole_reactor_main_figure_numbers", ok, detail)

    dynamic_source_path = (
        RESULTS
        / "dynamic_source_history_boundary"
        / "dynamic_source_history_boundary_summary.json"
    )
    ok = dynamic_source_path.exists()
    detail = "dynamic source-history boundary summary present"
    if ok:
        dynamic_source = json.loads(read(dynamic_source_path))["summary"]
        amp = dynamic_source["source_amplitude"]
        power = dynamic_source["dynamic_power_history"]
        telemetry = dynamic_source["telemetry_calibration"]
        ok = (
            close(amp["low_kernel_closed_p95_pct"], 17.092422116306647)
            and close(amp["low_kernel_tail_p95_pct"], 86.50301913195206)
            and close(amp["bracketed_inventory_closed_p95_pct"], 1.5650679384832504)
            and close(amp["bracketed_inventory_tail_p95_pct"], 0.0003241276786654743)
            and close(power["constant_mixed_closed_p95_pct"], 218.8522855170255)
            and close(power["known_power_spatial_hthto_closed_p95_pct"], 0.35369515609206603)
            and close(power["known_power_spatial_hthto_source_l1_p95_pct"], 0.05126899471397329)
            and close(telemetry["raw_closed_p95_pct"], 84.34255389241396)
            and close(telemetry["full_calibrated_closed_p95_pct"], 3.5756190654531226)
        )
        detail = (
            "dynamic source-history numbers align"
            if ok
            else "expected source amplitude, dynamic power and telemetry numbers"
        )
    add(rows, "dynamic_source_history_boundary_numbers", ok, detail)

    dynamic_accountancy_path = RESULTS / "dynamic_source_accountancy_law" / "dynamic_source_accountancy_law_summary.json"
    ok = dynamic_accountancy_path.exists()
    detail = "dynamic source-accountancy law summary present"
    if ok:
        accountancy = json.loads(read(dynamic_accountancy_path))
        ok = (
            close(accountancy["independent_balance_residual_fraction"], 0.0011708275856481204)
            and close(accountancy["balanced_balance_residual_fraction"], 1.3444106938820255e-17, 1e-18)
            and int(accountancy["new_local_failures_hidden_by_aggregate"]) == 73
            and accountancy["n_pass_checks"] == accountancy["n_checks"]
        )
        detail = (
            "dynamic source-accountancy balance numbers align"
            if ok
            else "expected balance residual 1.17e-3 -> 1.34e-17 and 73 hidden local failures"
        )
    add(rows, "dynamic_source_accountancy_law_numbers", ok, detail)

    add(
        rows,
        "dynamic_source_history_boundary_in_reproduce_workflow",
        "build_dynamic_source_history_boundary.py" in reproduce_text,
        "dynamic source-history boundary rebuilt before manuscript checks"
        if "build_dynamic_source_history_boundary.py" in reproduce_text
        else "reproduce workflow does not rebuild dynamic source-history boundary",
    )

    resolution_paths = [
        MANUSCRIPT / "generated_whole_reactor_resolution_table.tex",
        RESULTS / "whole_reactor_resolution_table.csv",
        RESULTS
        / "whole_reactor_diagnostic_resolution_convergence"
        / "whole_reactor_diagnostic_resolution_convergence_summary.json",
    ]
    ok, detail = exists_all(resolution_paths)
    resolution_table_text = read(resolution_paths[0]) if resolution_paths[0].exists() else ""
    if ok:
        resolution = json.loads(read(resolution_paths[2]))["summary"]
        markers = [
            r"\(<10^{-4}\)",
            "Aggregate release is below the displayed signal scale",
            "12-outlet coverage",
            "100.0\\%",
            "288",
        ]
        missing = [marker for marker in markers if marker not in resolution_table_text]
        ok = (
            not missing
            and close(resolution["aggregate_group_detection_signal"], 7.675663872883566e-15, 1e-16)
            and close(resolution["group12_top_budget_new_failure_coverage_pct"], 100.0)
            and close(resolution["group18_nonzero_new_failure_visibility_pct"], 100.0)
        )
        detail = (
            "288-component system observation-resolution table aligns"
            if ok
            else "; ".join(missing) or "expected aggregate signal, group12 and group18 resolution numbers"
        )
    add(rows, "whole_reactor_resolution_table", ok, detail)

    minimum_set_paths = [
        MANUSCRIPT / "generated_minimum_measurement_set_table.tex",
        MANUSCRIPT / "generated_main_minimum_measurement_set_table.tex",
        RESULTS / "minimum_measurement_set_table.csv",
        RESULTS
        / "minimal_multifield_diagnostic_prescription"
        / "minimal_multifield_diagnostic_prescription_summary.json",
    ]
    ok, detail = exists_all(minimum_set_paths)
    minimum_set_text = read(minimum_set_paths[0]) if minimum_set_paths[0].exists() else ""
    main_minimum_set_text = read(minimum_set_paths[1]) if minimum_set_paths[1].exists() else ""
    if ok:
        minimum_set = json.loads(read(minimum_set_paths[3]))["summary"]
        markers = [
            "field-calibrated spatial HT/HTO + inventory + TES",
            "2.56\\%",
            "2.55\\%",
            "No: 3D source unresolved",
        ]
        main_markers = [
            "Measurement set needed to carry a local PINN source estimate",
            "field-calibrated spatial HT/HTO + inventory + TES",
            "9.01\\%",
            "8.15\\%",
            "usable",
        ]
        missing = [marker for marker in markers if marker not in minimum_set_text]
        missing += [marker for marker in main_markers if marker not in main_minimum_set_text]
        ok = (
            not missing
            and close(minimum_set["inventory_tes_module_half_width_pct"], 2.556964715933044)
            and close(minimum_set["inventory_tes_p5_2pct_p95_pct"], 2.551735673893668)
            and close(minimum_set["inventory_tes_cross_shift_noise2_worst_p95_pct"], 4.1302948894199165)
            and "generated_minimum_measurement_set_table.tex" in si_text
            and "generated_main_minimum_measurement_set_table.tex" in main_text
        )
        detail = (
            "minimum measurement set table aligns"
            if ok
            else "; ".join(missing) or "expected 2.56/2.55/4.13 numbers"
        )
    add(rows, "minimum_measurement_set_table", ok, detail)

    required_main_phrases = [
        "A single mixed purge-gas outlet merges delayed tritium-release source terms with packed-bed residence time in solid-breeder pebble beds",
        "reduces the closed-pathway multiplier error from 7.29\\% to 0.57\\%",
        "slow/fast four-condition design has a 4.67\\% p95 maximum release-multiplier error",
        "The baseline two-condition design gives 15.00\\% and 46.0\\%",
        "effective release time scales of 158.5 min for HT and 123.3 min for HTO",
        "Bootstrap fits support a single effective time scale",
        "no bootstrap sample separates two independent time scales for either species",
        "no bootstrap sample separates two independent time scales for either species",
        "matched 3D spatial/species closed-pathway error of about 0.5\\%",
        "Total-only and mismatched-chemistry inverses remain biased",
        "The sparse curve narrows the time-scale part of the source uncertainty",
        "Selection by source time scale and 3D field response gives a smaller source/field distance than random selection",
        "this reference set reaches the target system-balance 95th-percentile range with fewer histories than random selection",
        "Over-weighting the same curve raises the matched 3D closed-pathway error to 18.57\\%",
        "Treating interpolated points as dense training histories raises the error to 28.02\\%",
        "TDS curves \\cite{ref11} show an 11.7-fold increase in high-temperature release fraction",
        "TDS data from Li$_2$TiO$_3$ release studies support a delayed high-temperature pathway \\cite{ref10,ref11}",
        "fix the material source and change the residence-time response seen by the outlet",
        "The outlet histories are then replaced by heterogeneous 3D packed-bed histories with the same source target",
        "Using the matched 3D response with spatial/species outlets returns the estimate to the matched-bed error range",
        "mixed-outlet uniform-transport path remains above 5\\% closed-pathway error",
        "mixed-outlet uniform basis gives 23.7\\% closed-pathway error",
        "3D four-outlet path stays in the matched-bed error range for the same field",
        "source-location comparison matches the velocity-specific source estimate",
        "Leaving out velocity or using a nominal velocity produces large source error",
        "Pressure-derived velocity with RTD consistency restores the pressure-calibrated result",
        "Ergun-form packed-bed shape \\cite{ref25}. The fitted coefficient and residual are reported in the Supplement",
        "Pressure-noise sweeps give the velocity-scale precision used in source-multiplier fitting",
        "Removing spatially resolved outlets increases spatial source uncertainty by 16.8-fold",
        "Outlet-only histories leave 7 unresolved parameter combinations",
        "Using all 7 measurement types gives rank 10 and no unresolved linear direction",
        "Removing HT/HTO speciation increases thermal/species uncertainty by 16.9-fold",
        "Removing pressure information increases transport uncertainty by 3.89-fold",
        "Removing retained-inventory or TES information increases recovery uncertainty by 14.5-fold",
        "The neural model combines multi-condition histories, calibrated 3D residence-time responses, species-resolved outlets, detector calibration and tritium balance in one differentiable calculation",
        "The source-estimation calculations start with the same-equation numerical reference",
        "The fitting and prediction histories are split by physical content",
        "Each prediction test leaves out one release mechanism, 3D transport field, chemistry field or thermal field during fitting",
        "the component calculations propagate the evaluated local source through sector summation",
        "The system calculation is a generic 288-component reduced tritium-balance model grouped into 18 sectors",
        "The component interface lists the fields required for a named blanket calculation: geometry, neutronics source, thermal-hydraulic field, purge network and TES response",
        "The reduced system estimator uses 600 dynamic cases and is evaluated on 300 withheld cases",
        "The system source-term output variables are parameterized by tritium balance",
        "with \\(G=R+\\Delta I+L\\) enforced at the output level",
        "Random partitions measure interpolation within one release-mechanism family",
        "The corresponding source-map p95 error is 3.2\\%",
        "When the release mechanism is withheld, the same feature family gives a 281\\% p95 error",
        "Target-mechanism pilot histories supply the missing release-mechanism information",
        "Together they reduce the withheld-mechanism source-map p95 to about 2.5\\%",
        "Without pilot histories, the severe release-mechanism case remains above 5\\% source error",
        "At \\(K=16\\), the zone-\\(L_1\\) and closed-pathway p95 errors are 3.01\\% and 3.76\\%",
        "The final comparisons cover release-mechanism separation, source-structure selection, measurement-set completeness and component/system propagation",
        "The source-estimation histories are computed extensions from a calibrated target-bed transport response",
        "a larger pooled training set remains above 5\\% source error for a mismatched field",
        "A pooled library of non-matching 3D fields remains at 24.0--24.7\\% closed-pathway p95 error from 260 to 2080 histories",
        "Computed target-field histories first fall below 5\\% source error at 80 histories and improve to 3.53\\% at 520 histories",
        "A source/field distance comparison marks high-error new-mechanism samples before component propagation",
        "In the high-heterogeneity test, a severity-withheld MLP gives 18.6\\% worst joint p95 error",
        "Adding RTD metadata directly gives 19.0\\%",
        "direct spatial HT/HTO features give 15.8\\%",
        "The model using the calibrated 3D transfer response reaches the internal closure range below 0.1\\% when the fitted and evaluated histories share the same bed response",
        "Target-bed reference calibration lowers the error to 0.4\\%",
        "The neural layer takes the calibrated 3D response as a physical input to the source-multiplier fit",
        "The PDE-constrained model lowers the independent-case equation residual in all three rotations",
        "with gains of 1.005--1.146 and a worst field-RMSE ratio of 0.920",
        "Direct differentiation of the predicted concentration field still gives a 93.84\\% median source-multiplier error",
        "The inverse model reports the bounded source-layer estimate produced by this output layer",
        "Removing that non-outlet information sharply degrades representation prediction",
        "Pressure/RTD data and spatial HT/HTO histories measure the bed response directly and keep transport delay out of the material multiplier",
        "the measurement set and the 3D residence-time response",
        "a matched residence-time model with mismatched dry chemistry gives a 7.95\\% closed-pathway error",
        "Four spatial HT/HTO outlets with matched chemistry return the chemistry-resolved source estimate to the matched-bed error range",
        "A partly calibrated detector gives 4.40\\% release error",
        "leaves the HT/HTO cross-talk error at 3.93 percentage points",
        "Reference-pulse calibration lowers the projected closed-pathway p95 error to 2.65\\% at 2\\% detector noise",
        "mismatched chemistry gives a 7.44\\% inventory bias. Matched 3D HT/HTO chemistry brings the inventory bias to the matched-chemistry result",
        "HTO sampling-line memory adds a delayed measurement response",
        "Noisy reference-pulse calibration reduces the p95 bias to about 0.8\\%",
        "In the thermal source-release case, the field spans 63.9 K",
        "A mismatched hot-wall temperature basis gives 20.37\\%, and the temperature-resolved model gives 2.70\\%",
        "A separate 72 K transfer comparison gives 23.92\\% closed-pathway p95 error",
        "The error is 2.30\\% when endpoint temperatures reconstruct the gradient and approaches the calibrated-gradient result with the full temperature field",
        "Component and generic 288-component reduced tritium-balance calculations track how local source errors survive spatial summation",
        "In the generic 288-component reduced dynamic tritium-balance model",
        "Scaling a low-source linear response to high-source histories gives 17.09\\% closed-pathway p95 error and 86.50\\% retained-tail error",
        "A single nominal source reference leaves 16.34\\% closed-pathway p95 error",
        "The same measurement set separates compatible and incompatible source structures",
        "all incompatible source models exceed 5\\% source error",
        "Low, nominal and high source references plus retained inventory and TES reduce the closed-pathway error to 1.57\\% and preserve the retained-tail response",
        "A constant-power shortcut gives 218.85\\% closed-pathway p95 error",
        "In this dynamic source-history calculation, the specified power-history input with spatial HT/HTO outlets returns the closed-pathway and source-zone errors to the dynamic-response result",
        "raw gain/time errors give 84.34\\% closed-pathway p95 error",
        "Gain and time-axis calibration reduce it to 3.58\\%",
        "A model that reaches the local 5\\% source-error reference at \\(K=8\\) gives a 5.46\\% system-balance 95th-percentile error",
        "generic 288-component reduced dynamic tritium-balance model",
        "Module and generic 288-component reduced tritium-balance calculation",
        "Twelve module outlet histories show spatial spread compressed by a mixed outlet",
        "Across 300 withheld cases",
        "Sector-release localization reaches 96\\% top-sector accuracy",
        "5.8\\% tail-fraction p95 error",
        "16.7\\% of the time",
        "tail-fraction p95 error to about 0.6\\%",
        "96\\% accuracy",
        "the worst neural 95th-percentile errors fall within the system uncertainty range for both tail fraction and slow-tail sector amplitude",
        "local slow-tail deviations cancel in the system-total signal",
        "the system-total release curve",
        "Adding TES or integrated extraction-balance measurements gives about 2.6\\% module half-width",
        "The PINN performs the source calculation for this measurement set",
        "Biased estimates arise when spatial/species observations are incomplete or the transport response is mismatched",
        "Experimental implementation",
        "The Li$_4$SiO$_4$-derived chemistry terms are kept separate from the Li$_2$TiO$_3$ material constants",
        "This detector matrix is the first calibration level",
        "the same inverse can use a dynamic detector and sampling-line response",
        "about 2.6\\% system-balance 95th-percentile error at 2\\% balance noise",
        "The worst cross-shift 95th-percentile error is 4.1\\%",
        "The numerical and external data are grouped by physical function",
        "The inverse inputs are outlet/species histories, pressure/RTD- and spatial-HT/HTO-constrained residence-time responses, detector calibration and balance measurements",
        "The matched 3D-response rows use the same transport response for history generation and inversion",
        "The 3D fields used here represent fixed-particle packed-bed response",
        "coarse porous/effective residence-time fields",
        "The 5\\% source-error reference is used to compare measurement sets",
        "A 2\\%, 5\\% and 10\\% scale scan in the Supplement gives the same ordering",
        "Six- and eight-measurement sets reduce source uncertainty to 4.62\\% and 4.41\\%",
        "Method sequence for PINN-assisted delayed-source estimation in solid-breeder pebble beds",
        "Withheld release-mechanism, 3D-field and chemistry-field cases evaluate release-curve prediction, source-map estimation and HT/HTO consistency beyond the fitted histories",
        "Table~\\ref{tab:error_metrics} defines the source, transport and system metrics used below",
        "Table~\\ref{tab:kernel_uncertainty} directly perturbs the residence-time response",
        "The 20--25\\% slow/fast perturbations are severe residence-time mismatch cases beyond the expected post-calibration uncertainty range",
        "Field-calibrated residence-time-response perturbations set the practical source-error range at about 3--4\\%",
        "The Supplement lists the units and normalizations of the transport, source and balance variables",
    ]
    required_main_phrases = [
        "measurement-constrained physics-informed neural-network (PINN) inverse framework",
        "Training and evaluation are split by release mechanism, transport field and chemistry/thermal field",
        "reduces the closed-pathway multiplier error from 7.29\\% to 0.57\\%",
        "a 1D averaged inverse gives 32.07\\% closed-pathway error",
        "matched-bed benchmark range",
        "field-calibrated recovery scale of about 3--4\\%",
        "Severe unmodelled slow/fast residence-time mismatch drives errors above 60\\%",
        "Classical FVM baselines, negative-control PINN fits and physical holdout tests",
        "explicit source-output branch preserves delayed-source information",
        "generic 288-component reduced tritium-balance observability model",
        "sector release-tail histories reach 96.0\\%",
        "The inverse problem is organized as a sequence of measurement operators",
        "model-informed response constrained by measurable bed data",
        "Field-matched pilot histories are the smaller measured pilot set plus computed histories generated from the calibrated same-bed response",
        "The PINN provides a differentiable inverse representation for combining the measurement operators",
        "The 3D transport tests show that residence-time representation is the dominant source of delayed-pathway bias",
        "Supplying the matched 3D residence-time response and spatial HT/HTO histories removes this transport-source exchange",
        "the recoverable source scale is governed by the calibrated bed-response family",
        "The physical split tests separate interpolation from source recovery",
        "Field-matched histories, spatial/species observations and an explicit source-output branch are the information channels",
        "Generic 288-component tritium-balance observability calculation",
        "The results define a measurement rule for delayed tritium-source recovery",
        "Residence-time response is part of the source measurement",
        "The PINN contribution is the joint assimilation of multi-condition release histories",
        "a low system-level error is not sufficient if the local source or module inventory remains unresolved",
        "This work shows that delayed tritium-release source recovery in solid-breeder pebble beds is controlled by measurement observability",
        "delayed tritium release should be reported together with the residence-time response",
        "The Supplement lists the units and normalizations of the transport, source and balance variables",
    ]
    missing = [phrase for phrase in required_main_phrases if phrase not in main_text_expanded]
    add(rows, "main_scientific_spine", not missing, "all spine markers present" if not missing else "; ".join(missing))

    required_si_phrases = [
        "outlet-only source profiling",
        "A 6000-sample local FVM-sensitivity bootstrap",
        "maximum open and closed multiplier errors of 1.83\\% and 0.30\\%",
        "A computed 3D residence-time response and spatial HT/HTO observations return the source estimate to the matched-bed error range",
        "return the source estimate to the matched-bed error range when the same 3D response is used for history generation and inversion",
        "Explicit residence-time-response perturbations give the field-calibrated 3--4\\% source-error range reported in the main text",
        "generated_transport_variable_units_table.tex",
        "leaving out velocity gives 28.21\\%",
        "solid-inventory sensitivity calculation",
        "median PDE-derived total-source multiplier error of 93.84\\%",
        "at least 2.57-fold RMS improvement over the PDE-derived source",
        "The transfer-response neural comparison gives the corresponding feature-control result",
        "The model using the calibrated 3D transfer response falls below 0.1\\% when the fitted and evaluated histories share the same bed response",
        "Target-bed reference calibration lowers this error to 0.41\\%",
        "Sparse external curves are used quantitatively at their measured point density",
        "Digitization is used only to quantify the published curve trend",
        "generated_static_bed_operating_envelope_table.tex",
        "11.7-fold increase",
        "generated_minimum_measurement_set_table.tex",
        r"\tau_s\frac{d y_s}{dt}+y_s=y_s^{0}(t)",
        r"y_{s,i}=y_{s,i-1}+\frac{\Delta t_i}{\tau_s+\Delta t_i}",
        "Using those calibrated time constants reduces the closed-pathway p95 error to 0.81\\% over 400 pulse trials",
    ]
    missing = [phrase for phrase in required_si_phrases if phrase not in si_text]
    add(rows, "appendix_science_support", not missing, "all Appendix/SI support markers present" if not missing else "; ".join(missing))

    forbidden_current_terms = [
        "P5",
        "objective summary",
        "52-row",
        "template executor",
        "component-interface",
        "module/plant",
        "288-component system sensitivity tests",
        "seven-term objective",
        "P4",
        "C7 P4",
        "objective summary",
        "claim-to-protocol",
    ]
    hits = [term for term in forbidden_current_terms if term in text]
    add(rows, "old_project_jargon_removed", not hits, "no old project jargon in current entry files" if not hits else "; ".join(hits))

    stale_build_terms = [
        "build_si_latex_submission.py",
        "supplementary_information_draft_en.tex",
    ]
    script_text = read(ROOT / "scripts" / "reproduce_submission_science.sh")
    hits = [term for term in stale_build_terms if term in script_text]
    add(rows, "reproduce_science_uses_current_sources", not hits, "current script no longer generates old SI draft" if not hits else "; ".join(hits))

    makefile = read(ROOT / "Makefile")
    default_pdf_ok = (
        "pdf:" in makefile
        and "scripts/build_submission_pdf_local.sh" in makefile
        and "reproduce-science:" in makefile
        and "scripts/reproduce_submission_science.sh" in makefile
    )
    add(rows, "default_pdf_is_submission_line", default_pdf_ok, "default pdf target uses stable local submission build")

    main_wide_table_files = [
        MANUSCRIPT / "generated_observation_ambiguity_table.tex",
        MANUSCRIPT / "generated_main_pinn_validation_table.tex",
        MANUSCRIPT / "generated_main_minimum_measurement_set_table.tex",
    ]
    official_iop_layout_ok = (
        r"\documentclass[twocolumn]{iopjournal}" in main_text
        and r"\twocolumn[" in main_text
        and r"\setlength{\columnsep}{6mm}" in main_text
        and main_text.count(r"\begin{figure*}") >= 5
        and main_text.count(r"\begin{figure}") == 0
        and all(r"\begin{table*}" in read(path) for path in main_wide_table_files)
    )
    add(
        rows,
        "official_iop_nuclear_fusion_review_layout",
        official_iop_layout_ok,
        "official iopjournal/Nuclear Fusion two-column reading layout with full-width figures and tables"
        if official_iop_layout_ok
        else "official iopjournal/Nuclear Fusion two-column layout is inconsistent",
    )

    release_figure_summary = RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_summary.json"
    if release_figure_summary.exists():
        fig4 = json.loads(read(release_figure_summary))
        fig4_ok = (
            close(fig4.get("random_split_source_p95_pct", -1), 3.1985876745833095)
            and close(fig4.get("source_family_holdout_p95_pct", -1), 280.97556317153504)
            and close(fig4.get("ordinary_3d_feature_p95_pct", -1), 66.2691419306857)
            and close(fig4.get("matched_3d_operator_p95_pct", -1), 2.4745932939607074)
            and close(fig4.get("pooled_best_worst_closed_p95_pct", -1), 24.0450381991563)
            and int(fig4.get("target_first_passing_n_train", -1)) == 80
            and close(fig4.get("target_520_worst_closed_p95_pct", -1), 3.527118365332331)
        )
        fig4_detail = (
            "Fig. 4 training-data scaling and physical split numbers align"
            if fig4_ok
            else "Fig. 4 training-data scaling values drifted"
        )
    else:
        fig4_ok = False
        fig4_detail = "missing pinn_training_release_evidence_summary.json"
    add(rows, "main_pinn_release_figure_physical_split_numbers", fig4_ok, fig4_detail)

    headline_trace_json = RESULTS / "headline_claim_traceability.json"
    headline_trace_csv = RESULTS / "headline_claim_traceability.csv"
    if headline_trace_json.exists():
        headline_trace = json.loads(read(headline_trace_json))
        headline_trace_ok = (
            headline_trace.get("all_pass") is True
            and int(headline_trace.get("n_claims", -1)) == 8
            and int(headline_trace.get("n_pass", -1)) == 8
            and headline_trace_csv.exists()
            and "build_headline_claim_traceability.py" in reproduce_text
        )
        headline_trace_detail = (
            "eight abstract, main-figure and source-term output method numbers trace to current artifacts"
            if headline_trace_ok
            else "headline traceability summary drifted or is not reproduced"
        )
    else:
        headline_trace_ok = False
        headline_trace_detail = "missing headline_claim_traceability.json"
    add(rows, "headline_claim_traceability", headline_trace_ok, headline_trace_detail)

    outputs = [
        ROOT / "code" / "li_ceramic_bed_1d.py",
        ROOT / "code" / "build_headline_claim_traceability.py",
        ROOT / "code" / "build_3d_fvm_chemistry_transport_pressure.py",
        ROOT / "code" / "train_3d_fvm_informed_pinn_inverse.py",
        ROOT / "code" / "nf_pinn_figure_style.py",
        ROOT / "code" / "build_3d_pde_mlp_source_operator_recovery.py",
        ROOT / "code" / "build_3d_source_operator_head_recovery.py",
        ROOT / "code" / "build_pinn_training_validation_extrapolation_law.py",
        ROOT / "code" / "build_training_data_observability_summary.py",
        ROOT / "code" / "build_target_observability_phase_diagram.py",
        ROOT / "code" / "build_main_observation_ambiguity_table.py",
        ROOT / "code" / "build_main_pinn_validation_table.py",
        ROOT / "code" / "build_external_timescale_to_system_summary.py",
        MANUSCRIPT / "generated_main_pinn_validation_table.tex",
        RESULTS / "main_pinn_validation_table.csv",
        RESULTS
        / "external_timescale_to_system_accountancy"
        / "external_timescale_to_system_accountancy_summary.json",
        RESULTS / "fvm3d_trainable_pde_mlp_leave_one_case" / "fvm3d_trainable_pde_mlp_leave_one_case_summary.json",
        RESULTS / "fvm3d_pde_mlp_source_operator_recovery" / "fvm3d_pde_mlp_source_operator_recovery_summary.json",
        RESULTS / "fvm3d_source_operator_head_recovery" / "fvm3d_source_operator_head_recovery_summary.json",
        RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_summary.json",
        RESULTS
        / "whole_reactor_dynamic_tritium_observability"
        / "whole_reactor_dynamic_accountancy_main_summary.json",
        ROOT / "code" / "build_whole_reactor_accountancy_main_figure.py",
        FIGURES / "whole_reactor_accountancy_main_figure.pdf",
        RESULTS
        / "whole_reactor_dynamic_tritium_observability"
        / "whole_reactor_accountancy_main_figure_summary.json",
        ROOT / "code" / "build_dynamic_source_history_boundary.py",
        ROOT / "code" / "build_dynamic_source_accountancy_law.py",
        RESULTS
        / "dynamic_source_history_boundary"
        / "dynamic_source_history_boundary_summary.json",
        ROOT / "code" / "build_minimum_measurement_set_table.py",
        MANUSCRIPT / "generated_minimum_measurement_set_table.tex",
        MANUSCRIPT / "generated_main_minimum_measurement_set_table.tex",
        RESULTS / "minimum_measurement_set_table.csv",
        ROOT / "code" / "build_chemistry_thermal_identity_evidence.py",
        RESULTS / "pinn_training_scientific_map" / "pinn_training_scientific_map_summary.json",
        RESULTS / "pinn_training_validation_separation.csv",
        RESULTS
        / "pinn_training_validation_extrapolation_law"
        / "pinn_training_validation_extrapolation_law.csv",
        RESULTS
        / "pinn_training_validation_extrapolation_law"
        / "pinn_training_validation_extrapolation_law_summary.json",
        MANUSCRIPT / "generated_training_data_observability_summary.tex",
        MANUSCRIPT / "generated_training_data_observability_si.tex",
        MANUSCRIPT / "generated_target_observability_table.tex",
        RESULTS
        / "target_observability_phase_diagram"
        / "target_observability_phase_diagram_summary.json",
        RESULTS
        / "fvm3d_chemistry_transport_pressure"
        / "fvm3d_chemistry_transport_summary.json",
        RESULTS
        / "fvm3d_informed_pinn_inverse"
        / "fvm3d_informed_pinn_inverse_summary.json",
        RESULTS
        / "fvm3d_informed_pinn_inverse"
        / "fvm3d_informed_pinn_inverse_holdout_prediction_curves.csv",
        RESULTS
        / "real_curve_regularized_3d_inverse"
        / "real_curve_regularized_3d_inverse_summary.csv",
        RESULTS
        / "real_curve_regularized_3d_inverse"
        / "real_curve_regularized_3d_inverse_summary.json",
        RESULTS / "headline_claim_traceability.csv",
        RESULTS / "headline_claim_traceability.json",
        RESULTS / "chemistry_thermal_identity_evidence" / "chemistry_thermal_identity_evidence_summary.json",
        RESULTS / "li2_outlet_profile_likelihood" / "li2_outlet_profile_likelihood_surfaces.csv",
        NOTES / "pinn_training_release_evidence_cn.md",
    ]
    ok, detail = exists_all(outputs)
    add(rows, "core_generated_science_outputs", ok, detail)

    out_csv = RESULTS / "submission_consistency_check.csv"
    out_md = NOTES / "submission_consistency_check.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(row["status"] == "PASS" for row in rows)
    n_fail = sum(row["status"] == "FAIL" for row in rows)
    lines = [
        "# Compact submission consistency check",
        "",
        f"SUMMARY: PASS={n_pass} FAIL={n_fail}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"PASS={n_pass} FAIL={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
