import copy
import numpy as np
import pandas as pd
import streamlit as st

from src.model_inputs import load_inputs, save_inputs
from src.zones_model import load_zones, save_zones, default_zone_template
from src.calculations import (
    orientation,
    interpolate_ks,
    total_fracturing_factor,
    rock_mass_fracturing_factor,
    k_porosity,
    k_dri_bilinear,
    critical_cutter_thrust,
    penetration_coefficient,
)

import altair as alt


# ------------------------------------------------------------
# HELPER: COMPUTE ALL OUTPUTS FOR ONE ZONE
# ------------------------------------------------------------
def compute_zone_outputs(zone: dict, inputs: dict) -> dict:
    """Compute all model outputs for one zone based on current inputs."""

    rpm = float(inputs["RPM"]["Mean"])
    thrust_MB = float(inputs["Thrust_MB"]["Mean"])
    tbm_diameter = float(inputs["TBM_diameter"]["Mean"])
    cutters = float(inputs["Cutters"]["Mean"])

    # --- Orientation for 3 joint sets ---
    o1 = orientation(zone["tunnel_direction"], zone["set1"]["strike"], zone["set1"]["dip"])
    o2 = orientation(zone["tunnel_direction"], zone["set2"]["strike"], zone["set2"]["dip"])
    o3 = orientation(zone["tunnel_direction"], zone["set3"]["strike"], zone["set3"]["dip"])

    # --- Fracturing factors ---
    ks1 = interpolate_ks(zone["set1"]["Fr_mean"], o1)
    ks2 = interpolate_ks(zone["set2"]["Fr_mean"], o2)
    ks3 = interpolate_ks(zone["set3"]["Fr_mean"], o3)

    ks_tot = total_fracturing_factor(
        ks1,
        ks2,
        ks3,
        zone["set1"]["Fr_mean"],
        zone["set2"]["Fr_mean"],
        zone["set3"]["Fr_mean"],
    )
    ks_rm = rock_mass_fracturing_factor(ks_tot)
    kpor = k_porosity(zone["Porosity"]["Mean"])
    kdri = k_dri_bilinear(ks_rm, zone["DRI"]["Mean"])

    # Equivalent fracturing
    k_equiv = ks_rm * kdri * kpor

    # Cutter diameter factor
    k_d = 1.0

    # Cutter spacing (mm)
    if cutters > 0:
        spacing = tbm_diameter * 1000.0 / (2.0 * cutters)
    else:
        spacing = np.nan

    if not np.isnan(spacing):
        k_a_raw = 1.0 + (-0.1 / 15.0) * (spacing - 70.0)
        k_a = min(k_a_raw, 1.0667)
    else:
        k_a = np.nan

    # Equivalent thrust
    if not np.isnan(k_a):
        M_ekv = thrust_MB * k_d * k_a
    else:
        M_ekv = np.nan

    # Critical cutter thrust (lookup)
    M1 = critical_cutter_thrust(k_equiv)

    # Penetration coefficient
    b_val = penetration_coefficient(k_equiv)

    # Basic penetration and net penetration
    if (
        M_ekv is None
        or M1 is None
        or isinstance(M_ekv, str)
        or isinstance(M1, str)
        or np.isnan(M_ekv)
        or np.isnan(M1)
        or M_ekv <= 0
        or M1 <= 0
        or rpm <= 0
    ):
        i0 = np.nan
        net_penetration = 0.0
    else:
        i0 = (M_ekv / M1) ** b_val
        net_penetration = i0 * rpm * 60.0 / 1000.0  # m/h

    # Boring time (h/m)
    if net_penetration > 0:
        boring_time = 1.0 / net_penetration
    else:
        boring_time = np.inf

    # Stroke length (m)
    stroke_length = float(inputs["Stroke_length_ls"]["Mean"])

    # Time related to segment installation
    if cutters > 0:
        time_per_segment_installation = tbm_diameter * 1000.0 / (2.0 * cutters)
    else:
        time_per_segment_installation = np.nan

    if stroke_length > 0 and not np.isnan(time_per_segment_installation):
        segment_installation = time_per_segment_installation / (60.0 * stroke_length)
    else:
        segment_installation = np.nan

    # Cutter change etc.
    time_per_changed_cutter = float(inputs["tc"]["Mean"])
    cutter_ring_life = 3.4853
    repair_tbm = float(inputs["TTBM"]["Mean"])
    repair_backup = float(inputs["Tback"]["Mean"])
    other_time = float(inputs["Tm"]["Mean"])

    if net_penetration > 0 and cutter_ring_life > 0:
        cutter_time = time_per_changed_cutter / (60.0 * cutter_ring_life * net_penetration)
    else:
        cutter_time = np.nan

    # Utilization
    terms = [boring_time, segment_installation, cutter_time, repair_tbm, repair_backup, other_time]

    if boring_time == np.inf or any(np.isnan(t) for t in terms):
        utilization = 0.0
    else:
        denom = sum(terms)
        utilization = boring_time / denom if denom > 0 else 0.0

    effective_hours = float(inputs["Effective_hours_Te"]["Mean"])
    daily_adv = net_penetration * utilization * effective_hours

    return {
        # Metadata
        "Zone name": zone["zone_name"],
        "Tunnel direction": zone["tunnel_direction"],
        "Length (m)": zone["length_m"],

        # Orientation & fracturing
        "Orientation Set1": o1,
        "Orientation Set2": o2,
        "Orientation Set3": o3,
        "Ks Set1": ks1,
        "Ks Set2": ks2,
        "Ks Set3": ks3,
        "Ks Total": ks_tot,
        "Ks Rock Mass": ks_rm,
        "k_porosity": kpor,
        "k_dri": kdri,
        "Equivalent fracturing": k_equiv,

        # Thrust & penetration
        "k_d (cutter diameter)": k_d,
        "k_a (cutter spacing)": k_a,
        "Equivalent thrust": M_ekv,
        "Critical cutter thrust (M1)": M1,
        "Penetration coefficient (b)": b_val,
        "Basic penetration (i0)": i0,
        "Net penetration (m/h)": net_penetration,

        # Time components
        "Boring time": boring_time,
        "Stroke length (m)": stroke_length,
        "Time per segment installation": time_per_segment_installation,
        "Segment installation time": segment_installation,
        "Time per changed cutter": time_per_changed_cutter,
        "Cutter ring life": cutter_ring_life,
        "Cutter time": cutter_time,
        "Repair and service of TBM": repair_tbm,
        "Repair and service of backup": repair_backup,
        "Other time consumption": other_time,
        "Utilization": utilization,
        "Effective working hours per day": effective_hours,

        # Result
        "Daily advancement (m/day)": daily_adv,
    }


# ------------------------------------------------------------
# HELPER: allowed range for one parameter (target analysis)
# ------------------------------------------------------------

def compute_allowed_range_for_param(
    param_name: str,
    selected_zone: dict,
    base_inputs: dict,
    current_values: dict,
    target_output_name: str,
    target_min: float,
    target_max: float,
    samples: int = 40,
):
    """
    Finner ca. intervall [x_min, x_max] for én input-parameter der valgt output
    holder seg innenfor [target_min, target_max], gitt at alle andre parametre
    står på 'current_values'.

    Søket gjøres i et intervall tilsvarende slider-området, ikke bare LB/UB.
    Returnerer (None, None) hvis ingen verdi i området gir output i målområdet.
    """

    # Bruk samme logikk som i TAB 5 for slider-range
    base_mean = float(base_inputs[param_name]["Mean"])

    if base_mean > 0:
        search_min = 0.0
        search_max = base_mean * 2.0
    elif base_mean < 0:
        search_min = base_mean * 2.0
        search_max = 0.0
    else:
        search_min = -1.0
        search_max = 1.0

    if search_max <= search_min:
        return None, None

    xs = np.linspace(search_min, search_max, samples)
    allowed_xs = []

    for x in xs:
        # Kopi av inputs
        mod_inputs = copy.deepcopy(base_inputs)

        # Sett alle parametere til nåværende slider-verdier
        for p, val in current_values.items():
            mod_inputs[p]["Mean"] = float(val)

        # Variér bare den ene parameteren vi analyserer
        mod_inputs[param_name]["Mean"] = float(x)

        # Trygg beregning av output
        try:
            out = compute_zone_outputs(selected_zone, mod_inputs)
            y = out.get(target_output_name, None)
        except Exception:
            # Hvis modellen knekker for en rar kombinasjon (deling på 0 etc),
            # hopper vi bare over den verdien.
            continue

        # Filtrer bort ugyldige outputs
        if y is None or not isinstance(y, (int, float)):
            continue
        if np.isnan(y) or np.isinf(y):
            continue

        # Ligger output innenfor målområdet?
        if target_min <= y <= target_max:
            allowed_xs.append(x)

    if not allowed_xs:
        return None, None

    return float(min(allowed_xs)), float(max(allowed_xs))



st.set_page_config(page_title="NTNU TBM Tool", layout="wide") 
st.title("NTNU TBM Tunneling Model Prototype") 
tab_machine, tab_zones, tab_output, tab_analysis, tab_target, tab_cutter, tab_perf, tab_time, tab_rock_perf, tab_profile = st.tabs( [ "Machine Input", "Zones Input", "Outputs", "Sensitivity analysis", "Target analysis", "Cutter life & consumption", "TBM performance envelope", "Time budget & utilization", "Rock vs performance", "Longitudinal profile", ] )


# ------------------------------------------------------------
# TAB 1: MACHINE PARAMETERS
# ------------------------------------------------------------
with tab_machine:
    st.header("Machine parameters (LB / Mean / UB)")

    inputs = load_inputs()
    updated = False

    for var_name, values in inputs.items():
        st.subheader(var_name)

        col1, col2, col3 = st.columns(3)
        lb = col1.number_input(f"{var_name} – LB", value=float(values["LB"]))
        mean = col2.number_input(f"{var_name} – Mean", value=float(values["Mean"]))
        ub = col3.number_input(f"{var_name} – UB", value=float(values["UB"]))

        if (lb, mean, ub) != (values["LB"], values["Mean"], values["UB"]):
            inputs[var_name] = {"LB": lb, "Mean": mean, "UB": ub}
            updated = True

    if st.button("💾 Save machine parameters"):
        save_inputs(inputs)
        st.success("Saved!")

    if updated:
        save_inputs(inputs)


# ------------------------------------------------------------
# TAB 2: ZONES INPUT
# ------------------------------------------------------------
with tab_zones:
    st.header("Zone configuration")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    num_zones = st.number_input(
        "Number of zones",
        min_value=1,
        max_value=50,
        value=len(zones) if len(zones) > 0 else 1,
    )

    while len(zones) < num_zones:
        zones.append(default_zone_template())
    while len(zones) > num_zones:
        zones.pop()

    st.divider()

    for i, zone in enumerate(zones):
        st.subheader(f"Zone {i+1}")

        zone["zone_name"] = st.text_input(f"Zone name {i+1}", value=zone["zone_name"])
        zone["rock_domain"] = st.text_input(f"Rock domain {i+1}", value=zone["rock_domain"])

        colA, colB = st.columns(2)
        zone["chainage_from"] = colA.text_input(
            f"Chainage from {i+1}", value=zone["chainage_from"]
        )
        zone["chainage_to"] = colB.text_input(
            f"Chainage to {i+1}", value=zone["chainage_to"]
        )

        zone["sd"] = st.text_input(f"SD {i+1}", value=zone["sd"])
        zone["station"] = st.text_input(f"Station {i+1}", value=zone["station"])

        st.markdown("### Rock parameters")
        for rock_key in ["DRI", "CLI", "Q", "Porosity"]:
            st.markdown(f"**{rock_key}**")
            c1, c2, c3 = st.columns(3)
            zone[rock_key]["Mean"] = c1.number_input(
                f"{rock_key} Mean {i+1}", value=float(zone[rock_key]["Mean"])
            )
            zone[rock_key]["LB"] = c2.number_input(
                f"{rock_key} LB {i+1}", value=float(zone[rock_key]["LB"])
            )
            zone[rock_key]["UB"] = c3.number_input(
                f"{rock_key} UB {i+1}", value=float(zone[rock_key]["UB"])
            )

        zone["tunnel_direction"] = st.number_input(
            f"Tunnel direction {i+1}",
            value=float(zone["tunnel_direction"]),
        )

        st.markdown("### Joint sets (strike, dip, Fr class)")
        for set_key in ["set1", "set2", "set3"]:
            st.markdown(f"**{set_key}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            zone[set_key]["strike"] = c1.number_input(
                f"{set_key} strike {i+1}", value=float(zone[set_key]["strike"])
            )
            zone[set_key]["dip"] = c2.number_input(
                f"{set_key} dip {i+1}", value=float(zone[set_key]["dip"])
            )
            zone[set_key]["Fr_mean"] = c3.text_input(
                f"{set_key} Fr_mean {i+1}", value=zone[set_key]["Fr_mean"]
            )
            zone[set_key]["Fr_LB"] = c4.text_input(
                f"{set_key} Fr_LB {i+1}", value=zone[set_key]["Fr_LB"]
            )
            zone[set_key]["Fr_UB"] = c5.text_input(
                f"{set_key} Fr_UB {i+1}", value=zone[set_key]["Fr_UB"]
            )

        zone["length_m"] = st.number_input(
            f"Length [m] zone {i+1}", value=float(zone["length_m"])
        )

        st.divider()

    if st.button("💾 Save zones"):
        save_zones({"zones": zones})
        st.success("Zones saved!")


# ------------------------------------------------------------
# TAB 3: OUTPUTS (PER ZONE, VERTICAL)
# ------------------------------------------------------------
with tab_output:
    st.header("Zone outputs")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined.")
    else:
        inputs = load_inputs()

        keys_to_show = [
            "Zone name",
            "Orientation Set1",
            "Orientation Set2",
            "Orientation Set3",
            "Ks Total",
            "Ks Rock Mass",
            "Equivalent fracturing",
            "Net penetration (m/h)",
            "Utilization",
            "Daily advancement (m/day)",
            "Length (m)",
        ]

        for idx, z in enumerate(zones):
            out = compute_zone_outputs(z, inputs)
            rows = []
            for key in keys_to_show:
                rows.append({"Parameter": key, "Value": out.get(key, "")})

            df_zone = pd.DataFrame(rows)
            df_zone["Value"] = df_zone["Value"].astype(str)

            zone_title = out.get("Zone name") or f"Zone {idx+1}"
            st.subheader(f"Zone: {zone_title}")
            st.dataframe(df_zone, width="stretch")

# ------------------------------------------------------------
# TAB 4: SENSITIVITY ANALYSIS (multi-output, with colors)
# ------------------------------------------------------------
with tab_analysis:
    st.header("Sensitivity analysis of input parameters")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()

        # Select zone to analyse
        zone_labels = [
            f"{i+1}: {z['zone_name']}" if z.get("zone_name") else f"Zone {i+1}"
            for i, z in enumerate(zones)
        ]
        selected_zone_label = st.selectbox("Select zone for analysis", zone_labels, index=0)
        selected_index = zone_labels.index(selected_zone_label)
        selected_zone = zones[selected_index]

        # Baseline outputs for selected zone
        baseline_outputs = compute_zone_outputs(selected_zone, inputs)

        # Only numeric outputs
        numeric_output_keys = [
            k for k, v in baseline_outputs.items() if isinstance(v, (int, float))
        ]

        # Default outputs: try these first
        default_outputs = [
            name for name in ["Net penetration (m/h)", "Daily advancement (m/day)"]
            if name in numeric_output_keys
        ]
        if not default_outputs and numeric_output_keys:
            default_outputs = [numeric_output_keys[0]]

        selected_outputs = st.multiselect(
            "Select output(s) as function of input",
            numeric_output_keys,
            default=default_outputs,
        )

        if not selected_outputs:
            st.warning("Select at least one output to run the analysis.")
            st.stop()

        baseline_values = {out: baseline_outputs[out] for out in selected_outputs}

        st.subheader("Baseline values for selected outputs")
        baseline_df = pd.DataFrame(
            [{"Output": out, "Baseline value": baseline_values[out]} for out in selected_outputs]
        )
        st.dataframe(baseline_df, width="stretch")

        # Input parameters that can be varied
        param_names = list(inputs.keys())
        default_params = [p for p in ["RPM", "Thrust_MB", "TBM_diameter", "Cutters"] if p in param_names]

        selected_params = st.multiselect(
            "Select one or more input parameters to vary",
            param_names,
            default=default_params,
        )

        col_range, col_steps = st.columns(2)
        pct_max = col_range.slider(
            "Maximum variation (± %)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Steps will be created between - and + this percentage.",
        )

        n_steps = col_steps.slider(
            "Number of steps between - and +",
            min_value=3,
            max_value=11,
            value=5,
            step=2,
            help="Should be an odd number to include 0 % (baseline).",
        )

        st.caption(
            "Note: For rock classes (Sf1, Sf2, ...) you should ideally move one class up/down, "
            "not use percentage change. This tab currently only handles continuous machine "
            "parameters (RPM, Thrust_MB, etc.)."
        )

        if st.button("Run detailed sensitivity analysis"):
            if not selected_params:
                st.warning("Select at least one input parameter.")
            else:
                # Grid of % changes, e.g. [-20, -10, 0, 10, 20]
                pct_grid = np.linspace(-pct_max, pct_max, n_steps)

                detail_records = []

                for p in selected_params:
                    base_val = float(inputs[p]["Mean"])

                    for pct_change in pct_grid:
                        factor = 1.0 + pct_change / 100.0
                        new_val = base_val * factor

                        # Copy of inputs with only one parameter changed
                        mod_inputs = copy.deepcopy(inputs)
                        mod_inputs[p]["Mean"] = new_val

                        out = compute_zone_outputs(selected_zone, mod_inputs)

                        for out_name in selected_outputs:
                            val = out[out_name]
                            base_out = baseline_values[out_name]

                            if base_out != 0:
                                delta_pct = (val - base_out) / base_out * 100.0
                            else:
                                delta_pct = np.nan

                            detail_records.append(
                                {
                                    "Parameter": p,
                                    "Output": out_name,
                                    "delta_input_pct": pct_change,
                                    "input_value": new_val,
                                    "output_value": val,
                                    "delta_output_pct": delta_pct,
                                }
                            )

                detail_df = pd.DataFrame(detail_records)

                if detail_df.empty:
                    st.warning("No rows in sensitivity analysis – check that you have selected parameters/outputs.")
                    st.stop()

                # Ensure numeric types
                num_cols_detail = [
                    "delta_input_pct",
                    "input_value",
                    "output_value",
                    "delta_output_pct",
                ]
                for c in num_cols_detail:
                    detail_df[c] = pd.to_numeric(detail_df[c], errors="coerce")

                st.subheader("Detailed sensitivity matrix")
                st.dataframe(detail_df, width="stretch")

                # -------------------------
                # Summary matrix per parameter & output
                # -------------------------
                summary_rows = []
                col_minus = f"Δ output ({-pct_max} %) [ % ]"
                col_plus = f"Δ output (+{pct_max} %) [ % ]"
                col_sens = "Sensitivity index [Δout% / Δin%]"

                for p in selected_params:
                    sub_p = detail_df[detail_df["Parameter"] == p].copy()

                    for out_name in selected_outputs:
                        sub = sub_p[sub_p["Output"] == out_name].copy()

                        sub_minus = sub[sub["delta_input_pct"] == -pct_max]
                        sub_plus = sub[sub["delta_input_pct"] == pct_max]

                        if sub_minus.empty or sub_plus.empty:
                            continue

                        d_minus = float(sub_minus["delta_output_pct"].iloc[0])
                        d_plus = float(sub_plus["delta_output_pct"].iloc[0])

                        base_out = baseline_values[out_name]

                        sens_index = (d_plus - d_minus) / (2.0 * pct_max) if pct_max != 0 else np.nan

                        summary_rows.append(
                            {
                                "Parameter": p,
                                "Output": out_name,
                                "Baseline input": float(inputs[p]["Mean"]),
                                "Baseline output": base_out,
                                col_minus: d_minus,
                                col_plus: d_plus,
                                col_sens: sens_index,
                            }
                        )

                summary_df = pd.DataFrame(summary_rows)

                if summary_df.empty:
                    st.warning("No summary data – check that the outputs actually change.")
                    st.stop()

                for c in ["Baseline input", "Baseline output", col_minus, col_plus, col_sens]:
                    if c in summary_df.columns:
                        summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

                st.subheader("Summary matrix (one row per parameter per output)")
                st.dataframe(summary_df, width="stretch")

                # -------------------------
                # Plot 1: curves (Δ output vs Δ input) for selected parameter
                # -------------------------
                st.subheader("Sensitivity curves (Δ output [%] vs Δ input [%])")

                import altair as alt

                param_to_plot = st.selectbox(
                    "Select parameter for curve plot",
                    selected_params,
                    index=0,
                )

                plot_sub = detail_df[detail_df["Parameter"] == param_to_plot].copy()
                plot_sub = plot_sub.dropna(subset=["delta_input_pct", "delta_output_pct"])

                if plot_sub.empty:
                    st.info("No data for selected parameter.")
                else:
                    plot_sub["output_name"] = plot_sub["Output"].astype(str)

                    line_chart = (
                        alt.Chart(plot_sub)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("delta_input_pct:Q", title="Δ input [%]"),
                            y=alt.Y("delta_output_pct:Q", title="Δ output [%]"),
                            color=alt.Color("output_name:N", title="Output"),
                            tooltip=[
                                "Parameter",
                                "output_name",
                                "delta_input_pct",
                                "output_value",
                                "delta_output_pct",
                            ],
                        )
                        .properties(width=700, height=400)
                    )

                    st.altair_chart(line_chart)

                # -------------------------
                # Plot 2: tornado – who influences most?
                # -------------------------
                st.subheader("Relative influence (tornado-style plot)")

                if not summary_df.empty:
                    tornado_df = summary_df[
                        ["Parameter", "Output", col_minus, col_plus]
                    ].copy()

                    tornado_df[col_minus] = pd.to_numeric(tornado_df[col_minus], errors="coerce")
                    tornado_df[col_plus] = pd.to_numeric(tornado_df[col_plus], errors="coerce")

                    tornado_df["max_delta_output"] = tornado_df[[col_minus, col_plus]].abs().max(axis=1)
                    tornado_df = tornado_df.replace([np.inf, -np.inf], np.nan)
                    tornado_df = tornado_df.dropna(subset=["max_delta_output"])

                    if tornado_df.empty:
                        st.info("No valid Δ output values (baseline may be 0 for all outputs).")
                    else:
                        tornado_df["param"] = tornado_df["Parameter"].astype(str)
                        tornado_df["out_name"] = tornado_df["Output"].astype(str)
                        tornado_df["label"] = tornado_df["param"] + " – " + tornado_df["out_name"]
                        tornado_df = tornado_df.sort_values("max_delta_output", ascending=True)

                        tornado_chart = (
                            alt.Chart(tornado_df)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "max_delta_output:Q",
                                    title="Maximum absolute Δ output [%]",
                                ),
                                y=alt.Y(
                                    "label:N",
                                    sort="-x",
                                    title="Parameter / Output",
                                ),
                                color=alt.Color("param:N", title="Parameter"),
                                tooltip=[
                                    "param",
                                    "out_name",
                                    "max_delta_output",
                                    col_minus,
                                    col_plus,
                                ],
                            )
                            .properties(width=700, height=400)
                        )

                        st.altair_chart(tornado_chart)

                # ----------------------------------------------------
                # NEW: Heatmap of input–output sensitivity
                # ----------------------------------------------------
                st.subheader("Input–Output sensitivity heatmap")

                # We already computed the sensitivity index per (Parameter, Output) in summary_df
                if summary_df.empty:
                    st.info("No sensitivity data available for heatmap.")
                else:
                    # Long-form dataframe with Parameter, Output, Sensitivity
                    heat_long = summary_df[["Parameter", "Output", col_sens]].copy()
                    heat_long = heat_long.rename(columns={col_sens: "Sensitivity"})
                    heat_long = heat_long.dropna(subset=["Sensitivity"])

                    if heat_long.empty:
                        st.info("No valid sensitivity values to show in the heatmap.")
                    else:
                        # For inspection also show matrix format
                        heat_matrix = heat_long.pivot(
                            index="Parameter", columns="Output", values="Sensitivity"
                        )
                        st.dataframe(heat_matrix, width="stretch")

                        # Altair heatmap
                        heat_chart = (
                            alt.Chart(heat_long)
                            .mark_rect()
                            .encode(
                                x=alt.X("Output:N", title="Output"),
                                y=alt.Y("Parameter:N", title="Input parameter"),
                                color=alt.Color(
                                    "Sensitivity:Q",
                                    title="Sensitivity [Δout% / Δin%]",
                                    scale=alt.Scale(scheme="redblue", domainMid=0),
                                ),
                                tooltip=[
                                    alt.Tooltip("Parameter:N", title="Parameter"),
                                    alt.Tooltip("Output:N", title="Output"),
                                    alt.Tooltip("Sensitivity:Q", title="Sensitivity", format=".3f"),
                                ],
                            )
                            .properties(width=700, height=400)
                        )

                        st.altair_chart(heat_chart, use_container_width=False)



# ------------------------------------------------------------
# TAB 5: TARGET ANALYSIS (target range on one output)
# ------------------------------------------------------------
with tab_target:
    st.header("Target analysis for a selected output")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        base_inputs = load_inputs()

        # Select zone
        zone_labels = [
            f"{i+1}: {z['zone_name']}" if z.get("zone_name") else f"Zone {i+1}"
            for i, z in enumerate(zones)
        ]
        selected_zone_label = st.selectbox("Select zone", zone_labels, index=0)
        selected_index = zone_labels.index(selected_zone_label)
        selected_zone = zones[selected_index]

        # Current input values (session_state or Mean)
        param_names = list(base_inputs.keys())
        current_values = {
            p: float(st.session_state.get(f"target_range_{p}", base_inputs[p]["Mean"]))
            for p in param_names
        }

        # Compute current outputs
        current_inputs = copy.deepcopy(base_inputs)
        for p, val in current_values.items():
            current_inputs[p]["Mean"] = val
        outputs_now = compute_zone_outputs(selected_zone, current_inputs)

        # Numeric outputs
        numeric_output_keys = [k for k, v in outputs_now.items() if isinstance(v, (int, float))]

        if not numeric_output_keys:
            st.warning("No numeric outputs available for analysis.")
            st.stop()

        default_out = (
            "Daily advancement (m/day)"
            if "Daily advancement (m/day)" in numeric_output_keys
            else numeric_output_keys[0]
        )

        # Remember previous output to reset min/max when it changes
        prev_output = st.session_state.get("target_output_select", default_out)

        target_output_name = st.selectbox(
            "Select output to define target range for",
            numeric_output_keys,
            index=numeric_output_keys.index(prev_output)
            if prev_output in numeric_output_keys
            else numeric_output_keys.index(default_out),
            key="target_output_select",
        )

        y_now = float(outputs_now[target_output_name])

        # Default: when output changes → set min/max to ±5 % around new value
        if (
            "target_min_val" not in st.session_state
            or "target_max_val" not in st.session_state
            or prev_output != target_output_name
        ):
            st.session_state["target_min_val"] = y_now * 0.95
            st.session_state["target_max_val"] = y_now * 1.05

        col_min, col_max = st.columns(2)
        with col_min:
            target_min = st.number_input(
                "Min limit",
                key="target_min_val",
            )
        with col_max:
            target_max = st.number_input(
                "Max limit",
                key="target_max_val",
            )

        if target_max <= target_min:
            st.warning("Max must be greater than min.")
            st.stop()

        inside = target_min <= y_now <= target_max

        # Big colored box with current output value
        color = "#0f993e" if inside else "#cc0000"
        st.markdown(
            f"""
            <div style="
                padding: 25px;
                border-radius: 10px;
                background-color: {color};
                color: white;
                font-size: 32px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;">
                {target_output_name}: {y_now:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"Target range: **[{target_min:.3f}, {target_max:.3f}]** – adjust inputs below."
        )

        st.divider()

        # ------------------------------------------------------------------
        # Find which input parameters actually influence the selected output
        # (5 % increase must create a noticeable change to be included)
        # ------------------------------------------------------------------
        influencing_params = []
        baseline_y = y_now
        influence_threshold = 1e-3  # relative change > 0.1 %

        for p in param_names:
            base_val = current_values[p]
            delta = 0.05 * abs(base_val) if base_val != 0 else 0.05

            if delta == 0:
                continue

            try:
                mod_inputs = copy.deepcopy(base_inputs)
                # set all to current values
                for q, v in current_values.items():
                    mod_inputs[q]["Mean"] = float(v)
                # bump only this one
                mod_inputs[p]["Mean"] = float(base_val + delta)

                out_up = compute_zone_outputs(selected_zone, mod_inputs)
                y_up = out_up.get(target_output_name, baseline_y)

                if not isinstance(y_up, (int, float)):
                    continue
                if np.isnan(y_up) or np.isinf(y_up):
                    continue

                if baseline_y != 0:
                    rel_diff = abs(y_up - baseline_y) / abs(baseline_y)
                else:
                    rel_diff = abs(y_up - baseline_y)

                if rel_diff > influence_threshold:
                    influencing_params.append(p)

            except Exception:
                continue

        if not influencing_params:
            st.info("No input parameters appear to influence the selected output in this region.")
            st.stop()

        # --------------------------
        # Compute allowed ranges (amin–amax) for influencing parameters only
        # --------------------------
        with st.spinner("Computing allowed intervals..."):
            allowed_ranges = {}
            for p in influencing_params:
                amin, amax = compute_allowed_range_for_param(
                    param_name=p,
                    selected_zone=selected_zone,
                    base_inputs=base_inputs,
                    current_values=current_values,
                    target_output_name=target_output_name,
                    target_min=target_min,
                    target_max=target_max,
                    samples=40,
                )
                allowed_ranges[p] = (amin, amax)

        st.subheader("Input parameters (that influence the selected output)")

        # --------------------------
        # Visual sliders – one row: name | slider | info
        # --------------------------
        for p in influencing_params:
            base_mean = float(base_inputs[p]["Mean"])

            # Slider range = ±200 % around base
            span_factor = 2.0
            slider_min = base_mean - abs(base_mean) * span_factor
            slider_max = base_mean + abs(base_mean) * span_factor

            if slider_min == slider_max:
                slider_min -= 1.0
                slider_max += 1.0

            current_val = float(
                st.session_state.get(f"target_range_{p}", base_mean)
            )
            current_val = max(slider_min, min(slider_max, current_val))

            span = slider_max - slider_min
            step = span / 200.0 if span != 0 else 0.01

            amin, amax = allowed_ranges.get(p, (None, None))

            col_name, col_slider, col_info = st.columns([1, 3, 2])

            # ---- NAME ----
            with col_name:
                st.markdown(f"### {p}")

            # ---- SLIDER + blue safe-zone overlay ----
            with col_slider:
                slider_val = st.slider(
                    f"{p}_slider",
                    min_value=float(slider_min),
                    max_value=float(slider_max),
                    value=float(current_val),
                    step=float(step),
                    key=f"target_range_{p}",
                    label_visibility="collapsed",
                )

                total_span = slider_max - slider_min

                if amin is not None and amax is not None and total_span > 0:
                    safe_min = max(amin, slider_min)
                    safe_max = min(amax, slider_max)

                    if safe_max > safe_min:
                        left_pct = (safe_min - slider_min) / total_span * 100
                        width_pct = (safe_max - slider_min) / total_span * 100 - left_pct
                    else:
                        left_pct = 0
                        width_pct = 0

                    st.markdown(
                        f"""
                        <div style="position: relative; height: 12px; margin-top: -10px;">
                            <div style="
                                position: absolute;
                                left: {left_pct}%;
                                width: {width_pct}%;
                                height: 12px;
                                background-color: #2c6bed;
                                border-radius: 4px;
                                opacity: 0.45;">
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="height: 12px; background-color: #e0e0e0;
                                    border-radius: 4px; opacity: 0.3;">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ---- INFO / WARNING ----
            with col_info:
                if amin is None or amax is None:
                    st.markdown(
                        f"<span style='color:#b30000;'>❌ No value gives output in the target range.</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"Allowed range: <b>{amin:.3g} → {amax:.3g}</b><br>"
                        f"Current: <b>{slider_val:.3g}</b>",
                        unsafe_allow_html=True,
                    )
                    if not (amin <= slider_val <= amax):
                        st.markdown(
                            f"<div style='padding:8px; background:#fff8dd; border-left:4px solid #ffcc00;'>"
                            f"⚠️ Outside safe range"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")


# ------------------------------------------------------------
# TAB 6: CUTTER LIFE & CONSUMPTION
# ------------------------------------------------------------
with tab_cutter:
    st.header("Cutter Life & Consumption")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()

        # --------------------------
        # Economic parameters
        # --------------------------
        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            cost_per_ring = st.number_input(
                "Cost per cutter ring [NOK]",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
            )
        with col_cost2:
            rings_per_change = st.number_input(
                "Rings replaced per change (avg.)",
                min_value=0.1,
                value=1.0,
                step=0.1,
            )

        cost_per_change = cost_per_ring * rings_per_change

        # --------------------------
        # Calculate cutter-related values per zone
        # --------------------------
        records = []

        for z in zones:
            out = compute_zone_outputs(z, inputs)

            zone_name = out.get("Zone name", "") or z.get("zone_name", "")
            length_m = float(out.get("Length (m)", z.get("length_m", 0.0)) or 0.0)

            net_pen = float(out.get("Net penetration (m/h)", 0.0) or 0.0)
            daily_adv = float(out.get("Daily advancement (m/day)", 0.0) or 0.0)
            cutter_life_h = float(out.get("Cutter ring life", 3.4853) or 3.4853)
            tc_min = float(out.get("Time per changed cutter", inputs["tc"]["Mean"]) or 0.0)

            meters_per_ring = np.nan
            changes_per_100m = np.nan
            changes_per_day = np.nan
            changes_zone = np.nan
            rings_zone = np.nan
            downtime_h_zone = np.nan
            downtime_days_zone = np.nan
            cost_zone = np.nan

            if net_pen > 0 and cutter_life_h > 0:
                # meters drilled per ring
                meters_per_ring = net_pen * cutter_life_h

                if meters_per_ring > 0:
                    changes_per_meter = 1.0 / meters_per_ring
                    changes_per_100m = changes_per_meter * 100.0
                    changes_per_day = changes_per_meter * daily_adv
                    changes_zone = changes_per_meter * length_m

                    rings_zone = changes_zone * rings_per_change

                    time_per_change_h = tc_min / 60.0
                    downtime_h_zone = changes_zone * time_per_change_h
                    downtime_days_zone = downtime_h_zone / 24.0

                    cost_zone = rings_zone * cost_per_ring

            records.append(
                {
                    "Zone": zone_name,
                    "Length (m)": length_m,
                    "Net penetration (m/h)": net_pen,
                    "Daily advancement (m/day)": daily_adv,
                    "Cutter life (h)": cutter_life_h,
                    "Meters per ring [m/ring]": meters_per_ring,
                    "Cutter changes per 100 m": changes_per_100m,
                    "Cutter changes per day": changes_per_day,
                    "Cutter changes (zone)": changes_zone,
                    "Ring consumption (zone)": rings_zone,
                    "Downtime from changes (h)": downtime_h_zone,
                    "Downtime from changes (days)": downtime_days_zone,
                    "Cutter cost (zone) [NOK]": cost_zone,
                }
            )

        df_cutter = pd.DataFrame(records)

        # Ensure numeric types
        num_cols = [
            "Length (m)",
            "Net penetration (m/h)",
            "Daily advancement (m/day)",
            "Cutter life (h)",
            "Meters per ring [m/ring]",
            "Cutter changes per 100 m",
            "Cutter changes per day",
            "Cutter changes (zone)",
            "Ring consumption (zone)",
            "Downtime from changes (h)",
            "Downtime from changes (days)",
            "Cutter cost (zone) [NOK]",
        ]
        for c in num_cols:
            df_cutter[c] = pd.to_numeric(df_cutter[c], errors="coerce")

        # --------------------------
        # Project-level key figures
        # --------------------------
        total_changes = df_cutter["Cutter changes (zone)"].sum(skipna=True)
        total_rings = df_cutter["Ring consumption (zone)"].sum(skipna=True)
        total_cost = df_cutter["Cutter cost (zone) [NOK]"].sum(skipna=True)
        total_downtime_days = df_cutter["Downtime from changes (days)"].sum(skipna=True)

        st.subheader("Key project figures")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total cutter changes", f"{total_changes:,.1f}")
        m2.metric("Total rings consumed", f"{total_rings:,.1f}")
        m3.metric("Total cutter cost [NOK]", f"{total_cost:,.0f}")
        m4.metric("Total downtime (days)", f"{total_downtime_days:,.2f}")

        st.divider()

        # --------------------------
        # Table per zone
        # --------------------------
        st.subheader("Cutter consumption per zone")
        st.dataframe(df_cutter, width="stretch")

        # --------------------------
        # Visualisations
        # --------------------------
        import altair as alt

        # 1) Cutter changes per 100 m
        st.subheader("Cutter changes per 100 m (per zone)")
        plot_df = df_cutter.dropna(subset=["Cutter changes per 100 m"])
        if not plot_df.empty:
            changes_chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("Zone:N", title="Zone"),
                    y=alt.Y("Cutter changes per 100 m:Q", title="Changes per 100 m"),
                    tooltip=[
                        "Zone",
                        "Cutter changes per 100 m",
                        "Cutter changes (zone)",
                    ],
                )
                .properties(width=700, height=350)
            )
            st.altair_chart(changes_chart)
        else:
            st.info("No valid data for cutter changes per 100 m.")

        # 2) Cutter cost per zone
        st.subheader("Cutter cost per zone")

        # Rename cost column to a simple name for plotting
        cost_df = df_cutter.dropna(subset=["Cutter cost (zone) [NOK]"]).copy()
        cost_df["CutterCost"] = pd.to_numeric(
            cost_df["Cutter cost (zone) [NOK]"], errors="coerce"
        )

        if not cost_df.empty:
            cost_chart = (
                alt.Chart(cost_df)
                .mark_bar()
                .encode(
                    x=alt.X("Zone:N", title="Zone"),
                    y=alt.Y("CutterCost:Q", title="Cost [NOK]"),
                    tooltip=[
                        "Zone",
                        alt.Tooltip("CutterCost:Q", title="Cost [NOK]", format=",.0f"),
                        "Ring consumption (zone)",
                        "Cutter changes (zone)",
                    ],
                )
                .properties(width=700, height=350)
            )
            st.altair_chart(cost_chart)
        else:
            st.info("No valid cutter cost data.")

# ------------------------------------------------------------
# TAB 7: TBM PERFORMANCE ENVELOPE (scatter)
# ------------------------------------------------------------
with tab_perf:
    st.header("TBM performance envelope")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()

        records = []
        for z in zones:
            out = compute_zone_outputs(z, inputs)
            zone_name = out.get("Zone name", "") or z.get("zone_name", "")

            records.append(
                {
                    "Zone": zone_name,
                    "Ks Rock Mass": out.get("Ks Rock Mass", np.nan),
                    "Equivalent fracturing": out.get("Equivalent fracturing", np.nan),
                    "Equivalent thrust": out.get("Equivalent thrust", np.nan),
                    "Net penetration (m/h)": out.get("Net penetration (m/h)", np.nan),
                    "Daily advancement (m/day)": out.get("Daily advancement (m/day)", np.nan),
                }
            )

        df_perf = pd.DataFrame(records)
        for c in [
            "Ks Rock Mass",
            "Equivalent fracturing",
            "Equivalent thrust",
            "Net penetration (m/h)",
            "Daily advancement (m/day)",
        ]:
            df_perf[c] = pd.to_numeric(df_perf[c], errors="coerce")

        x_options = {
            "Ks Rock Mass": "Ks Rock Mass",
            "Equivalent fracturing": "Equivalent fracturing",
            "Equivalent thrust": "Equivalent thrust",
        }
        y_options = {
            "Net penetration (m/h)": "Net penetration (m/h)",
            "Daily advancement (m/day)": "Daily advancement (m/day)",
        }

        col_x, col_y = st.columns(2)
        with col_x:
            x_label = st.selectbox("X-axis parameter", list(x_options.keys()), index=0)
        with col_y:
            y_label = st.selectbox("Y-axis parameter", list(y_options.keys()), index=0)

        x_col = x_options[x_label]
        y_col = y_options[y_label]

        plot_df = df_perf.dropna(subset=[x_col, y_col])

        if plot_df.empty:
            st.info("No valid data for the selected axes.")
        else:
            chart = (
                alt.Chart(plot_df)
                .mark_circle(size=120)
                .encode(
                    x=alt.X(f"{x_col}:Q", title=x_label, scale=alt.Scale(zero=False, nice=True)),
                    y=alt.Y(f"{y_col}:Q", title=y_label, scale=alt.Scale(zero=False, nice=True)),
                    color=alt.Color("Zone:N", title="Zone"),
                    tooltip=[
                        "Zone",
                        alt.Tooltip(x_col, title=x_label),
                        alt.Tooltip(y_col, title=y_label),
                    ],
                )
                .properties(width=800, height=450)
            )
            st.altair_chart(chart, use_container_width=False)


# ------------------------------------------------------------
# TAB 8: TIME BUDGET & UTILIZATION
# ------------------------------------------------------------
with tab_time:
    st.header("Time budget & utilization")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()
        records = []

        for z in zones:
            out = compute_zone_outputs(z, inputs)
            zone_name = out.get("Zone name", "") or z.get("zone_name", "")

            boring_time = out.get("Boring time", np.nan)
            seg = out.get("Segment installation time", np.nan)
            cutter_time = out.get("Cutter time", np.nan)
            repair_tbm = out.get("Repair and service of TBM", np.nan)
            repair_back = out.get("Repair and service of backup", np.nan)
            other_time = out.get("Other time consumption", np.nan)

            terms = [boring_time, seg, cutter_time, repair_tbm, repair_back, other_time]
            terms = [t if not np.isinf(t) else np.nan for t in terms]

            if any(np.isnan(t) for t in terms):
                total = np.nan
            else:
                total = sum(terms)

            if total and not np.isnan(total) and total > 0:
                share_boring = boring_time / total
                share_seg = seg / total
                share_cutter = cutter_time / total
                share_tbm = repair_tbm / total
                share_back = repair_back / total
                share_other = other_time / total
            else:
                share_boring = share_seg = share_cutter = share_tbm = share_back = share_other = np.nan

            records.append(
                {
                    "Zone": zone_name,
                    "Boring": share_boring,
                    "Segment installation": share_seg,
                    "Cutter-related": share_cutter,
                    "TBM repair": share_tbm,
                    "Backup repair": share_back,
                    "Other": share_other,
                    "Utilization": out.get("Utilization", np.nan),
                }
            )

        df_time = pd.DataFrame(records)

        st.subheader("Utilization per zone")
        util_df = df_time[["Zone", "Utilization"]].copy()
        st.dataframe(util_df, width="stretch")

        st.subheader("Time budget breakdown per zone")

        melt_cols = ["Boring", "Segment installation", "Cutter-related", "TBM repair", "Backup repair", "Other"]
        plot_df = df_time[["Zone"] + melt_cols].copy()
        for c in melt_cols:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

        plot_df = plot_df.melt(id_vars="Zone", value_vars=melt_cols, var_name="Category", value_name="Share")
        plot_df = plot_df.dropna(subset=["Share"])

        if plot_df.empty:
            st.info("No valid time budget data.")
        else:
            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("Zone:N", title="Zone"),
                    y=alt.Y("Share:Q", title="Share of cycle", stack="normalize"),
                    color=alt.Color("Category:N", title="Time category"),
                    tooltip=["Zone", "Category", alt.Tooltip("Share:Q", format=".2f")],
                )
                .properties(width=800, height=450)
            )
            st.altair_chart(chart, use_container_width=False)


# ------------------------------------------------------------
# TAB 9: ROCK VS PERFORMANCE – scatter dashboard
# ------------------------------------------------------------
with tab_rock_perf:
    st.header("Rock vs performance")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()

        # Local cost settings (only used if you choose cost on y-axis)
        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            cost_per_ring = st.number_input(
                "Cutter ring cost [NOK]",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
                key="rwp_cost_per_ring",
            )
        with col_cost2:
            rings_per_change = st.number_input(
                "Rings changed per intervention",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="rwp_rings_per_change",
            )

        records = []
        for z in zones:
            out = compute_zone_outputs(z, inputs)

            zone_name = out.get("Zone name", "") or z.get("zone_name", "")
            rock_domain = z.get("rock_domain", "")

            try:
                dri = float(z["DRI"]["Mean"])
            except Exception:
                dri = np.nan
            try:
                cli = float(z["CLI"]["Mean"])
            except Exception:
                cli = np.nan
            try:
                q_val = float(z["Q"]["Mean"])
            except Exception:
                q_val = np.nan
            try:
                por = float(z["Porosity"]["Mean"])
            except Exception:
                por = np.nan

            ks_rm = float(out.get("Ks Rock Mass", np.nan) or np.nan)
            k_equiv = float(out.get("Equivalent fracturing", np.nan) or np.nan)

            net_pen = float(out.get("Net penetration (m/h)", np.nan) or np.nan)
            daily_adv = float(out.get("Daily advancement (m/day)", np.nan) or np.nan)
            util = float(out.get("Utilization", np.nan) or np.nan)

            cutter_life_h = float(out.get("Cutter ring life", np.nan) or np.nan)

            changes_per_100m = np.nan
            cost_per_m = np.nan

            if (
                net_pen is not None
                and cutter_life_h is not None
                and net_pen > 0
                and cutter_life_h > 0
            ):
                meters_per_ring = net_pen * cutter_life_h  # [m/ring]
                if meters_per_ring > 0:
                    changes_per_meter = 1.0 / meters_per_ring
                    changes_per_100m = changes_per_meter * 100.0

                    rings_per_meter = changes_per_meter * rings_per_change
                    cost_per_m = rings_per_meter * cost_per_ring

            fr_set1 = z.get("set1", {}).get("Fr_mean", "")

            records.append(
                {
                    "Zone": zone_name,
                    "rock_domain": rock_domain,
                    "Fr_set1": fr_set1,
                    "DRI": dri,
                    "CLI": cli,
                    "Q": q_val,
                    "Porosity": por,
                    "Ks Rock Mass": ks_rm,
                    "Equivalent fracturing": k_equiv,
                    "Net penetration (m/h)": net_pen,
                    "Daily advancement (m/day)": daily_adv,
                    "Utilization": util,
                    "Cutter changes per 100 m": changes_per_100m,
                    "Cutter cost per meter [NOK/m]": cost_per_m,
                }
            )

        df_rwp = pd.DataFrame(records)

        numeric_cols = [
            "DRI",
            "CLI",
            "Q",
            "Porosity",
            "Ks Rock Mass",
            "Equivalent fracturing",
            "Net penetration (m/h)",
            "Daily advancement (m/day)",
            "Utilization",
            "Cutter changes per 100 m",
            "Cutter cost per meter [NOK/m]",
        ]
        for c in numeric_cols:
            df_rwp[c] = pd.to_numeric(df_rwp[c], errors="coerce")

        x_options = {
            "DRI": "DRI",
            "CLI": "CLI",
            "Q": "Q",
            "Porosity": "Porosity",
            "Ks Rock Mass": "Ks Rock Mass",
            "Equivalent fracturing": "Equivalent fracturing",
        }

        y_options = {
            "Net penetration (m/h)": "Net penetration (m/h)",
            "Daily advancement (m/day)": "Daily advancement (m/day)",
            "Utilization": "Utilization",
            "Cutter changes per 100 m": "Cutter changes per 100 m",
            "Cutter cost per meter [NOK/m]": "Cutter cost per meter [NOK/m]",
        }

        col_x, col_y = st.columns(2)
        with col_x:
            x_label = st.selectbox("Rock parameter (x-axis)", list(x_options.keys()), index=0)
        with col_y:
            y_label = st.selectbox(
                "Performance parameter (y-axis)",
                list(y_options.keys()),
                index=1,  # default: Daily advancement
            )

        x_col = x_options[x_label]
        y_col = y_options[y_label]

        color_choice = st.selectbox(
            "Color points by",
            ["Rock domain", "Fr-class (set1)", "Zone"],
            index=0,
        )
        if color_choice == "Rock domain":
            color_field = "rock_domain"
        elif color_choice == "Fr-class (set1)":
            color_field = "Fr_set1"
        else:
            color_field = "Zone"

        add_reg = st.checkbox("Add regression line", value=True)

        plot_df = df_rwp.dropna(subset=[x_col, y_col])

        if plot_df.empty:
            st.info("No valid data for the selected axes.")
        else:
            base = (
                alt.Chart(plot_df)
                .mark_circle(size=120)
                .encode(
                    x=alt.X(
                        f"{x_col}:Q",
                        title=x_label,
                        scale=alt.Scale(zero=False, nice=True),
                    ),
                    y=alt.Y(
                        f"{y_col}:Q",
                        title=y_label,
                        scale=alt.Scale(zero=False, nice=True),
                    ),
                    color=alt.Color(f"{color_field}:N", title=color_choice),
                    tooltip=[
                        "Zone",
                        "rock_domain",
                        "Fr_set1",
                        alt.Tooltip(x_col, title=x_label),
                        alt.Tooltip(y_col, title=y_label),
                        alt.Tooltip("DRI:Q", title="DRI"),
                        alt.Tooltip("Q:Q", title="Q"),
                        alt.Tooltip("Porosity:Q", title="Porosity"),
                        alt.Tooltip("Ks Rock Mass:Q", title="Ks Rock Mass"),
                        alt.Tooltip("Equivalent fracturing:Q", title="Equivalent fracturing"),
                    ],
                )
                .properties(width=800, height=450)
            )

            chart = base

            r2_text = ""
            if add_reg and plot_df[x_col].nunique() > 1:
                reg_line = (
                    base.transform_regression(
                        x_col,
                        y_col,
                        method="linear",
                        as_=[x_col, "y_fit"],
                    )
                    .mark_line(color="black", strokeDash=[4, 4])
                    .encode(y="y_fit:Q")
                )
                chart = base + reg_line

                x_vals = plot_df[x_col].to_numpy()
                y_vals = plot_df[y_col].to_numpy()
                try:
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    y_pred = coeffs[0] * x_vals + coeffs[1]
                    ss_res = np.sum((y_vals - y_pred) ** 2)
                    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    if not np.isnan(r2):
                        r2_text = f"Estimated R² (linear fit): {r2:.2f}"
                except Exception:
                    r2_text = ""

            st.altair_chart(chart, use_container_width=False)

            if r2_text:
                st.caption(r2_text)


# ------------------------------------------------------------
# TAB 10: LONGITUDINAL PROFILE
# ------------------------------------------------------------
with tab_profile:
    st.header("Longitudinal tunnel profile")

    zone_data = load_zones()
    zones = zone_data.get("zones", [])

    if len(zones) == 0:
        st.warning("No zones defined – please define zones first in the 'Zones Input' tab.")
    else:
        inputs = load_inputs()

        # simple helper: "0+244" -> 244
        def parse_chainage(ch_str: str) -> float:
            try:
                if "+" in ch_str:
                    a, b = ch_str.split("+")
                    return float(a) * 1000.0 + float(b)
                return float(ch_str)
            except Exception:
                return np.nan

        records = []
        for z in zones:
            out = compute_zone_outputs(z, inputs)

            zone_name = out.get("Zone name", "") or z.get("zone_name", "")
            rock_domain = z.get("rock_domain", "")

            ch_from = parse_chainage(z.get("chainage_from", "0"))
            ch_to = parse_chainage(z.get("chainage_to", "0"))
            ch_mid = (ch_from + ch_to) / 2.0 if not np.isnan(ch_from) and not np.isnan(ch_to) else np.nan

            daily_adv = out.get("Daily advancement (m/day)", np.nan)
            ks_rm = out.get("Ks Rock Mass", np.nan)
            k_equiv = out.get("Equivalent fracturing", np.nan)

            records.append(
                {
                    "Zone": zone_name,
                    "rock_domain": rock_domain,
                    "chainage_from_m": ch_from,
                    "chainage_to_m": ch_to,
                    "chainage_mid_m": ch_mid,
                    "Daily advancement (m/day)": daily_adv,
                    "Ks Rock Mass": ks_rm,
                    "Equivalent fracturing": k_equiv,
                }
            )

        df_prof = pd.DataFrame(records)
        df_prof = df_prof.dropna(subset=["chainage_from_m", "chainage_to_m"])

        if df_prof.empty:
            st.info("No valid chainage data to build a longitudinal profile.")
        else:
            st.subheader("Rock domains along chainage")

            rect_chart = (
                alt.Chart(df_prof)
                .mark_rect()
                .encode(
                    x=alt.X("chainage_from_m:Q", title="Chainage [m]"),
                    x2="chainage_to_m:Q",
                    y=alt.value(0),
                    color=alt.Color("rock_domain:N", title="Rock domain"),
                    tooltip=[
                        "Zone",
                        "rock_domain",
                        alt.Tooltip("chainage_from_m:Q", title="From [m]"),
                        alt.Tooltip("chainage_to_m:Q", title="To [m]"),
                    ],
                )
                .properties(width=800, height=80)
            )

            st.altair_chart(rect_chart, use_container_width=False)

            st.subheader("Daily advancement and fracturing along chainage")

            line_adv = (
                alt.Chart(df_prof)
                .mark_line(point=True)
                .encode(
                    x=alt.X("chainage_mid_m:Q", title="Chainage [m]"),
                    y=alt.Y("Daily advancement (m/day):Q", title="Daily advancement [m/day]"),
                    color=alt.value("#1f77b4"),
                    tooltip=[
                        "Zone",
                        "rock_domain",
                        alt.Tooltip("Daily advancement (m/day):Q", format=".2f"),
                    ],
                )
            )

            line_ks = (
                alt.Chart(df_prof)
                .mark_line(point=True)
                .encode(
                    x=alt.X("chainage_mid_m:Q", title="Chainage [m]"),
                    y=alt.Y("Ks Rock Mass:Q", title="Ks Rock Mass"),
                    color=alt.value("#ff7f0e"),
                    tooltip=[
                        "Zone",
                        "rock_domain",
                        alt.Tooltip("Ks Rock Mass:Q", format=".2f"),
                    ],
                )
            )

            combo = alt.layer(line_adv, line_ks).resolve_scale(
                y="independent"
            ).properties(width=800, height=400)

            st.altair_chart(combo, use_container_width=False)
