"""
Run OSCAR v4 (customized mode) on gcages emissions, through OSCAR's IAMC CSV input
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import (
    update_index_levels_from_other_func,
)

from gcages.renaming import (
    convert_variable_name,
)

if TYPE_CHECKING:
    import xarray as xr

# TODO: This conversion involves different sources and assumptions from me
# It would be good to double check
OSCAR_INPUT_VARIABLES = pd.DataFrame(
    [
        # gcages variable, gcages unit, OSCAR variable, spc_halo, factor, OSCAR unit
        (
            "Emissions|CO2|Fossil",
            "Mt CO2/yr",
            "D_Eff",
            None,
            12 / 44 / 1000,
            "PgC yr-1",
        ),
        (
            "Emissions|CO2|Biosphere",
            "Mt CO2/yr",
            "D_Eluc",
            None,
            12 / 44 / 1000,
            "PgC yr-1",
        ),
        ("Emissions|CH4", "Mt CH4/yr", "D_Eant_CH4", None, 12 / 16, "TgC yr-1"),
        ("Emissions|N2O", "kt N2O/yr", "D_Eant_N2O", None, 28 / 44 / 1000, "TgN yr-1"),
        ("Emissions|CO", "Mt CO/yr", "D_Eant_CO", None, 12 / 28, "TgC yr-1"),
        ("Emissions|NOx", "Mt NO2/yr", "D_Eant_NOx", None, 14 / 46, "TgN yr-1"),
        ("Emissions|SOx", "Mt SO2/yr", "D_Eant_SO2", None, 32 / 64, "TgS yr-1"),
        ("Emissions|NH3", "Mt NH3/yr", "D_Eant_NH3", None, 14 / 17, "TgN yr-1"),
        ("Emissions|BC", "Mt BC/yr", "D_Eant_BC", None, 1.0, "TgC yr-1"),
        ("Emissions|OC", "Mt OC/yr", "D_Eant_OC", None, 1.0, "TgC yr-1"),
        ("Emissions|NMVOC", "Mt VOC/yr", "D_Eant_VOC", None, 1.0, "Tg yr-1"),
        # Halogens: values stay in kt (= Gg, what the model uses);
        # the SCI-lite library labels them "Tg yr-1" and OSCAR requires that label
        ("Emissions|C2F6", "kt C2F6/yr", "D_Eant_Xhalo", "C2F6", 1.0, "Tg yr-1"),
        ("Emissions|C3F8", "kt C3F8/yr", "D_Eant_Xhalo", "C3F8", 1.0, "Tg yr-1"),
        ("Emissions|C4F10", "kt C4F10/yr", "D_Eant_Xhalo", "C4F10", 1.0, "Tg yr-1"),
        ("Emissions|C5F12", "kt C5F12/yr", "D_Eant_Xhalo", "C5F12", 1.0, "Tg yr-1"),
        ("Emissions|C6F14", "kt C6F14/yr", "D_Eant_Xhalo", "C6F14", 1.0, "Tg yr-1"),
        ("Emissions|C7F16", "kt C7F16/yr", "D_Eant_Xhalo", "C7F16", 1.0, "Tg yr-1"),
        ("Emissions|C8F18", "kt C8F18/yr", "D_Eant_Xhalo", "C8F18", 1.0, "Tg yr-1"),
        ("Emissions|CCl4", "kt CCl4/yr", "D_Eant_Xhalo", "CCl4", 1.0, "Tg yr-1"),
        ("Emissions|CF4", "kt CF4/yr", "D_Eant_Xhalo", "CF4", 1.0, "Tg yr-1"),
        ("Emissions|CFC11", "kt CFC11/yr", "D_Eant_Xhalo", "CFC-11", 1.0, "Tg yr-1"),
        ("Emissions|CFC113", "kt CFC113/yr", "D_Eant_Xhalo", "CFC-113", 1.0, "Tg yr-1"),
        ("Emissions|CFC114", "kt CFC114/yr", "D_Eant_Xhalo", "CFC-114", 1.0, "Tg yr-1"),
        ("Emissions|CFC115", "kt CFC115/yr", "D_Eant_Xhalo", "CFC-115", 1.0, "Tg yr-1"),
        ("Emissions|CFC12", "kt CFC12/yr", "D_Eant_Xhalo", "CFC-12", 1.0, "Tg yr-1"),
        ("Emissions|CH2Cl2", "kt CH2Cl2/yr", "D_Eant_Xhalo", "CH2Cl2", 1.0, "Tg yr-1"),
        ("Emissions|CH3Br", "kt CH3Br/yr", "D_Eant_Xhalo", "CH3Br", 1.0, "Tg yr-1"),
        (
            "Emissions|CH3CCl3",
            "kt CH3CCl3/yr",
            "D_Eant_Xhalo",
            "CH3CCl3",
            1.0,
            "Tg yr-1",
        ),
        ("Emissions|CH3Cl", "kt CH3Cl/yr", "D_Eant_Xhalo", "CH3Cl", 1.0, "Tg yr-1"),
        ("Emissions|CHCl3", "kt CHCl3/yr", "D_Eant_Xhalo", "CHCl3", 1.0, "Tg yr-1"),
        (
            "Emissions|HCFC141b",
            "kt HCFC141b/yr",
            "D_Eant_Xhalo",
            "HCFC-141b",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HCFC142b",
            "kt HCFC142b/yr",
            "D_Eant_Xhalo",
            "HCFC-142b",
            1.0,
            "Tg yr-1",
        ),
        ("Emissions|HCFC22", "kt HCFC22/yr", "D_Eant_Xhalo", "HCFC-22", 1.0, "Tg yr-1"),
        ("Emissions|HFC125", "kt HFC125/yr", "D_Eant_Xhalo", "HFC-125", 1.0, "Tg yr-1"),
        (
            "Emissions|HFC134a",
            "kt HFC134a/yr",
            "D_Eant_Xhalo",
            "HFC-134a",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HFC143a",
            "kt HFC143a/yr",
            "D_Eant_Xhalo",
            "HFC-143a",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HFC152a",
            "kt HFC152a/yr",
            "D_Eant_Xhalo",
            "HFC-152a",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HFC227ea",
            "kt HFC227ea/yr",
            "D_Eant_Xhalo",
            "HFC-227ea",
            1.0,
            "Tg yr-1",
        ),
        ("Emissions|HFC23", "kt HFC23/yr", "D_Eant_Xhalo", "HFC-23", 1.0, "Tg yr-1"),
        (
            "Emissions|HFC236fa",
            "kt HFC236fa/yr",
            "D_Eant_Xhalo",
            "HFC-236fa",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HFC245fa",
            "kt HFC245fa/yr",
            "D_Eant_Xhalo",
            "HFC-245fa",
            1.0,
            "Tg yr-1",
        ),
        ("Emissions|HFC32", "kt HFC32/yr", "D_Eant_Xhalo", "HFC-32", 1.0, "Tg yr-1"),
        (
            "Emissions|HFC365mfc",
            "kt HFC365mfc/yr",
            "D_Eant_Xhalo",
            "HFC-365mfc",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|HFC4310mee",
            "kt HFC4310/yr",
            "D_Eant_Xhalo",
            "HFC-43-10mee",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|Halon1202",
            "kt Halon1202/yr",
            "D_Eant_Xhalo",
            "Halon-1202",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|Halon1211",
            "kt Halon1211/yr",
            "D_Eant_Xhalo",
            "Halon-1211",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|Halon1301",
            "kt Halon1301/yr",
            "D_Eant_Xhalo",
            "Halon-1301",
            1.0,
            "Tg yr-1",
        ),
        (
            "Emissions|Halon2402",
            "kt Halon2402/yr",
            "D_Eant_Xhalo",
            "Halon-2402",
            1.0,
            "Tg yr-1",
        ),
        ("Emissions|NF3", "kt NF3/yr", "D_Eant_Xhalo", "NF3", 1.0, "Tg yr-1"),
        ("Emissions|SF6", "kt SF6/yr", "D_Eant_Xhalo", "SF6", 1.0, "Tg yr-1"),
        ("Emissions|SO2F2", "kt SO2F2/yr", "D_Eant_Xhalo", "SO2F2", 1.0, "Tg yr-1"),
        ("Emissions|cC4F8", "kt cC4F8/yr", "D_Eant_Xhalo", "c-C4F8", 1.0, "Tg yr-1"),
    ],
    columns=["gcages", "gcages_unit", "oscar", "spc_halo", "factor", "oscar_unit"],
)
"""gcages to OSCAR SCI-lite input (OSCAR value = gcages value x factor)"""

# Not passed to OSCAR
NOT_SUPPORTED = {
    # D_Eluc is computed by OSCAR, not a forcing: passing it NaNs the whole ensemble
    # "Emissions|CO2|Biosphere": "land-use CO2 comes from baseline_forcing",
    # TODO: check, it should be solved with v4-beta3
    "Emissions|Halon1202": "not in OSCAR's registry",
}

OSCAR_OUTPUT_VARIABLES = pd.DataFrame(
    [
        ("D_Tg", "Surface Air Temperature Change", "K"),
        ("ERF", "Effective Radiative Forcing", "W/m^2"),
        ("ERF_CO2", "Effective Radiative Forcing|CO2", "W/m^2"),
        ("ERF_CH4", "Effective Radiative Forcing|CH4", "W/m^2"),
        ("ERF_N2O", "Effective Radiative Forcing|N2O", "W/m^2"),
        ("ERF_aer", "Effective Radiative Forcing|Aerosols", "W/m^2"),
        ("ERF_O3", "Effective Radiative Forcing|Ozone", "W/m^2"),
        ("CO2", "Atmospheric Concentrations|CO2", "ppm"),
        ("CH4", "Atmospheric Concentrations|CH4", "ppb"),
        ("N2O", "Atmospheric Concentrations|N2O", "ppb"),
        ("d_OHC", "Heat Uptake", "W/m^2"),
    ],
    columns=["oscar", "gcages", "unit"],
)
"""OSCAR to gcages output name"""


def prepare_emissions_for_oscar(
    indf: pd.DataFrame,
    scen_labels: dict[tuple[str, str], str],
    preindustrial: xr.Dataset,
) -> xr.Dataset:
    """
    Convert gcages emissions to OSCAR SCI input.

    OSCAR takes non-CO2 emissions as the change from pre-industrial
    (`D_Eant_X = total - Eant_X_pi`), fossil and land-use CO2 as they are.
    Variables not supplied are taken by OSCAR from its baseline scenario.

    Parameters
    ----------
    indf
        Input data

    scen_labels
        OSCAR scenario label for each (model, scenario)

    preindustrial
        OSCAR library dataset holding the pre-industrial emissions
        (`forcing_hist.nc` of the SCI-lite library)

    Returns
    -------
    :
        Dataset to pass as `forcing` to `oscar.run`
    """
    import xarray as xr  # noqa: PLC0415

    table = OSCAR_INPUT_VARIABLES.set_index("gcages")

    # Every (variable, unit) must be in the table, otherwise the factor is wrong
    supplied = pd.MultiIndex.from_frame(indf.index.to_frame()[["variable", "unit"]])
    known = pd.MultiIndex.from_frame(OSCAR_INPUT_VARIABLES[["gcages", "gcages_unit"]])
    unexpected = supplied.unique().difference(known)
    if not unexpected.empty:
        msg = f"No OSCAR input for (variable, unit): {unexpected.to_list()}"
        raise KeyError(msg)

    emms = indf.mul(table["factor"], axis=0, level="variable")
    emms = update_index_levels_from_other_func(
        emms,
        {
            "scen": (("model", "scenario"), scen_labels),
            "oscar": ("variable", table["oscar"].to_dict()),
            "spc_halo": ("variable", table["spc_halo"].to_dict()),
        },
    )
    emms.columns = emms.columns.astype(int).rename("year")

    pi = preindustrial.reset_coords(drop=True)  # drop OSCAR's scalar `scen` coord
    units = dict(zip(table["oscar"], table["oscar_unit"]))

    out = {}
    for oscar_var, df in emms.groupby(level="oscar"):
        dims = ["spc_halo", "scen"] if oscar_var == "D_Eant_Xhalo" else ["scen"]
        da = xr.DataArray.from_series(
            df.droplevel(df.index.names.difference(dims)).stack()
        )
        if oscar_var.startswith("D_Eant_"):
            # e.g. D_Eant_CH4 - Eant_CH4_pi (halogens aligned on spc_halo)
            da = da - pi[f"{oscar_var[2:]}_pi"]
        out[oscar_var] = da.transpose(*dims, "year").assign_attrs(
            units=units[oscar_var]
        )

    return xr.Dataset(out)


def oscar_results_to_df(
    results: xr.Dataset,
    scen_labels: dict[tuple[str, str], str],
    oscar_variables: list[str],
) -> pd.DataFrame:
    """
    Convert OSCAR results to gcages timeseries, one row per ensemble member

    OSCAR's shared `Historical` run is joined onto each scenario.

    Parameters
    ----------
    results
        OSCAR results

    scen_labels
        Scenario labels for each (model, scenario)

    oscar_variables
        OSCAR variables to convert

    Returns
    -------
    :
        DataFrame needed for OSCAR
    """
    oscar_hist_end = int(results.attrs["historical_end"])
    hist = results.sel(scen="Historical", year=slice(None, oscar_hist_end))
    scens = results.sel(year=slice(oscar_hist_end + 1, None))
    output = OSCAR_OUTPUT_VARIABLES.set_index("oscar")

    out = []
    for (model, scenario), label in scen_labels.items():
        for var in oscar_variables:
            hist_df = cast(pd.DataFrame, hist[var].to_pandas())
            scen_df = cast(pd.DataFrame, scens[var].sel(scen=label).to_pandas())
            df = pd.concat([hist_df, scen_df]).T.rename_axis("run_id")
            name, unit = output.at[var, "gcages"], output.at[var, "unit"]
            out.append(
                pd.concat(
                    {(model, scenario, name, unit): df},
                    names=["model", "scenario", "variable", "unit"],
                )
            )

    res = pd.concat(out)
    res.columns = res.columns.astype(int)
    res = res.assign(climate_model="OSCARv4-beta3", region="World").set_index(
        ["climate_model", "region"], append=True
    )

    res = res.reorder_levels(
        ["climate_model", "model", "scenario", "region", "run_id", "unit", "variable"]
    )

    return res


@define
class OSCARSCMRunner:
    """OSCAR v4 runner (customized mode, IAMC CSV input, full 500-member ensemble)"""

    output_variables: tuple[str, ...] = ("Surface Air Temperature Change",)
    """
    Variables to include in the output
    """
    # TODO: Double check that's what we want.
    baseline_forcing: str = "scen7-VL"  # every driver we do not supply, incl. land use
    """
    OSCAR library scenario that supplies every input
    we do not provide (incl. `ERF_luc`).
    """

    output_file: Path | None = None
    """
    NetCDF file to save OSCAR's raw results to (not saved if `None`)
    """

    overwrite_output: bool = False
    """
    Replace `output_file` if it already exists
    """

    def __call__(self, emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Run OSCAR on `emissions` (gcages naming convention)

        Parameters
        ----------
        emissions
            Emissions to run

        Returns
        -------
        :
            Raw results from the simple climate model
        """
        import oscar  # type: ignore # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        oscar_variables = [
            convert_variable_name(v, "gcages", "oscar", OSCAR_OUTPUT_VARIABLES)  # type: ignore[arg-type]
            for v in self.output_variables
        ]

        # OSCAR has one scenario dimension but no model.
        # To not avoid duplication the scenario names
        # are rewritten as {model} | {scenario}.
        model_scenarios = emissions.index.droplevel(
            emissions.index.names.difference(["model", "scenario"])  # type: ignore # pandas-stubs out of date
        ).unique()
        scen_labels = {(m, s): f"{m} | {s}" for m, s in model_scenarios}

        # Pre-industrial emissions, subtracted from the D_Eant inputs
        library = oscar.download(mode="advanced", option="SCI-lite")
        preindustrial = xr.load_dataset(library / "forcing_hist.nc")

        results = oscar.run(
            mode="advanced",
            option="SCI-lite",
            variables=oscar_variables,
            forcing=prepare_emissions_for_oscar(emissions, scen_labels, preindustrial),
            baseline_forcing=self.baseline_forcing,
            # OSCAR raises if the emissions go beyond its last year (2100)
            projection_end_year=int(max(emissions.columns)),
            output=self.output_file,
            overwrite=self.overwrite_output,
        )

        res = oscar_results_to_df(results, scen_labels, oscar_variables)
        if res.isnull().any().any():
            msg = "NaN in OSCAR output"
            raise AssertionError(msg)

        return res
