"""
Tests of gcages.renaming for IAMC variables
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
    rename_variables,
)

cases_to_check_ar6_wg3 = pytest.mark.parametrize(
    "ar6_wg3_variable, gcages_variable",
    tuple(
        pytest.param(
            ar6_wg3_variable,
            gcages_variable,
            id=gcages_variable,
        )
        for ar6_wg3_variable, gcages_variable in (
            ("Emissions|BC", "Emissions|BC"),
            ("Emissions|PFC|C2F6", "Emissions|C2F6"),
            ("Emissions|PFC|C3F8", "Emissions|C3F8"),
            ("Emissions|PFC|C4F10", "Emissions|C4F10"),
            ("Emissions|PFC|C5F12", "Emissions|C5F12"),
            ("Emissions|PFC|C6F14", "Emissions|C6F14"),
            ("Emissions|PFC|C7F16", "Emissions|C7F16"),
            ("Emissions|PFC|C8F18", "Emissions|C8F18"),
            ("Emissions|PFC|CF4", "Emissions|CF4"),
            ("Emissions|CH4", "Emissions|CH4"),
            ("Emissions|CO", "Emissions|CO"),
            ("Emissions|CO2", "Emissions|CO2"),
            ("Emissions|CO2|AFOLU", "Emissions|CO2|Biosphere"),
            (
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CO2|Fossil",
            ),
            ("Emissions|HFC|HFC125", "Emissions|HFC125"),
            ("Emissions|HFC|HFC134a", "Emissions|HFC134a"),
            ("Emissions|HFC|HFC143a", "Emissions|HFC143a"),
            ("Emissions|HFC|HFC152a", "Emissions|HFC152a"),
            ("Emissions|HFC|HFC227ea", "Emissions|HFC227ea"),
            ("Emissions|HFC|HFC23", "Emissions|HFC23"),
            ("Emissions|HFC|HFC236fa", "Emissions|HFC236fa"),
            ("Emissions|HFC|HFC245ca", "Emissions|HFC245fa"),
            ("Emissions|HFC|HFC32", "Emissions|HFC32"),
            ("Emissions|HFC|HFC365mfc", "Emissions|HFC365mfc"),
            ("Emissions|HFC|HFC43-10", "Emissions|HFC4310mee"),
            ("Emissions|CCl4", "Emissions|CCl4"),
            ("Emissions|CFC11", "Emissions|CFC11"),
            ("Emissions|CFC113", "Emissions|CFC113"),
            ("Emissions|CFC114", "Emissions|CFC114"),
            ("Emissions|CFC115", "Emissions|CFC115"),
            ("Emissions|CFC12", "Emissions|CFC12"),
            ("Emissions|CH2Cl2", "Emissions|CH2Cl2"),
            ("Emissions|CH3Br", "Emissions|CH3Br"),
            ("Emissions|CH3CCl3", "Emissions|CH3CCl3"),
            ("Emissions|CH3Cl", "Emissions|CH3Cl"),
            ("Emissions|CHCl3", "Emissions|CHCl3"),
            ("Emissions|HCFC141b", "Emissions|HCFC141b"),
            ("Emissions|HCFC142b", "Emissions|HCFC142b"),
            ("Emissions|HCFC22", "Emissions|HCFC22"),
            ("Emissions|Halon1202", "Emissions|Halon1202"),
            ("Emissions|Halon1211", "Emissions|Halon1211"),
            ("Emissions|Halon1301", "Emissions|Halon1301"),
            ("Emissions|Halon2402", "Emissions|Halon2402"),
            ("Emissions|N2O", "Emissions|N2O"),
            ("Emissions|NF3", "Emissions|NF3"),
            ("Emissions|NH3", "Emissions|NH3"),
            ("Emissions|NOx", "Emissions|NOx"),
            ("Emissions|OC", "Emissions|OC"),
            ("Emissions|SF6", "Emissions|SF6"),
            ("Emissions|SO2F2", "Emissions|SO2F2"),
            ("Emissions|Sulfur", "Emissions|SOx"),
            ("Emissions|VOC", "Emissions|NMVOC"),
            ("Emissions|PFC|cC4F8", "Emissions|cC4F8"),
        )
    ),
)


@cases_to_check_ar6_wg3
def test_convert_ar6_wg3_variable_to_gcages(ar6_wg3_variable, gcages_variable):
    assert (
        convert_variable_name(
            ar6_wg3_variable,
            from_convention=SupportedNamingConventions.AR6_WG3,
            to_convention=SupportedNamingConventions.GCAGES,
        )
        == gcages_variable
    )


@cases_to_check_ar6_wg3
def test_convert_gcages_variable_to_ar6_wg3(ar6_wg3_variable, gcages_variable):
    assert (
        convert_variable_name(
            gcages_variable,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.AR6_WG3,
        )
        == ar6_wg3_variable
    )


@cases_to_check_ar6_wg3
def test_convert_variable_name_iamc_warns_and_works(ar6_wg3_variable, gcages_variable):
    with pytest.warns(FutureWarning, match="Use 'AR6_WG3' instead"):
        res = convert_variable_name(
            ar6_wg3_variable,
            from_convention=SupportedNamingConventions.IAMC,
            to_convention=SupportedNamingConventions.GCAGES,
        )

    assert res == gcages_variable


def test_rename_variables_iamc_warns_and_works():
    start = pd.DataFrame(
        [[1.0]],
        columns=[2020],
        index=pd.MultiIndex.from_tuples(
            [("Emissions|CF4", "Mt CF4/yr")], names=["variable", "unit"]
        ),
    )

    with pytest.warns(FutureWarning, match="Use 'AR6_WG3' instead"):
        res = rename_variables(
            start,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.IAMC,
        )

    assert res.index.get_level_values("variable").tolist() == ["Emissions|PFC|CF4"]


@cases_to_check_ar6_wg3
def test_ar6_wg3_does_not_warn(ar6_wg3_variable, gcages_variable):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)

        res = convert_variable_name(
            ar6_wg3_variable,
            from_convention=SupportedNamingConventions.AR6_WG3,
            to_convention=SupportedNamingConventions.GCAGES,
        )

    assert res == gcages_variable
