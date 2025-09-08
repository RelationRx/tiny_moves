from typing import Literal

from pydantic import BaseModel


class Directionality(BaseModel):
    """A Direcitonality class to hold the directionality of effect of a hypothesis on BMD."""

    hgnc_symbol: str
    impact_on_bmd_if_antagonized: Literal["INCREASED", "DECREASED", "NO IMPACT", "UNKNOWN"]
    rationale: str
