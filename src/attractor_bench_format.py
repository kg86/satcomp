from dataclasses import dataclass
from dataclasses_json import dataclass_json
import datetime
from attractor import AttractorType


@dataclass_json
@dataclass
class AttractorExp:
    date: str
    status: str
    algo: str
    file_name: str
    file_len: int
    time_prep: float
    time_total: float
    sol_nvars: int
    sol_nhard: int
    sol_nsoft: int
    attractor_size: int
    attractor: AttractorType

    @classmethod
    def create(cls):
        return AttractorExp(
            str(datetime.datetime.now()),
            "",
            "",
            "",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            AttractorType([]),
        )
