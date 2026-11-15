from dataclasses import dataclass,field
@dataclass
class Request:
    input_ids: list
    decisions: list
    labels: list | None = None
    profile: str = "accepted-single"

    @property
    def weight(self):
        return sum(d["loss_weight"] for d in self.decisions)

