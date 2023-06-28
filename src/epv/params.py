from dataclasses import dataclass, field


@dataclass(frozen=True)
class EPV:
    seed: int = 42
    cols: list[str] = field(default_factory=lambda: [
        "shot_clock", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y", "p4_x", "p4_y", "p5_x", "p5_y", "p6_x", "p6_y",
        "p7_x", "p7_y", "p8_x", "p8_y", "p9_x", "p9_y", "p10_x", "p10_y", "b_x", "b_y", "b_z"
    ])
    test_size: float = 0.15
    valid_size: float = 0.20
