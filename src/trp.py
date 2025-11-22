from dataclasses import dataclass
import math

@dataclass
class TRPScheduler:
    base_lr: float
    alpha: float
    min_mult: float = 1e-6
    max_mult: float = 1.0

    def multiplier(self, I_struct_t: float) -> float:
        m = math.exp(-self.alpha * float(I_struct_t))
        return max(self.min_mult, min(self.max_mult, m))

    def lr(self, I_struct_t: float) -> float:
        return self.base_lr * self.multiplier(I_struct_t)
