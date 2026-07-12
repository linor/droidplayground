import math
import torch

class PhaseModulator:
    def __init__(
        self,
        time_step: float,
        num_envs: int,
        device: str,
        frequency: float = 0.5,
    ):
        self._time_step = time_step
        self._frequency = frequency
        self.num_envs = num_envs
        self.device = device

        self._phase = torch.zeros(
            num_envs,
            1,
            dtype=torch.float32,
            device=device,
        )

    def reset(
        self,
        env_ids: torch.Tensor,
        deterministic: bool = False,
    ) -> None:
        if deterministic:
            self._phase[env_ids] = 0.0
        else:
            self._phase[env_ids] = torch.empty(
                len(env_ids),
                1,
                device=self.device,
            ).uniform_(0.0, math.tau)

    def compute(self) -> torch.Tensor:
        self._phase.add_(
            math.tau * self._frequency * self._time_step
        )
        self._phase.remainder_(math.tau)

        return self._phase

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    @property
    def frequency(self) -> float:
        return self._frequency