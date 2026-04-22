import matplotlib.pyplot as plt
import torch
from svetlanna.elements.slm import QuantizerFromStepFunction
from svetlanna.elements.slm import one_step_cos, one_step_tanh
from pathlib import Path


N = 4
max_value = 2 * torch.pi

x = 2 * torch.pi * torch.linspace(0, 2, 1000)

functions = (
    (one_step_cos, [0, 1, 6]),
    (one_step_tanh, [0, 4, 10]),
)

for style in ["default", "dark_background"]:
    plt.style.use(style)

    for one_step_function, alphas in functions:
        f = QuantizerFromStepFunction(
            N=N,
            max_value=max_value,
            one_step_function=one_step_function,
        )

        plt.figure()

        for alpha in torch.tensor(alphas):
            plt.plot(x, f(x, alpha=alpha), label=f"alpha={alpha}")

        ax = plt.gca()
        ax.set_aspect("equal")

        def pi_formatter(value, _):
            multiple = torch.tensor(value) / torch.pi
            multiple = multiple.to(torch.get_default_dtype())
            if torch.isclose(multiple, torch.tensor(0.0)):
                return "0"
            if torch.isclose(multiple, torch.tensor(1.0)):
                return "$\\pi$"
            if torch.isclose(multiple, torch.tensor(-1.0)):
                return "$-\\pi$"
            return f"{multiple:.0f}$\\pi$"

        ax.xaxis.set_major_formatter(pi_formatter)
        ax.yaxis.set_major_formatter(pi_formatter)

        ax.set_xticks(torch.arange(0, 5) * torch.pi)
        ax.set_yticks(torch.arange(0, 3) * torch.pi)
        plt.title(
            f"{one_step_function.__name__}(x) with N={N} "
            f"and max_value={pi_formatter(max_value, None)}"
        )
        plt.xlabel('x')
        plt.legend()
        plt.savefig(
            Path(__file__).parent
            / f"one_step_function_{one_step_function.__name__}_{style}.jpg",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()
