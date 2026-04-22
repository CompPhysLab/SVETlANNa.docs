import svetlanna as sv
import torch

from svetlanna.visualization.widgets import _ElementsIterator
from svetlanna.visualization.widgets import generate_structure_html

from svetlanna.networks import SimpleReservoir, LinearAutoencoder
from svetlanna.networks import ConvLayer4F, ConvDiffNetwork4F

from pathlib import Path
from jinja2 import Environment
from jinja2 import FileSystemLoader


def render_template(template_name: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
    template = env.get_template(template_name)
    return template.render(**context)


def write_text(path: Path, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


########################################################################
# Show structure widgets
########################################################################


Nx = Ny = 128
sim_params = sv.SimulationParameters(
    x=torch.linspace(-1, 1, Nx),
    y=torch.linspace(-1, 1, Ny),
    wavelength=0.1,
)


simple_reservoir = SimpleReservoir(
    nonlinear_element=sv.elements.NonlinearElement(
        simulation_parameters=sim_params,
        response_function=lambda x: x**2,
    ),
    delay_element=sv.elements.FreeSpace(
        simulation_parameters=sim_params, distance=0.2, method="AS"
    ),
    feedback_gain=0.5,
    input_gain=0.5,
    delay=3,
)


linear_autoencoder = LinearAutoencoder(
    encoder_elements=(
        sv.elements.FreeSpace(
            simulation_parameters=sim_params, distance=0.1, method="AS"
        ),
        sv.elements.ThinLens(simulation_parameters=sim_params, focal_length=0.1),
        sv.elements.FreeSpace(
            simulation_parameters=sim_params, distance=0.1, method="AS"
        ),
    ),
    decoder_elements=(
        sv.elements.FreeSpace(
            simulation_parameters=sim_params, distance=0.1, method="AS"
        ),
    ),
)


conv_layer_4f = ConvLayer4F(
    simulation_parameters=sim_params,
    focal_length=0.1,
    conv_diffractive_mask=torch.rand(sim_params.axis_sizes(("y", "x"))),
)

conv_diff_network_4f = ConvDiffNetwork4F(
    simulation_parameters=sim_params,
    network_elements=(
        sv.elements.FreeSpace(
            simulation_parameters=sim_params, distance=0.1, method="AS"
        ),
        sv.elements.ThinLens(simulation_parameters=sim_params, focal_length=0.1),
        sv.elements.FreeSpace(
            simulation_parameters=sim_params, distance=0.1, method="AS"
        ),
    ),
    focal_length=0.1,
    conv_diffractive_mask=torch.rand(sim_params.axis_sizes(("y", "x"))),
)


setups = (
    ("reservoir", simple_reservoir),
    ("autoencoder", linear_autoencoder),
    ("conv4f", conv_layer_4f),
    ("conv4f", conv_diff_network_4f),
)


for name_md, setup in setups:
    structure_html = generate_structure_html(
        _ElementsIterator(setup, directory="").tree
    )
    structure_html_open = structure_html.replace("<details>", "<details open>")

    path = Path(__file__).parent / name_md
    path.mkdir(exist_ok=True)
    write_text(
        path / f"show_structure_{setup.__class__.__name__}.html",
        render_template(
            "show_structure.html.j2",
            {"structure_html": structure_html_open},
        ),
    )
