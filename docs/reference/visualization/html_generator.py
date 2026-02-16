import svetlanna as sv
import torch

from svetlanna.visualization.widgets import _ElementsIterator
from svetlanna.visualization.widgets import generate_structure_html
from svetlanna.visualization.widgets import show_specs
from svetlanna.visualization.widgets import show_stepwise_forward

from pathlib import Path
import json
import base64
from jinja2 import Environment
from jinja2 import FileSystemLoader

Nx = Ny = 128
sim_params = sv.SimulationParameters(
    x=torch.linspace(-1, 1, Nx),
    y=torch.linspace(-1, 1, Ny),
    wavelength=0.1,
)

setup = sv.LinearOpticalSetup(
    [
        sv.elements.RectangularAperture(sim_params, width=0.5, height=0.5),
        sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
        sv.elements.DiffractiveLayer(sim_params, mask=torch.rand(Ny, Nx), mask_norm=1),
        sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
    ]
)


def render_template(template_name: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
    template = env.get_template(template_name)
    return template.render(**context)


def write_text(path: Path, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def b64_json(value: dict) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


########################################################################
# Show structure widget
########################################################################

# here the internal functions are used
structure_html = generate_structure_html(_ElementsIterator(setup, directory="").tree)
structure_html_open = structure_html.replace("<details>", "<details open>")

write_text(
    Path(__file__).parent / "show_structure.html",
    render_template(
        "show_structure.html.j2",
        {"structure_html": structure_html_open},
    ),
)

########################################################################
# Show specs widget
########################################################################

specs_widget = show_specs(setup)

write_text(Path(__file__).parent / "show_specs_esm.js", specs_widget._esm)

specs_b64 = b64_json(specs_widget.elements)
specs_structure_html_open = specs_widget.structure_html.replace(
    "<details>", "<details open>"
)

write_text(
    Path(__file__).parent / "show_specs.html",
    render_template(
        "show_specs.html.j2",
        {
            "css": specs_widget._css,
            "esm_path": "./show_specs_esm.js",
            "elements_b64": specs_b64,
            "structure_html": specs_structure_html_open,
        },
    ),
)


########################################################################
# Show stepwise forward widget
########################################################################

show_stepwise_forward_widget = show_stepwise_forward(
    setup,
    input=sv.Wavefront.plane_wave(sim_params),
    simulation_parameters=sim_params,
    types_to_plot=("I", "phase", "Re"),
)

write_text(
    Path(__file__).parent / "show_stepwise_forward_esm.js",
    show_stepwise_forward_widget._esm,
)

stepwise_b64 = b64_json(show_stepwise_forward_widget.elements)
stepwise_structure_html_open = show_stepwise_forward_widget.structure_html.replace(
    "<details>", "<details open>"
)

write_text(
    Path(__file__).parent / "show_stepwise_forward.html",
    render_template(
        "show_stepwise_forward.html.j2",
        {
            "css": show_stepwise_forward_widget._css,
            "esm_path": "./show_stepwise_forward_esm.js",
            "elements_b64": stepwise_b64,
            "structure_html": stepwise_structure_html_open,
        },
    ),
)
