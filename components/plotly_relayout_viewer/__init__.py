from pathlib import Path
from streamlit.components.v1 import declare_component

_component_path = Path(__file__).parent

_plotly_relayout_viewer = declare_component(
    "plotly_relayout_viewer",
    path=str(_component_path),
)


def plotly_relayout_viewer(fig_json_str, height=900, key=None):
    return _plotly_relayout_viewer(
        fig_json_str=fig_json_str,
        height=int(height),
        default=None,
        key=key,
    )
