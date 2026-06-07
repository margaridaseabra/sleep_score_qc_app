from pathlib import Path
from streamlit.components.v1 import declare_component

_component_path = Path(__file__).parent

_plotly_relayout_viewer = declare_component(
    "plotly_relayout_viewer",
    path=str(_component_path),
)


def plotly_relayout_viewer(fig_json, height=900, key=None):
    """
    Render a Plotly figure and return interaction events from JavaScript.

    Returned values can include:
      {"event_type": "relayout", "x0": ..., "x1": ...}
      {"event_type": "selection", "x0": ..., "x1": ...}
    """
    return _plotly_relayout_viewer(
        fig_json=fig_json,
        height=int(height),
        default=None,
        key=key,
    )
