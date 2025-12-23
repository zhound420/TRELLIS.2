
import gradio as gr
from app import demo as app
import os

_docs = {'Unified3D': {'description': 'Creates a component allows users to upload or view 3D Model files (.obj, .glb, .stl, .gltf, .splat, or .ply).\n', 'members': {'__init__': {'value': {'type': 'str | Callable | None', 'default': 'value = None', 'description': 'path to (.obj, .glb, .stl, .gltf, .splat, or .ply) file to show in model3D viewer. If a function is provided, the function will be called each time the app loads to set the initial value of this component.'}, 'display_mode': {'type': "Literal['solid', 'point_cloud', 'wireframe'] | None", 'default': 'value = None', 'description': 'the display mode of the 3D model in the scene. Can be "solid" (which renders the model as a solid object), "point_cloud", or "wireframe". For .splat, or .ply files, this parameter is ignored, as those files can only be rendered as solid objects.'}, 'clear_color': {'type': 'tuple[float, float, float, float] | None', 'default': 'value = None', 'description': 'background color of scene, should be a tuple of 4 floats between 0 and 1 representing RGBA values.'}, 'camera_position': {'type': 'tuple[int | float | None, int | float | None, int | float | None]', 'default': 'value = (None, None, None)', 'description': 'initial camera position of scene, provided as a tuple of `(alpha, beta, radius)`. Each value is optional. If provided, `alpha` and `beta` should be in degrees reflecting the angular position along the longitudinal and latitudinal axes, respectively. Radius corresponds to the distance from the center of the object to the camera.'}, 'zoom_speed': {'type': 'float', 'default': 'value = 1', 'description': 'the speed of zooming in and out of the scene when the cursor wheel is rotated or when screen is pinched on a mobile device. Should be a positive float, increase this value to make zooming faster, decrease to make it slower. Affects the wheelPrecision property of the camera.'}, 'pan_speed': {'type': 'float', 'default': 'value = 1', 'description': 'the speed of panning the scene when the cursor is dragged or when the screen is dragged on a mobile device. Should be a positive float, increase this value to make panning faster, decrease to make it slower. Affects the panSensibility property of the camera.'}, 'height': {'type': 'int | str | None', 'default': 'value = None', 'description': 'The height of the model3D component, specified in pixels if a number is passed, or in CSS units if a string is passed.'}, 'label': {'type': 'str | I18nData | None', 'default': 'value = None', 'description': 'the label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.'}, 'show_label': {'type': 'bool | None', 'default': 'value = None', 'description': 'if True, will display label.'}, 'every': {'type': 'Timer | float | None', 'default': 'value = None', 'description': 'Continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.'}, 'inputs': {'type': 'Component | Sequence[Component] | set[Component] | None', 'default': 'value = None', 'description': 'Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.'}, 'container': {'type': 'bool', 'default': 'value = True', 'description': 'If True, will place the component in a container - providing some extra padding around the border.'}, 'scale': {'type': 'int | None', 'default': 'value = None', 'description': 'relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.'}, 'min_width': {'type': 'int', 'default': 'value = 160', 'description': 'minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.'}, 'interactive': {'type': 'bool | None', 'default': 'value = None', 'description': 'if True, will allow users to upload a file; if False, can only be used to display files. If not provided, this is inferred based on whether the component is used as an input or output.'}, 'visible': {'type': "bool | Literal['hidden']", 'default': 'value = True', 'description': 'If False, component will be hidden. If "hidden", component will be visually hidden and not take up space in the layout but still exist in the DOM'}, 'elem_id': {'type': 'str | None', 'default': 'value = None', 'description': 'An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'elem_classes': {'type': 'list[str] | str | None', 'default': 'value = None', 'description': 'An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'render': {'type': 'bool', 'default': 'value = True', 'description': 'If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.'}, 'key': {'type': 'int | str | tuple[int | str, ...] | None', 'default': 'value = None', 'description': "in a gr.render, Components with the same key across re-renders are treated as the same component, not a new component. Properties set in 'preserved_by_key' are not reset across a re-render."}, 'preserved_by_key': {'type': 'list[str] | str | None', 'default': 'value = "value"', 'description': "A list of parameters from this component's constructor. Inside a gr.render() function, if a component is re-rendered with the same key, these (and only these) parameters will be preserved in the UI (if they have been changed by the user or an event listener) instead of re-rendered based on the values provided during constructor."}}, 'postprocess': {'value': {'type': 'str| pathlib.Path| None', 'description': 'Expects function to return a {str} or {pathlib.Path} filepath of type (.obj, .glb, .stl, or .gltf)'}}, 'preprocess': {'return': {'type': 'str| None', 'description': 'Passes the uploaded file as a {str} filepath to the function.'}, 'value': None}}, 'events': {'change': {'type': None, 'default': None, 'description': 'Triggered when the value of the Unified3D changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input.'}, 'upload': {'type': None, 'default': None, 'description': 'This listener is triggered when the user uploads a file into the Unified3D.'}, 'edit': {'type': None, 'default': None, 'description': 'This listener is triggered when the user edits the Unified3D (e.g. image) using the built-in editor.'}, 'clear': {'type': None, 'default': None, 'description': 'This listener is triggered when the user clears the Unified3D using the clear button for the component.'}}}, '__meta__': {'additional_interfaces': {}, 'user_fn_refs': {'Unified3D': []}}}

abs_path = os.path.join(os.path.dirname(__file__), "css.css")

with gr.Blocks(
    css=abs_path,
    theme=gr.themes.Default(
        font_mono=[
            gr.themes.GoogleFont("Inconsolata"),
            "monospace",
        ],
    ),
) as demo:
    gr.Markdown(
"""
# `gradio_unified3d`

<div style="display: flex; gap: 7px;">
<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%200.0.1%20-%20orange">  
</div>

Python library for easily interacting with trained machine learning models
""", elem_classes=["md-custom"], header_links=True)
    app.render()
    gr.Markdown(
"""
## Installation

```bash
pip install gradio_unified3d
```

## Usage

```python

import gradio as gr
from gradio_unified3d import Unified3D


with gr.Blocks() as demo:
    Unified3D()


if __name__ == "__main__":
    demo.launch()

```
""", elem_classes=["md-custom"], header_links=True)


    gr.Markdown("""
## `Unified3D`

### Initialization
""", elem_classes=["md-custom"], header_links=True)

    gr.ParamViewer(value=_docs["Unified3D"]["members"]["__init__"], linkify=[])


    gr.Markdown("### Events")
    gr.ParamViewer(value=_docs["Unified3D"]["events"], linkify=['Event'])




    gr.Markdown("""

### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As input:** Is passed, passes the uploaded file as a {str} filepath to the function.
- **As output:** Should return, expects function to return a {str} or {pathlib.Path} filepath of type (.obj, .glb, .stl, or .gltf).

 ```python
def predict(
    value: str| None
) -> str| pathlib.Path| None:
    return value
```
""", elem_classes=["md-custom", "Unified3D-user-fn"], header_links=True)




    demo.load(None, js=r"""function() {
    const refs = {};
    const user_fn_refs = {
          Unified3D: [], };
    requestAnimationFrame(() => {

        Object.entries(user_fn_refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}-user-fn`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })

        Object.entries(refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })
    })
}

""")

demo.launch()
