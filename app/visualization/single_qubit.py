import logging

import numpy as np

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.qstate_representations import convert_state_vector_to_bloch_vector

logger = logging.getLogger(" [PLOTLY]")


def make_bloch_sphere(targets=None):

    u_angle = np.linspace(0, 2 * np.pi, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u_angle), np.sin(v_angle))
    y = np.outer(np.sin(u_angle), np.sin(v_angle))
    z = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))

    traces = []

    sphere = go.Surface(
        x=x,
        y=y,
        z=z,
        name="Sphere",
        opacity=0.2,
        colorscale=[[0, "#179c7d"],
                    [1, "#179c7d"]],
        surfacecolor=np.ones_like(z),
        showscale=False,
        contours=dict(
            x=dict(
                start=0,
                end=0.001,
                size=0.001,
                show=True,
                color="#006247"
            ),
            y=dict(
                start=0,
                end=0.001,
                size=0.001,
                show=True,
                color="#006247"
            ),
            z=dict(
                start=-1,
                end=1,
                size=0.25,
                show=True,
                color="#006247"
            )
        )
    )

    traces.append(sphere)

    coordinates = [[0, 0, 1.4], [0, 0, -1.4], [1.4, 0, 0], [-1.4, 0, 0]]
    annotation_labels = [r"|0⟩", r"|1⟩", r"|+⟩", r"|-⟩"]
    colors = 4 * ["black"]
    sizes = 4 * [12]

    if targets is not None:
        for n, t in enumerate(targets):
            coordinates.append(1.1 * t)
            annotation_labels.append(f'Ψ{n}')
            colors.append(px.colors.qualitative.D3[n])
            sizes.append(14)

    annotations = []
    for c, l, col, s in zip(coordinates, annotation_labels, colors, sizes):
        d = dict(
            x=c[0],
            y=c[1],
            z=c[2],
            text=l,
            font=dict(
                color=col,
                size=s
            ),
            showarrow=False
        )
        annotations.append(d)

    destinations = [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]
    sources = [[-1.2, 0, 0], [0, -1.2, 0], [0, 0, -1.2]]

    for s, d in zip(sources, destinations):
        coords = np.array([s, d])
        ax = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            name="axes",
            mode='lines',
            showlegend=False,
            line=dict(
                color='gray',
                width=1.2,
            ),
        )
        traces.append(ax)

        cone = go.Cone(
            x=[d[0]],
            y=[d[1]],
            z=[d[2]],
            u=[-0.2 * s[0]],
            v=[-0.2 * s[1]],
            w=[-0.2 * s[2]],
            name="Axes Arrows",
            showlegend=False,
            showscale=False,
        )
        traces.append(cone)

    return traces, annotations


def make_bloch_state_traces(bloch_vectors, labels=None, num_classes=None):

    if bloch_vectors is None:
        bloch_vectors = np.zeros((4, 3))

    if labels is None:
        labels = np.array([0, 1, 2, 3], dtype=np.uint)
    else:
        labels = labels.values

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    colors = px.colors.qualitative.D3[:num_classes]

    state_traces = []

    for c in range(num_classes):

        idx = np.where(labels == c)
        states = bloch_vectors[idx]

        state_trace = go.Scatter3d(
            x=states[:, 0].tolist(),
            y=states[:, 1].tolist(),
            z=states[:, 2].tolist(),
            name=f"Class {c} States",
            showlegend=False,
            mode='markers',
            marker=dict(
                size=2,
                color=colors[c],
                opacity=0.5
            ))

        state_traces.append(state_trace)

    return state_traces


def make_bloch_plot(bloch_vectors=None, labels=None, targets=None):
    fig = go.Figure()

    if targets is not None:
        targets = np.array([convert_state_vector_to_bloch_vector(t) for t in targets.numpy()])

    traces, annotations = make_bloch_sphere(targets=targets)
    state_traces  = make_bloch_state_traces(bloch_vectors, labels)

    traces += state_traces

    for t in traces:
        fig.add_trace(t)

    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.7, y=0.7, z=0.7)
    )

    fig.update_layout(
        autosize=False,
        width=300,
        height=300,
        margin=dict(b=0, t=0, l=0, r=0),
        scene_camera=camera,
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
        )
    )

    return fig


def make_bloch_model_plot(bloch_vectors, labels, num_layers, targets=None):

    if bloch_vectors is None and num_layers is None:
        raise ValueError("Must specify either bloch_vectors or num_layers")

    if num_layers is None:
        num_layers = len(bloch_vectors) - 1
    else:
        num_layers -= 1

    num_rows = int(np.ceil(num_layers / 2))

    if targets is not None:
        targets = np.array([convert_state_vector_to_bloch_vector(t) for t in targets.numpy()])

    traces, annotations = make_bloch_sphere(targets=targets)

    fig = make_subplots(rows=num_rows,
                        cols=2,
                        specs=[[{'type': 'surface'}, {'type': 'surface'}] for i in range(num_rows)],
                        # subplot_titles=([f"Layer {i+1}" for i in range(num_layers)])
                        )

    for n in range(num_layers):

        bv = bloch_vectors[n] if bloch_vectors is not None else None
        state_traces = make_bloch_state_traces(bloch_vectors=bv, labels=labels)

        for t in traces:
            fig.add_trace(t, row=n // 2 + 1, col=n % 2 + 1)

        for t in state_traces:
            fig.add_trace(t, row=n // 2 + 1, col=n % 2 + 1)


    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.65, y=0.65, z=0.65)
    )

    layout = dict(
        autosize=False,
        width=600,
        height=300 * num_layers / 2,
        margin=dict(b=0, t=0, l=0, r=0),
    )

    for i in range(1, num_layers + 1):
        num = "" if i == 1 else i
        layout[f"scene{num}"] = dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
        )
        layout[f"scene{num}_camera"] = camera

    fig.update_layout(layout)

    return fig


def make_bloch_comparison_plot(ideal_vectors=None, noisy_vectors=None):
    """Create a side-by-side Bloch sphere comparison for ideal vs. noisy trajectories."""

    if ideal_vectors is not None:
        ideal_vectors = np.asarray(ideal_vectors, dtype=float)
        if ideal_vectors.size == 0:
            ideal_vectors = None
        else:
            ideal_vectors = np.reshape(ideal_vectors, (-1, 3))
    if noisy_vectors is not None:
        noisy_vectors = np.asarray(noisy_vectors, dtype=float)
        if noisy_vectors.size == 0:
            noisy_vectors = None
        else:
            noisy_vectors = np.reshape(noisy_vectors, (-1, 3))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ideal Evolution", "Noisy Evolution"),
        horizontal_spacing=0.08,
    )

    ideal_traces, ideal_annotations = make_bloch_sphere()
    noisy_traces, noisy_annotations = make_bloch_sphere()

    for trace in ideal_traces:
        trace.update(showlegend=False)
        fig.add_trace(trace, row=1, col=1)

    for trace in noisy_traces:
        trace.update(showlegend=False)
        fig.add_trace(trace, row=1, col=2)

    if ideal_vectors is not None:
        fig.add_trace(
            go.Scatter3d(
                x=ideal_vectors[:, 0],
                y=ideal_vectors[:, 1],
                z=ideal_vectors[:, 2],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=4),
                marker=dict(size=3, color='#1f77b4'),
                name='Ideal Trajectory'
            ),
            row=1,
            col=1
        )

    if noisy_vectors is not None:
        fig.add_trace(
            go.Scatter3d(
                x=noisy_vectors[:, 0],
                y=noisy_vectors[:, 1],
                z=noisy_vectors[:, 2],
                mode='lines+markers',
                line=dict(color='#d62728', width=4),
                marker=dict(size=3, color='#d62728'),
                name='Noisy Trajectory'
            ),
            row=1,
            col=2
        )

    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.75, y=0.75, z=0.75)
    )

    layout = dict(
        autosize=False,
        width=650,
        height=320,
        margin=dict(b=0, t=40, l=0, r=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5
        )
    )

    layout["scene"] = dict(
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        annotations=ideal_annotations,
        xaxis=axes_config,
        yaxis=axes_config,
        zaxis=axes_config,
    )
    layout["scene_camera"] = camera

    layout["scene2"] = dict(
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        annotations=noisy_annotations,
        xaxis=axes_config,
        yaxis=axes_config,
        zaxis=axes_config,
    )
    layout["scene2_camera"] = camera

    fig.update_layout(layout)

    return fig
