import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from src.utils.graph_generator import get_cb_node_positions, get_cb_graph_edges
from src.utils.gnn_model import predict_fields_mock

dash.register_page(__name__, path='/ml-phantom', name="ML Phantom", order=6)

layout = html.Div([
    html.H1("ML Phantom - GNN Field Prediction"),
    
    html.Div([
        html.Button("Toggle Grid Connections", id="btn-toggle-grid", n_clicks=0, className="btn btn-secondary", style={'marginRight': '10px'}),
        html.Button("Toggle Interpolated Surface", id="btn-toggle-surface", n_clicks=0, className="btn btn-info", style={'marginRight': '10px'}),
        html.Button("Run GNN Prediction", id="btn-run-gnn", n_clicks=0, className="btn btn-primary"),
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='gnn-output-container'),
    
    dcc.Graph(id='phantom-3d-graph', style={'height': '80vh'})
])

@callback(
    Output('phantom-3d-graph', 'figure'),
    Output('gnn-output-container', 'children'),
    Input('btn-toggle-grid', 'n_clicks'),
    Input('btn-toggle-surface', 'n_clicks'),
    Input('btn-run-gnn', 'n_clicks'),
    State('phantom-3d-graph', 'figure') # Preserve camera state if possible, though we might just rebuild
)
def update_graph(n_clicks_grid, n_clicks_surface, n_clicks_gnn, current_figure):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    show_grid = (n_clicks_grid % 2) == 0 # Default to showing grid
    show_surface = (n_clicks_surface % 2) == 1 # Default to hiding surface
    
    # Get Data
    positions = get_cb_node_positions()
    edges = get_cb_graph_edges()
    
    # Prepare Node Data
    node_names = []
    x_vals = []
    y_vals = []
    z_vals = []
    
    for name, pos in positions.items():
        node_names.append(name)
        x_vals.append(pos['x'])
        y_vals.append(pos['y'])
        z_vals.append(pos['z'])
        
    # Run Prediction if requested
    predictions = None
    output_msg = ""
    if triggered_id == 'btn-run-gnn' or n_clicks_gnn > 0:
        predictions = predict_fields_mock()
        output_msg = html.Div([
            html.H4("GNN Prediction Run Successfully"),
            html.P("Displaying predicted displacement error (Actual vs Nominal) using trained model.")
        ], style={'color': 'green'})
    
    # Create Scatter Plot for Nodes
    marker_color = 'blue'
    marker_size = 8
    hover_text = node_names
    
    if predictions:
        # Color by prediction
        pred_vals = [predictions[name] for name in node_names]
        marker_color = pred_vals
        hover_text = [f"{name}: {val:.4f} mm" for name, val in zip(node_names, pred_vals)]
        
    trace_nodes = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers+text',
        marker=dict(
            size=marker_size,
            color=marker_color,
            colorscale='Viridis',
            colorbar=dict(title="Displacement (mm)") if predictions else None,
            opacity=0.8
        ),
        text=node_names,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        name='CB Nodes'
    )
    
    data = [trace_nodes]
    
    # Create Interpolated Surface
    if show_surface:
        # 1. Create the Mesh (Surface)
        # alphahull=-1 (Delaunay) creates a solid shape connecting all points
        # intensity interpolates the color values between points (Nearest Neighbor/Barycentric)
        trace_mesh = go.Mesh3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            intensity=marker_color if predictions else [0]*len(x_vals),
            colorscale='Viridis',
            opacity=0.4,
            alphahull=-1, 
            name='Interpolated Field',
            showscale=False,
            hoverinfo='skip' # Reduce clutter
        )
        data.append(trace_mesh)

        # 2. Create the Perimeter (Convex Hull Wireframe)
        # We calculate the 3D Convex Hull to find the "outer points" and edges
        points = np.column_stack((x_vals, y_vals, z_vals))
        if len(points) >= 4: # ConvexHull needs at least 4 points
            hull = ConvexHull(points)
            
            # Extract the edges of the hull simplices (triangles)
            hull_x = []
            hull_y = []
            hull_z = []
            
            for simplex in hull.simplices:
                # simplex is a list of 3 indices forming a triangle on the hull
                # We draw the triangle edges: A-B, B-C, C-A
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                hull_x.extend([p1[0], p2[0], None, p2[0], p3[0], None, p3[0], p1[0], None])
                hull_y.extend([p1[1], p2[1], None, p2[1], p3[1], None, p3[1], p1[1], None])
                hull_z.extend([p1[2], p2[2], None, p2[2], p3[2], None, p3[2], p1[2], None])
            
            trace_hull = go.Scatter3d(
                x=hull_x,
                y=hull_y,
                z=hull_z,
                mode='lines',
                line=dict(color='white', width=4),
                name='Outer Perimeter',
                hoverinfo='none'
            )
            data.append(trace_hull)
    
    # Create Lines for Edges
    if show_grid:
        edge_x = []
        edge_y = []
        edge_z = []
        
        for start, end in edges:
            if start in positions and end in positions:
                p1 = positions[start]
                p2 = positions[end]
                edge_x.extend([p1['x'], p2['x'], None])
                edge_y.extend([p1['y'], p2['y'], None])
                edge_z.extend([p1['z'], p2['z'], None])
                
        trace_edges = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none',
            name='Grid Connections'
        )
        data.append(trace_edges)
        
    layout = go.Layout(
        title="Phantom CB Smart Grid",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    return fig, output_msg
