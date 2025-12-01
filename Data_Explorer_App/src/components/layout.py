
from dash import html, dcc

def create_layout(array_config):
    prefix = array_config.id_prefix
    nominals = array_config.get_nominal_markers()
    
    # Generate center options from nominals
    center_options = [{'label': 'Global Center', 'value': 'global'}]
    for key in nominals.keys():
        center_options.append({'label': key, 'value': key})

    return html.Div([
        html.H1(f"{array_config.name} 3D Visualization"),
        html.P(f"Interactive 3D model of the {array_config.name}."),
        
        html.Div([
            html.Label("Navigation Menu (Folders & Files):", className="form-label"),
            dcc.Dropdown(
                id=f'{prefix}-file-folder-dropdown',
                multi=True,
                placeholder="Select folders or files...",
                style={'width': '100%'},
                persistence=True,
                persistence_type='session'
            ),
            html.Br(),
            dcc.Input(
                id=f'{prefix}-search-input',
                type='text',
                placeholder='Search CSV Filename...',
                className="form-control",
                style={'marginBottom': '10px'},
                persistence=True,
                persistence_type='session'
            )
        ], className="mb-3 p-3 bg-light border rounded"),

        html.Div([
            html.Button("Refresh Data", id=f'{prefix}-refresh-data-btn', n_clicks=0, className="btn btn-primary me-2"),
            html.Span(id=f'{prefix}-data-status', style={'font-style': 'italic'})
        ], className="mb-3"),
        
        html.Div([
            html.Div([
                html.Label("Center View On:", className="form-label"),
                dcc.Dropdown(
                    id=f'{prefix}-center-dropdown',
                    options=center_options,
                    value='global',
                    clearable=False,
                    persistence=True,
                    persistence_type='session'
                )
            ], className="col-md-3"),
            html.Div([
                html.Label("Point Size (1x - 5x):", className="form-label"),
                dcc.Slider(
                    id=f'{prefix}-point-size-slider',
                    min=1,
                    max=5,
                    step=0.1,
                    value=1,
                    marks={1: '1x', 2: '2x', 3: '3x', 4: '4x', 5: '5x'},
                    persistence=True,
                    persistence_type='session'
                )
            ], className="col-md-4"),
            html.Div([
                html.Label("Color Scale:", className="form-label"),
                dcc.Dropdown(
                    id=f'{prefix}-colorscale-dropdown',
                    options=[
                        {'label': 'Default (Time Gradient)', 'value': 'default'},
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'},
                        {'label': 'YlOrRd', 'value': 'YlOrRd'},
                        {'label': 'Bluered', 'value': 'Bluered'},
                        {'label': 'RdBu', 'value': 'RdBu'},
                    ],
                    value='default',
                    clearable=False,
                    persistence=True,
                    persistence_type='session'
                )
            ], className="col-md-3"),
        ], className="row mb-3"),

        html.Div([
            html.Div([
                html.Label("Outlier Sigma:", className="form-label"),
                dcc.Dropdown(
                    id=f'{prefix}-sigma-dropdown',
                    options=[
                        {'label': '0.5 Sigma', 'value': 0.5},
                        {'label': '1 Sigma', 'value': 1},
                        {'label': '2 Sigma', 'value': 2},
                        {'label': '3 Sigma', 'value': 3},
                        {'label': '4 Sigma', 'value': 4},
                        {'label': '5 Sigma', 'value': 5},
                    ],
                    value=3,
                    clearable=False,
                    persistence=True,
                    persistence_type='session'
                )
            ], className="col-md-3"),
            html.Div([
                html.Label("Selected Point Info:", className="form-label"),
                html.Div(id=f'{prefix}-selection-info', className="alert alert-info", style={'padding': '5px'})
            ], className="col-md-9"),
        ], className="row mb-3"),

        # New Section for Vector Deviations
        html.Div([
            html.Div([
                html.Label("Expansion Factor (1x - 20x):", className="form-label"),
                dcc.Slider(
                    id=f'{prefix}-expansion-slider',
                    min=1,
                    max=20,
                    step=1,
                    value=1,
                    marks={i: f'{i}x' for i in range(1, 21, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    persistence=True,
                    persistence_type='session'
                )
            ], className="col-md-8"),
            html.Div([
                html.Label("Kit Deviations:", className="form-label"),
                html.Div(
                    id=f'{prefix}-deviation-list',
                    className="border rounded p-2",
                    style={'height': '200px', 'overflowY': 'auto', 'backgroundColor': '#f8f9fa'}
                )
            ], className="col-md-4"),
        ], className="row mb-4"),
        
        dcc.Graph(id=f'{prefix}-graph', style={'height': '80vh'}),
        dcc.Store(id=f'{prefix}-selected-point-store')
    ])
