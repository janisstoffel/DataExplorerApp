import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import os
import sys

# Add src to path to ensure imports work correctly when frozen
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
    sys.path.append(base_path)

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Header / Navbar
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
        ],
        brand="Data Explorer App",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    
    html.Div([
        # Sidebar
        html.Div([
            html.H5("Navigation", className="display-7"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink(page['name'], href=page['path'], active="exact")
                    for page in dash.page_registry.values()
                ],
                vertical=True,
                pills=True,
            ),
        ], style={
            "position": "fixed",
            "top": "6rem",
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "backgroundColor": "#f8f9fa",
            "overflowY": "auto"
        }),
        
        # Main Content
        html.Div([
            dash.page_container
        ], style={
            "marginLeft": "18rem",
            "marginRight": "2rem",
            "padding": "2rem 1rem",
        })
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
