import dash
from dash import html, dcc, callback, Input, Output
import os

dash.register_page(__name__, path="/", name="Folder Structure", order=7)

CSV_DATA_PATH = "/Users/home/Downloads/CSVDATA/CSV Data"

def build_file_tree(path):
    d = {'name': os.path.basename(path), 'path': path}
    if os.path.isdir(path):
        d['type'] = "directory"
        try:
            children = [build_file_tree(os.path.join(path, x)) for x in os.listdir(path) if not x.startswith('.')]
            d['children'] = sorted(children, key=lambda x: (x['type'] != 'directory', x['name']))
        except PermissionError:
             d['children'] = []
    else:
        d['type'] = "file"
    return d

def render_tree(node):
    if node['type'] == 'file':
        return html.Div([
            html.Span("📄 ", style={'margin-right': '5px'}),
            html.Span(node['name'])
        ], style={'margin-left': '20px'})
    
    children = [render_tree(child) for child in node.get('children', [])]
    
    return html.Details([
        html.Summary([
            html.Span("📁 ", style={'margin-right': '5px'}),
            html.Span(node['name'])
        ], style={'cursor': 'pointer'}),
        html.Div(children, style={'margin-left': '20px'})
    ])

layout = html.Div([
    html.H1("Folder Structure"),
    html.P(f"Browsing: {CSV_DATA_PATH}"),
    html.Button("Refresh", id="refresh-btn", n_clicks=0, className="btn btn-primary mb-3"),
    html.Div(id="folder-tree-container")
])

@callback(
    Output("folder-tree-container", "children"),
    Input("refresh-btn", "n_clicks")
)
def update_tree(n):
    if not os.path.exists(CSV_DATA_PATH):
        return html.Div(f"Path not found: {CSV_DATA_PATH}", style={'color': 'red'})
    
    tree_data = build_file_tree(CSV_DATA_PATH)
    return render_tree(tree_data)
