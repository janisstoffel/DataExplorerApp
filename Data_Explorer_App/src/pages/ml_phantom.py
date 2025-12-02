import dash
from dash import html

# Disable this page by not registering it or registering it with a different path/name if needed.
# For now, we just make it a placeholder so imports don't break the build.
dash.register_page(__name__, path='/ml-phantom-disabled', name="ML Phantom (Disabled)", order=99)

layout = html.Div([
    html.H1("ML Phantom - Disabled for Lite Build"),
    html.P("This feature has been removed to reduce application size.")
])

# No callbacks needed

