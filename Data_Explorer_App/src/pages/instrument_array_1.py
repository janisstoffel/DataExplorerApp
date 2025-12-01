import dash
from src.utils.array_config import IA1_CONFIG
from src.components.layout import create_layout
from src.components.callbacks import register_callbacks

dash.register_page(__name__, path=IA1_CONFIG.url_path, name=IA1_CONFIG.name, order=1)

layout = create_layout(IA1_CONFIG)
register_callbacks(IA1_CONFIG)
