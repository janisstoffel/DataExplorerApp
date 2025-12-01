import dash
from src.utils.array_config import IA3_CONFIG
from src.components.layout import create_layout
from src.components.callbacks import register_callbacks

dash.register_page(__name__, path=IA3_CONFIG.url_path, name=IA3_CONFIG.name, order=3)

layout = create_layout(IA3_CONFIG)
register_callbacks(IA3_CONFIG)
