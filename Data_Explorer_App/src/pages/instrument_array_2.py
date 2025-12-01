import dash
from src.utils.array_config import IA2_CONFIG
from src.components.layout import create_layout
from src.components.callbacks import register_callbacks

dash.register_page(__name__, path=IA2_CONFIG.url_path, name=IA2_CONFIG.name, order=2)

layout = create_layout(IA2_CONFIG)
register_callbacks(IA2_CONFIG)
