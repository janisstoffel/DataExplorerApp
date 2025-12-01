import dash
from src.utils.array_config import IA5_CONFIG
from src.components.layout import create_layout
from src.components.callbacks import register_callbacks

dash.register_page(__name__, path=IA5_CONFIG.url_path, name=IA5_CONFIG.name, order=5)

layout = create_layout(IA5_CONFIG)
register_callbacks(IA5_CONFIG)
