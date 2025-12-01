import dash
from src.utils.array_config import PHANTOM_CONFIG
from src.components.layout import create_layout
from src.components.callbacks import register_callbacks

dash.register_page(__name__, path=PHANTOM_CONFIG.url_path, name=PHANTOM_CONFIG.name, order=0)

layout = create_layout(PHANTOM_CONFIG)
register_callbacks(PHANTOM_CONFIG)
