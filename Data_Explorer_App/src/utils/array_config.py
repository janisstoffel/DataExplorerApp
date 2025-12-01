
from src.utils.data_loader import load_cs_data_robust
from src.utils.nominals import NOMINALS_MAP

class ArrayConfig:
    def __init__(self, key, name, file_prefix, url_path, axis_mapping='phantom', connection_order=None):
        self.key = key
        self.name = name
        self.file_prefix = file_prefix
        self.url_path = url_path
        self.id_prefix = key  # Use key as ID prefix
        self.nominals = NOMINALS_MAP.get(key, {})
        self.axis_mapping = axis_mapping
        # Default connection order: 1-2-3-4
        self.connection_order = connection_order if connection_order else [1, 2, 3, 4]
        self._data_cache = None

    def get_data(self, refresh=False):
        if self._data_cache is None or refresh:
            self._data_cache = load_cs_data_robust(file_prefix=self.file_prefix)
        return self._data_cache

    def get_nominal_markers(self):
        return self.nominals

# Define configurations for all arrays
PHANTOM_CONFIG = ArrayConfig(
    key='phantom',
    name='Phantom',
    file_prefix='VIANT_PHANTOM_',
    url_path='/phantom',
    axis_mapping='phantom',
    connection_order=[1, 2, 3, 4]
)

IA1_CONFIG = ArrayConfig(
    key='ia1',
    name='Instrument Array 1',
    file_prefix='VIANT_INSTRUMENT_ARRAY_1_',
    url_path='/ia1',
    axis_mapping='instrument',
    connection_order=[1, 2, 4, 3]
)

IA2_CONFIG = ArrayConfig(
    key='ia2',
    name='Instrument Array 2',
    file_prefix='VIANT_INSTRUMENT_ARRAY_2_',
    url_path='/ia2',
    axis_mapping='instrument',
    connection_order=[1, 2, 4, 3]
)

IA3_CONFIG = ArrayConfig(
    key='ia3',
    name='Instrument Array 3',
    file_prefix='VIANT_INSTRUMENT_ARRAY_3_',
    url_path='/ia3',
    axis_mapping='instrument',
    connection_order=[1, 2, 4, 3]
)

IA4_CONFIG = ArrayConfig(
    key='ia4',
    name='Instrument Array 4',
    file_prefix='VIANT_INSTRUMENT_ARRAY_4_',
    url_path='/ia4',
    axis_mapping='instrument',
    connection_order=[1, 2, 4, 3]
)

IA5_CONFIG = ArrayConfig(
    key='ia5',
    name='Instrument Array 5',
    file_prefix='VIANT_INSTRUMENT_ARRAY_5_',
    url_path='/ia5',
    axis_mapping='instrument',
    connection_order=[1, 2, 4, 3]
)
