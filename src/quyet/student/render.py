"""E4B chat affixes come from the model; head readouts carry no symbol IDs."""
from ..adapter.render import Renderer as AdapterRenderer


class Renderer(AdapterRenderer):
    def _pack(self, ex, kind, key, symbol_ids):
        request = super()._pack(ex, kind, key, symbol_ids)
        if request['decision']['readout'] == 'head':
            request['decision']['symbol_ids'] = []
        return request
