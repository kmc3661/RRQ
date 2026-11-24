"""Selected common execution settings. No prefill CUDA graph."""
def install(model, graphs=True, compact=True):
    from . import memory, silu, rms, decode_graph, attention_metadata
    result = {}
    if compact: result['memory'] = memory.compact(model)
    result['silu'] = silu.install(model)
    result['rms'] = rms.install(model)
    if graphs:
        result['decode_graphs'] = decode_graph.install(model)
        result['attention_metadata'] = attention_metadata.install(model)
    return result
