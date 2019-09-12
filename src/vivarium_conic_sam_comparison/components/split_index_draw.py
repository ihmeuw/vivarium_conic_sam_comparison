import pandas as pd

def profile(f):
    import time
    import tracemalloc
    megabyte = 1024*1024
    def _wrapped(*args, **kwargs):
        print(f'Profiling {f.__name__}')
        start = time.time()
        tracemalloc.start()
        mem_start, _ = tracemalloc.get_traced_memory()
        out = f(*args, **kwargs)  # Assume no side effects free memory
        mem_final, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_used = peak - mem_start
        elapsed = time.time() -  start
        print(f'Memory used: {peak_mem_used/megabyte:.2f} MB')
        print(f'Time elapsed: {elapsed:.2f} sec')
        print()
        return out
    return _wrapped

def data_size(data):
    megabyte = 1024*1024
    print(f'Data size: {data.memory_usage(deep=True).sum()/megabyte:.2f} MB')

def select_data(hdf_path, entity_key, where=None, columns=None):
    with pd.HDFStore(hdf_path) as s:
        data = s.select(entity_key, where=where, columns=columns)
    return data

def relabel(data):
    data.index.set_levels(['Global'], level='location', inplace=True)
    return data
    
def affected_entities(data):
    return data.index.levels[data.index.names.index('affected_entity')]

def select_and_stack(data, affected_entity):
    chunk = data.xs(affected_entity, level='affected_entity', drop_level=False)
    chunk = chunk.stack().to_frame().rename(columns={0: 'value'})
    return chunk

def append_data(store, data, key, **kwargs):
    store.append(key, data, **kwargs)
    
def filter_to_affected(data):
    entities = ['neonatal_sepsis_and_other_neonatal_infections',
                'neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma',
                'hemolytic_disease_and_other_neonatal_jaundice',
                'diarrheal_diseases',
                'lower_respiratory_infections']
    return pd.concat([data.xs(e, level='affected_entity', drop_level=False) for e in entities])

    
def write_data(path, key, data):
    with pd.HDFStore(path, complevel=9, mode='w') as store:        
        store.put(key + '/index', data.index.to_frame(index=False))
        data = data.reset_index(drop=True)
        for c in data.columns:
            store.put(f'{key}/{c}', data[c])
        
        
def read_data(path, key, draw):
    with pd.HDFStore(path, mode='r') as store:
        index = store.get(f'{key}/index')
        draw = store.get(f'{key}/draw_{draw}')
        draw.rename("value", inplace=True)
    return pd.concat([index, draw], axis=1)