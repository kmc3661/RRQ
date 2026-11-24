"""Dependency-light evaluation manifest validation."""
import json
from pathlib import Path
TASKS = ('ai2d', 'chartqa', 'mmstar', 'mmmu', 'textvqa', 'vizwiz', 'nocaps', 'textcaps')

def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]
    if not rows:
        raise ValueError('Empty evaluation manifest')
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        seen.add(row['id'])
        if row['task'] not in TASKS:
            raise ValueError(f"Unknown task: {row['task']}")
        if not isinstance(row['prompt'], str) or not row['images']:
            raise ValueError('Each record needs a prompt and at least one image')
        for image in row['images']:
            p = Path(image)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Image paths must be relative to --data-root')
        generation_options(row)
    return rows

def generation_options(row):
    options = dict(row['generation'])
    if int(options.get('max_new_tokens', 0)) <= 0:
        raise ValueError('generation.max_new_tokens must be positive')
    options.setdefault('do_sample', False)
    options['use_cache'] = True
    return options
