"""Verify all shipped files against the relative-path integrity manifest."""
from pathlib import Path
import hashlib,json,sys

def main():
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'MANIFEST.json').read_text())['files']
    failed=[]
    for name,info in expected.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
        path=root/rel
        if not path.is_file() or path.is_symlink():failed.append(name);continue
        data=path.read_bytes()
        if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:failed.append(name)
    # Bytecode created by imports is not part of the submitted source archive.
    ignored={'.git','__pycache__','.pytest_cache','build','dist','runs','.venv','.cache'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not any(v in ignored or v.endswith('.egg-info') for v in p.relative_to(root).parts) and p.name!='MANIFEST.json'}
    extra=sorted(actual-set(expected))
    result={'passed':not failed and not extra,'files':len(expected),'failed':failed,'unexpected_files':extra}
    print(json.dumps(result,indent=2))
    if not result['passed']:sys.exit(1)
if __name__=='__main__':main()
