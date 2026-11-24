"""Dependency-light runner for the reference/scoring test suite."""
import importlib.util, inspect, json, tempfile
from pathlib import Path

def main():
    passed=[]
    for p in sorted((Path(__file__).resolve().parents[1]/'tests').glob('test_*.py')):
        spec=importlib.util.spec_from_file_location(p.stem,p); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith('test_'):continue
            if 'tmp_path' in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as root:fn(Path(root))
            else:fn()
            passed.append(name)
    print(json.dumps({'passed':True,'tests':passed},indent=2))
if __name__=='__main__':main()
