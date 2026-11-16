"""Check generated HTML references, ZIP integrity, and deterministic rerun evidence."""
import argparse
import hashlib
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ['href', 'src']:
            if key in attrs:
                self.links.append(attrs[key])


def verify(compare_first=False):
    parser = Links()
    parser.feed((ROOT/'docs/index.html').read_text(encoding='utf-8'))
    assert len(parser.ids) == len(set(parser.ids)), 'Duplicate HTML IDs'
    local = []
    for link in parser.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path:
            assert not parsed.path.startswith('/'), f'Absolute asset URL breaks Pages subpaths: {link}'
            path = ROOT/'docs'/unquote(parsed.path)
            assert path.is_file(), f'Missing local asset: {link}'
            local.append(link)
        elif parsed.fragment:
            assert parsed.fragment in parser.ids, f'Broken anchor: {link}'
    with zipfile.ZipFile(ROOT/'docs/assets/reproducibility.zip') as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        for required in ['README.md', 'requirements.txt', 'PRD.md', 'data/raw/online_shoppers_intention.csv', 'src/run_all.py', 'src/report_template.html', 'notebooks/01_data_and_reasoning.ipynb', 'notebooks/02_models_and_results.ipynb', 'results/test_predictions.csv', 'results/selection.json', '.github/workflows/pages.yml', 'scripts/verify_artifacts.py', 'docs/index.html']:
            assert required in names, f'Bundle missing {required}'
        assert not any(name.startswith('.venv/') or '/models/' in name for name in names)
    rerun = {}
    if compare_first:
        for name in ['validation_metrics', 'test_metrics', 'validation_predictions', 'test_predictions']:
            before = ROOT/'.tmp'/f'first_{name.replace("_metrics", "")}.csv' if name.endswith('_metrics') else ROOT/'.tmp'/f'first_{name}.csv'
            after = ROOT/'results'/f'{name}.csv'
            assert before.read_bytes() == after.read_bytes(), f'Deterministic rerun changed {name}'
            rerun[name] = hashlib.sha256(after.read_bytes()).hexdigest()
    return {'local_asset_references_checked': len(local), 'unique_html_ids': len(parser.ids), 'zip_integrity': 'passed', 'rerun_byte_identical_results': rerun}


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--compare-first-run', action='store_true')
    print(json.dumps(verify(args.parse_args().compare_first_run), indent=2))
