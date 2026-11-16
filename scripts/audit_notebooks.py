"""Verify saved notebook execution counts and error-free cell outputs."""
import json
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
audit = {}
for path in (root/'notebooks').glob('0*.ipynb'):
    notebook = nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    audit[path.name] = {
        'code_cells': len(cells),
        'executed_cells': sum(cell.execution_count is not None for cell in cells),
        'errors': sum(output.output_type == 'error' for cell in cells for output in cell.outputs),
    }
assert len(audit) == 2
assert all(row['code_cells'] == row['executed_cells'] and row['errors'] == 0 for row in audit.values())
record = root/'results/verification.json'
verification = json.loads(record.read_text())
verification['notebook_cell_audit'] = audit
verification['notebooks']['notes'] = 'All code cells completed with zero error outputs. Windows selector fallback and local TCP notices were nonfatal; one final kernel teardown emitted a connection-reset notice after computation.'
record.write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
