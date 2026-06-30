"""
Notebook Assembler — Combines cell definition files into a Jupyter Notebook.

This script imports get_cells() from each section file, concatenates them,
and produces the final .ipynb file using nbformat.
"""
import nbformat
import sys
import os

# Add project root to path
project_root = r'c:\Users\HP\Documents\Stock_pulse'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'notebook_parts'))

def assemble_notebook():
    """Assemble all notebook sections into a single .ipynb file."""
    
    # Import cell definitions from each part
    from sections_0_to_7 import get_cells as get_cells_part1
    from sections_8_to_16 import get_cells as get_cells_part2
    
    # Combine all cells
    all_cells = get_cells_part1() + get_cells_part2()
    
    # Create notebook
    nb = nbformat.v4.new_notebook()
    
    # Set notebook metadata
    nb.metadata.update({
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.13',
            'mimetype': 'text/x-python',
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'pygments_lexer': 'ipython3',
            'nbconvert_exporter': 'python',
            'file_extension': '.py'
        },
        'title': 'Sri Lankan CSE Market Analysis - Phase 1: Exploratory Data Analysis',
    })
    
    # Build cells
    cell_count = {'markdown': 0, 'code': 0}
    for cell_type, source in all_cells:
        source = source.strip()
        if not source:
            continue
            
        if cell_type == 'markdown':
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
            cell_count['markdown'] += 1
        elif cell_type == 'code':
            nb.cells.append(nbformat.v4.new_code_cell(source))
            cell_count['code'] += 1
        else:
            print(f"WARNING: Unknown cell type '{cell_type}', skipping.")
    
    # Write notebook
    output_path = os.path.join(project_root, '01_CSE_Exploratory_Data_Analysis.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"\n{'='*60}")
    print(f"Notebook assembled successfully!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total cells: {cell_count['markdown'] + cell_count['code']}")
    print(f"  Markdown cells: {cell_count['markdown']}")
    print(f"  Code cells: {cell_count['code']}")
    print(f"{'='*60}")
    
    return output_path


if __name__ == '__main__':
    assemble_notebook()
