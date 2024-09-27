"""Module for executing Jypter Notebook and generating a report from it."""
import logging
import os
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets.config.loader import Config as ExporterConfig

from src.config.core import Config

logger = logging.getLogger(__name__)


class CustomExecutePreprocessor(ExecutePreprocessor):
    """ExecutePreProcessor with custom logging an error handling."""

    def preprocess_cell(self, cell, resources, index):
        """
        Execute a single code in Jypter Notebook Cell.

        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        index : int
            Index of the cell being processed
        """
        logger.info(f'Executing cell {index + 1}...')
        self._check_assign_resources(resources)
        try:
            cell = self.execute_cell(cell, index, store_history=True)
        except Exception as e:
            logger.info(f'Error executing cell {index + 1} moving to next cell: {e}')
        return cell, self.resources


def execute_notebook(nb: nbformat.NotebookNode):
    """Execute a notebook."""
    ep = CustomExecutePreprocessor(timeout=None, kernel_name='python3')
    logger.info(f'Found {len(nb.cells)} cells in the notebook.')
    nb, _ = ep.preprocess(nb, {'metadata': {}})
    return nb


def export_to_html(nb: nbformat.NotebookNode, path: str | Path):
    """Export a notebook to HTML without input cells and prompts."""
    config = ExporterConfig(
        {'TemplateExporter': {'exclude_input': True, 'exclude_input_prompt': True}}
    )
    exporter = HTMLExporter(config=config)
    body, _ = exporter.from_notebook_node(nb)
    with open(path, 'w') as f:
        f.write(body)


def generate_html_report(config: Config):
    """Generate a HTML report from a Jupyter Notebook for given experiment config."""
    os.environ['SANDBOX_EXPERIMENT_DIR'] = config.experiment_dir.__str__()
    nb_path = Path(__file__).parent / 'inference.ipynb'
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    logger.info('> Executing notebook...')
    nb = execute_notebook(nb)
    logger.info('> Exporting to HTML...')
    export_to_html(nb, config.experiment_dir / 'report.html')
    logger.info(
        f'> Report generated successfully for experiment {config.experiment_name}'
    )
    return config.experiment_dir / 'report.html'
