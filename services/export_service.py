# services/export_service.py
import json
import csv
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import zipfile
import io
from enum import Enum

from config.settings import settings
from core.utils.serializers import JSONEncoder
from core.utils.logger import get_logger

logger = get_logger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    PICKLE = "pickle"
    HTML = "html"
    MARKDOWN = "markdown"


class ExportService:
    """Service for exporting simulation results and data."""

    def __init__(self):
        self.exports_dir = settings.EXPORTS_DIR
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Export service initialized at {self.exports_dir}")

    def export_simulation(
            self,
            result: Any,
            format: ExportFormat = ExportFormat.JSON,
            filename: Optional[str] = None,
            include_metadata: bool = True
    ) -> Path:
        """Export simulation result."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}.{format.value}"

        filepath = self.exports_dir / filename

        try:
            if format == ExportFormat.JSON:
                self._export_json(result, filepath, include_metadata)
            elif format == ExportFormat.CSV:
                self._export_csv(result, filepath, include_metadata)
            elif format == ExportFormat.EXCEL:
                self._export_excel(result, filepath, include_metadata)
            elif format == ExportFormat.PARQUET:
                self._export_parquet(result, filepath, include_metadata)
            elif format == ExportFormat.PICKLE:
                self._export_pickle(result, filepath)
            elif format == ExportFormat.HTML:
                self._export_html(result, filepath, include_metadata)
            elif format == ExportFormat.MARKDOWN:
                self._export_markdown(result, filepath, include_metadata)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported simulation to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    def export_batch(
            self,
            results: List[Any],
            format: ExportFormat = ExportFormat.JSON,
            filename: Optional[str] = None
    ) -> Path:
        """Export multiple simulations."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulations_batch_{timestamp}.{format.value}"

        filepath = self.exports_dir / filename

        if format == ExportFormat.JSON:
            data = [self._prepare_for_export(r) for r in results]
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=JSONEncoder, indent=2)
        elif format == ExportFormat.EXCEL:
            self._export_batch_excel(results, filepath)
        else:
            # Create ZIP archive with individual files
            self._export_batch_zip(results, format, filepath)

        return filepath

    def export_report(
            self,
            result: Any,
            template: str = "standard",
            filename: Optional[str] = None
    ) -> Path:
        """Generate a comprehensive report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"

        filepath = self.exports_dir / filename

        # Generate report based on template
        if template == "standard":
            self._generate_standard_report(result, filepath)
        elif template == "executive":
            self._generate_executive_report(result, filepath)
        elif template == "technical":
            self._generate_technical_report(result, filepath)
        else:
            raise ValueError(f"Unknown template: {template}")

        return filepath

    def create_zip_archive(
            self,
            files: List[Path],
            filename: Optional[str] = None
    ) -> Path:
        """Create a ZIP archive of multiple files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"archive_{timestamp}.zip"

        zip_path = self.exports_dir / filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if file.exists():
                    zipf.write(file, file.name)

        return zip_path

    def export_to_memory(self, result: Any, format: ExportFormat = ExportFormat.JSON) -> bytes:
        """Export to memory buffer (for API responses)."""
        buffer = io.BytesIO()

        if format == ExportFormat.JSON:
            data = self._prepare_for_export(result)
            json_str = json.dumps(data, cls=JSONEncoder, indent=2)
            buffer.write(json_str.encode('utf-8'))
        elif format == ExportFormat.CSV:
            df = self._convert_to_dataframe(result)
            df.to_csv(buffer, index=False)
        elif format == ExportFormat.PICKLE:
            pickle.dump(result, buffer)

        buffer.seek(0)
        return buffer.getvalue()

    def get_export_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of recent exports."""
        exports = []

        for file in sorted(self.exports_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if file.is_file():
                stats = file.stat()
                exports.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': stats.st_size,
                    'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
                })

            if len(exports) >= limit:
                break

        return exports

    def cleanup_old_exports(self, days_old: int = 30):
        """Clean up exports older than specified days."""
        cutoff = datetime.now().timestamp() - (days_old * 86400)
        deleted = 0

        for file in self.exports_dir.glob("*"):
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
                deleted += 1

        logger.info(f"Cleaned up {deleted} old export files")

    # Private methods
    def _prepare_for_export(self, result: Any) -> Dict[str, Any]:
        """Prepare result for export."""
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        elif isinstance(result, dict):
            return result
        else:
            return {'data': result}

    def _export_json(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to JSON format."""
        data = self._prepare_for_export(result)

        if include_metadata:
            data = self._add_export_metadata(data)

        with open(filepath, 'w') as f:
            json.dump(data, f, cls=JSONEncoder, indent=2)

    def _export_csv(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to CSV format."""
        data = self._prepare_for_export(result)

        if include_metadata:
            data = self._add_export_metadata(data)

        # Flatten nested structures
        flat_data = self._flatten_dict(data)

        df = pd.DataFrame([flat_data])
        df.to_csv(filepath, index=False)

    def _export_excel(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to Excel format with multiple sheets."""
        data = self._prepare_for_export(result)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main results sheet
            main_data = {k: [v] for k, v in data.items() if not isinstance(v, (dict, list))}
            pd.DataFrame(main_data).to_excel(writer, sheet_name='Summary', index=False)

            # Metrics sheet
            if 'metrics' in data:
                metrics_df = pd.DataFrame([data['metrics']])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            # Parameters sheet
            if 'parameters' in data:
                params_df = pd.DataFrame([data['parameters']])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)

            # Training history sheet
            if 'training_history' in data:
                history_df = pd.DataFrame(data['training_history'])
                history_df.to_excel(writer, sheet_name='Training History', index=True)

    def _export_parquet(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to Parquet format."""
        data = self._prepare_for_export(result)
        flat_data = self._flatten_dict(data)

        df = pd.DataFrame([flat_data])
        df.to_parquet(filepath, index=False)

    def _export_pickle(self, result: Any, filepath: Path):
        """Export to pickle format."""
        with open(filepath, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _export_html(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to HTML format."""
        data = self._prepare_for_export(result)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metric { background: #e8f5e9; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .warning { background: #fff3e0; }
                .danger { background: #ffebee; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Simulation Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().isoformat())

        # Add data sections
        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<h2>{key.replace('_', ' ').title()}</h2>"
                html += "<ul>"
                for k, v in value.items():
                    html += f"<li><strong>{k}:</strong> {v}</li>"
                html += "</ul>"
            else:
                html += f"<p><strong>{key}:</strong> {value}</p>"

        html += "</body></html>"

        with open(filepath, 'w') as f:
            f.write(html)

    def _export_markdown(self, result: Any, filepath: Path, include_metadata: bool = True):
        """Export to Markdown format."""
        data = self._prepare_for_export(result)

        markdown = f"""# Simulation Report\n\n"""
        markdown += f"**Generated:** {datetime.now().isoformat()}\n\n"

        for key, value in data.items():
            if isinstance(value, dict):
                markdown += f"## {key.replace('_', ' ').title()}\n\n"
                for k, v in value.items():
                    markdown += f"- **{k}:** {v}\n"
                markdown += "\n"
            else:
                markdown += f"**{key}:** {value}\n\n"

        with open(filepath, 'w') as f:
            f.write(markdown)

    def _export_batch_excel(self, results: List[Any], filepath: Path):
        """Export batch to Excel with multiple sheets."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for i, result in enumerate(results):
                data = self._prepare_for_export(result)
                df = pd.DataFrame([self._flatten_dict(data)])
                df.to_excel(writer, sheet_name=f'Simulation_{i + 1}', index=False)

    def _export_batch_zip(self, results: List[Any], format: ExportFormat, zip_path: Path):
        """Export batch as ZIP archive."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, result in enumerate(results):
                # Export individual file
                filename = f"simulation_{i + 1}.{format.value}"
                temp_path = self.exports_dir / f"temp_{filename}"

                self.export_simulation(result, format, filename=temp_path.name)

                # Add to ZIP
                zipf.write(temp_path, filename)

                # Clean up temp file
                temp_path.unlink()

    def _generate_standard_report(self, result: Any, filepath: Path):
        """Generate standard report."""
        # Implementation would use reportlab or similar
        # For now, create a simple text file
        data = self._prepare_for_export(result)

        with open(filepath, 'w') as f:
            f.write("SIMULATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            for key, value in data.items():
                if isinstance(value, dict):
                    f.write(f"{key.upper()}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")
                else:
                    f.write(f"{key}: {value}\n")

    def _generate_executive_report(self, result: Any, filepath: Path):
        """Generate executive summary report."""
        # Simplified implementation
        self._generate_standard_report(result, filepath)

    def _generate_technical_report(self, result: Any, filepath: Path):
        """Generate technical detailed report."""
        # Simplified implementation
        self._generate_standard_report(result, filepath)

    def _add_export_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add export metadata to data."""
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'export_format': 'json',
            'gags_version': settings.APP_VERSION,
            'environment': settings.ENVIRONMENT
        }

        if 'metadata' in data:
            data['metadata'].update(metadata)
        else:
            data['metadata'] = metadata

        return data

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings
                items.append((new_key, json.dumps(v, cls=JSONEncoder)))
            else:
                items.append((new_key, v))

        return dict(items)

    def _convert_to_dataframe(self, result: Any) -> pd.DataFrame:
        """Convert result to pandas DataFrame."""
        data = self._prepare_for_export(result)
        flat_data = self._flatten_dict(data)
        return pd.DataFrame([flat_data])