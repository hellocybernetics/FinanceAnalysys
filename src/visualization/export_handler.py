"""
Export handler for multi-format chart output with fallback mechanisms.
"""

import io
import json
from typing import Optional, Literal, Union
from pathlib import Path
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


class ExportHandler:
    """
    Multi-format chart export handler.
    Handles PNG/JPEG/SVG export with Kaleido, and HTML/JSON fallbacks.
    """

    def __init__(self):
        """Initialize the export handler."""
        self._kaleido_available = self._check_kaleido()

    def _check_kaleido(self) -> bool:
        """
        Check if Kaleido is available for static image export.

        Returns:
            True if Kaleido is available
        """
        try:
            import kaleido
            logger.info("Kaleido is available for static image export")
            return True
        except ImportError:
            logger.warning("Kaleido not available. Static image export will fall back to HTML.")
            return False

    def export_chart(
        self,
        fig: go.Figure,
        format: Literal['png', 'jpeg', 'svg', 'html', 'json'] = 'png',
        output_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 1.0
    ) -> Optional[bytes]:
        """
        Export chart to specified format.

        Args:
            fig: Plotly Figure object
            format: Output format ('png', 'jpeg', 'svg', 'html', 'json')
            output_path: File path to save (returns bytes if None)
            width: Image width in pixels (uses figure layout if None)
            height: Image height in pixels (uses figure layout if None)
            scale: Image scale factor

        Returns:
            Bytes of exported content (if output_path is None), otherwise None
        """
        if fig is None:
            logger.error("Figure is None, cannot export")
            return None

        # Try static image export for png/jpeg/svg
        if format in ['png', 'jpeg', 'svg']:
            try:
                return self._export_static_image(fig, format, output_path, width, height, scale)
            except Exception as e:
                logger.warning(f"Static image export failed: {e}")
                logger.info("Falling back to HTML export")
                format = 'html'

        # HTML export
        if format == 'html':
            return self._export_html(fig, output_path)

        # JSON export
        if format == 'json':
            return self._export_json(fig, output_path)

        logger.error(f"Unsupported format: {format}")
        return None

    def _export_static_image(
        self,
        fig: go.Figure,
        format: str,
        output_path: Optional[Union[str, Path]],
        width: Optional[int],
        height: Optional[int],
        scale: float
    ) -> Optional[bytes]:
        """
        Export static image using Kaleido.

        Args:
            fig: Plotly Figure
            format: Image format
            output_path: Output file path
            width: Image width
            height: Image height
            scale: Scale factor

        Returns:
            Image bytes (if output_path is None)

        Raises:
            RuntimeError: If Kaleido is not available
        """
        if not self._kaleido_available:
            raise RuntimeError("Kaleido is not available for static image export")

        # Build export kwargs
        export_kwargs = {
            'format': format,
            'scale': scale
        }
        if width is not None:
            export_kwargs['width'] = width
        if height is not None:
            export_kwargs['height'] = height

        if output_path:
            # Save to file
            fig.write_image(str(output_path), **export_kwargs)
            logger.info(f"Exported {format.upper()} to {output_path}")
            return None
        else:
            # Return bytes
            img_bytes = fig.to_image(**export_kwargs)
            logger.info(f"Exported {format.upper()} to bytes")
            return img_bytes

    def _export_html(
        self,
        fig: go.Figure,
        output_path: Optional[Union[str, Path]]
    ) -> Optional[bytes]:
        """
        Export interactive HTML.

        Args:
            fig: Plotly Figure
            output_path: Output file path

        Returns:
            HTML bytes (if output_path is None)
        """
        if output_path:
            # Save to file
            fig.write_html(str(output_path))
            logger.info(f"Exported HTML to {output_path}")
            return None
        else:
            # Return bytes
            html_bytes = fig.to_html().encode('utf-8')
            logger.info("Exported HTML to bytes")
            return html_bytes

    def _export_json(
        self,
        fig: go.Figure,
        output_path: Optional[Union[str, Path]]
    ) -> Optional[bytes]:
        """
        Export chart as JSON specification.

        Args:
            fig: Plotly Figure
            output_path: Output file path

        Returns:
            JSON bytes (if output_path is None)
        """
        # Convert figure to JSON
        fig_json = fig.to_json()

        if output_path:
            # Save to file
            with open(output_path, 'w') as f:
                f.write(fig_json)
            logger.info(f"Exported JSON to {output_path}")
            return None
        else:
            # Return bytes
            json_bytes = fig_json.encode('utf-8')
            logger.info("Exported JSON to bytes")
            return json_bytes

    def export_to_buffer(
        self,
        fig: go.Figure,
        format: Literal['png', 'jpeg', 'svg', 'html', 'json'] = 'png',
        **kwargs
    ) -> io.BytesIO:
        """
        Export chart to BytesIO buffer.

        Args:
            fig: Plotly Figure
            format: Output format
            **kwargs: Additional export parameters

        Returns:
            BytesIO buffer containing exported content
        """
        content_bytes = self.export_chart(fig, format=format, output_path=None, **kwargs)

        if content_bytes is None:
            raise RuntimeError(f"Failed to export chart to {format}")

        buffer = io.BytesIO(content_bytes)
        buffer.seek(0)
        return buffer

    @property
    def supported_formats(self) -> list:
        """
        Get list of supported export formats.

        Returns:
            List of format strings
        """
        formats = ['html', 'json']
        if self._kaleido_available:
            formats.extend(['png', 'jpeg', 'svg'])
        return formats

    @property
    def kaleido_available(self) -> bool:
        """Check if Kaleido is available."""
        return self._kaleido_available
