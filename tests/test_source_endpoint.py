"""Tests for the /api/source endpoint (full file context)."""

import pytest
from unittest.mock import patch, mock_open


class TestSourceEndpoint:
    """Tests for GET /api/source?file=..."""

    def test_returns_file_content(self, test_client, tmp_path):
        """Valid Fortran file should return its content."""
        f = tmp_path / "dgemm.f"
        f.write_text("      SUBROUTINE DGEMM\n      END\n")

        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=dgemm.f")
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert "SUBROUTINE DGEMM" in data["content"]
        assert data["file"] == "dgemm.f"

    def test_missing_file_returns_404(self, test_client, tmp_path):
        """Nonexistent file should return 404."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=nonexistent.f")
        assert resp.status_code == 404

    def test_missing_param_returns_422(self, test_client):
        """Missing file parameter should return 422."""
        resp = test_client.get("/api/source")
        assert resp.status_code == 422

    def test_path_traversal_blocked(self, test_client, tmp_path):
        """Path traversal attempts should be rejected."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=../../.env")
        assert resp.status_code == 400

    def test_path_traversal_encoded_blocked(self, test_client, tmp_path):
        """Even sneaky traversal should be blocked."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=..%2F..%2F.env")
        assert resp.status_code == 400

    def test_non_fortran_extension_blocked(self, test_client, tmp_path):
        """Only Fortran extensions should be servable."""
        txt = tmp_path / "readme.txt"
        txt.write_text("not fortran")
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=readme.txt")
        assert resp.status_code == 400

    def test_subdirectory_file(self, test_client, tmp_path):
        """Files in subdirectories like SRC/dgemm.f should work."""
        src = tmp_path / "SRC"
        src.mkdir()
        f = src / "dgemm.f"
        f.write_text("      SUBROUTINE DGEMM\n      END\n")

        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=SRC/dgemm.f")
        assert resp.status_code == 200
        assert "SUBROUTINE DGEMM" in resp.json()["content"]

    def test_line_count_in_response(self, test_client, tmp_path):
        """Response should include total line count."""
        f = tmp_path / "test.f"
        f.write_text("line1\nline2\nline3\n")
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)):
            resp = test_client.get("/api/source?file=test.f")
        assert resp.status_code == 200
        assert resp.json()["total_lines"] == 3
