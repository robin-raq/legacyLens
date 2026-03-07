"""Tests for the /api/source endpoint (full file context with GitHub fallback)."""

import pytest
from unittest.mock import patch, mock_open, MagicMock
import urllib.error


class TestSourceEndpointLocal:
    """Tests for GET /api/source?file=... serving from local disk."""

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

    def test_local_preferred_over_github(self, test_client, tmp_path):
        """Local file should be served without hitting GitHub."""
        f = tmp_path / "dgemm.f"
        f.write_text("      LOCAL VERSION\n")

        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github") as mock_gh:
            resp = test_client.get("/api/source?file=dgemm.f")
        assert resp.status_code == 200
        assert "LOCAL VERSION" in resp.json()["content"]
        mock_gh.assert_not_called()


class TestSourceEndpointGitHubFallback:
    """Tests for GitHub fallback when local file is missing."""

    def test_falls_back_to_github(self, test_client, tmp_path):
        """Missing local file should fetch from GitHub."""
        github_content = "      SUBROUTINE DGEMM\n      FROM GITHUB\n      END\n"

        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github", return_value=github_content):
            resp = test_client.get("/api/source?file=SRC/dgemm.f")
        assert resp.status_code == 200
        data = resp.json()
        assert "FROM GITHUB" in data["content"]
        assert data["file"] == "SRC/dgemm.f"
        assert data["total_lines"] == 3

    def test_github_failure_returns_404(self, test_client, tmp_path):
        """If both local and GitHub fail, return 404."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github", return_value=None):
            resp = test_client.get("/api/source?file=SRC/nonexistent.f")
        assert resp.status_code == 404

    def test_github_builds_correct_url(self, test_client, tmp_path):
        """GitHub fallback should map SRC/dgemm.f to BLAS/SRC/dgemm.f."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github") as mock_gh:
            mock_gh.return_value = None
            test_client.get("/api/source?file=SRC/dgemm.f")
        mock_gh.assert_called_once_with("SRC/dgemm.f")

    def test_path_traversal_still_blocked_with_fallback(self, test_client, tmp_path):
        """Security checks must run before any fallback."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github") as mock_gh:
            resp = test_client.get("/api/source?file=../../etc/passwd.f")
        assert resp.status_code == 400
        mock_gh.assert_not_called()

    def test_non_fortran_blocked_with_fallback(self, test_client, tmp_path):
        """Extension check must run before fallback."""
        with patch("app.api.routes.SOURCE_DIR", str(tmp_path)), \
             patch("app.api.routes._fetch_from_github") as mock_gh:
            resp = test_client.get("/api/source?file=readme.txt")
        assert resp.status_code == 400
        mock_gh.assert_not_called()
