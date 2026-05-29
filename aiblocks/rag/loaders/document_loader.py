"""Document loader wrapping langchain_community loaders."""

from __future__ import annotations

from pathlib import Path

from aiblocks.rag.config import LoaderConfig


class DocumentLoader:
    """Loads documents from files or directories using langchain_community loaders."""

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config
        self._built = False

    def build(self) -> DocumentLoader:
        try:
            import langchain_community  # noqa: F401
        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")
        self._built = True
        return self

    def load(self, source: str | list[str]) -> list:
        """Load documents from a file path, directory, or list of paths."""
        if not self._built:
            raise RuntimeError("Call build() before load()")

        sources = [source] if isinstance(source, str) else list(source)
        documents = []

        for path in sources:
            p = Path(path)
            if p.is_dir():
                documents.extend(self._load_directory(p))
            elif p.is_file():
                ext = p.suffix.lower()
                if ext not in self.config.supported_extensions:
                    raise ValueError(
                        f"Unsupported file type: {ext}. "
                        f"Supported: {self.config.supported_extensions}"
                    )
                try:
                    documents.extend(self._load_file(p))
                except Exception as exc:
                    print(f"  [warn] Could not load {p.name}: {exc}")
            elif not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            else:
                raise FileNotFoundError(f"File not found: {p}")

        if not documents:
            raise ValueError(f"No documents could be loaded from {source}")

        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_directory(self, directory: Path) -> list:
        pattern = "**/*" if self.config.recursive else "*"
        docs = []
        for fp in sorted(directory.glob(pattern)):
            if fp.is_file() and fp.suffix.lower() in self.config.supported_extensions:
                try:
                    docs.extend(self._load_file(fp))
                except Exception as exc:
                    print(f"  [warn] Skipping {fp.name}: {exc}")
        return docs

    def _load_file(self, path: Path) -> list:
        ext = path.suffix.lower()
        if ext not in self.config.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.config.supported_extensions}"
            )
        return self._get_loader(ext, str(path)).load()

    def _get_loader(self, ext: str, path: str):
        try:
            from langchain_community.document_loaders import (
                BSHTMLLoader,
                CSVLoader,
                Docx2txtLoader,
                PyPDFLoader,
                TextLoader,
            )
        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        dispatch = {
            ".pdf":  lambda p: PyPDFLoader(p),
            ".docx": lambda p: Docx2txtLoader(p),
            ".txt":  lambda p: TextLoader(p, encoding=self.config.encoding),
            ".md":   lambda p: TextLoader(p, encoding=self.config.encoding),
            ".csv":  lambda p: CSVLoader(p, encoding=self.config.encoding),
            ".html": lambda p: BSHTMLLoader(p, open_encoding=self.config.encoding),
            ".json": lambda p: TextLoader(p, encoding=self.config.encoding),
        }

        if ext not in dispatch:
            raise ValueError(f"No loader registered for extension: {ext}")
        return dispatch[ext](path)
