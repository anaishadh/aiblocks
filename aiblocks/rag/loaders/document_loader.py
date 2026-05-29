"""Document loader wrapping langchain_community loaders."""

from __future__ import annotations

from pathlib import Path

from aiblocks.rag.config import LoaderConfig

# PDFs with fewer extractable characters than this are treated as scanned.
_SCANNED_PDF_THRESHOLD = 50


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
        if ext == ".pdf":
            return self._load_pdf(path)
        return self._get_loader(ext, str(path)).load()

    def _load_pdf(self, path: Path) -> list:
        """Load a PDF, falling back to OCR when the file appears to be scanned."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        docs = PyPDFLoader(str(path)).load()
        combined_text = " ".join(d.page_content for d in docs).strip()

        if len(combined_text) >= _SCANNED_PDF_THRESHOLD:
            return docs

        # Too little text — likely a scanned PDF.
        if not self.config.use_ocr:
            print(
                f"  [warn] {path.name} appears to be a scanned PDF. "
                "Set use_ocr=True to enable OCR extraction. "
                "Requires: pip install pytesseract pdf2image"
            )
            return []

        return self._ocr_pdf(path)

    def _ocr_pdf(self, path: Path) -> list:
        """Extract text from a scanned PDF via pytesseract + pdf2image."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "OCR requires: pip install pytesseract pdf2image\n"
                "Also install Tesseract OCR engine:\n"
                "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        images = convert_from_path(str(path))
        pages = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": str(path), "page": i},
                ))
        return pages

    def _get_loader(self, ext: str, path: str):
        try:
            from langchain_community.document_loaders import (
                BSHTMLLoader,
                CSVLoader,
                Docx2txtLoader,
                TextLoader,
            )
        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        dispatch = {
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
