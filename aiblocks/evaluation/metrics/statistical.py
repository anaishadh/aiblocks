"""Statistical text-overlap metrics: BLEU, ROUGE, BERTScore, METEOR."""

from __future__ import annotations

from aiblocks.evaluation.config import StatisticalMetricsConfig


class StatisticalMetrics:
    """Computes reference-based statistical metrics over prediction/reference pairs."""

    def __init__(self, config: StatisticalMetricsConfig) -> None:
        self.config = config
        self._built = False

    def build(self) -> StatisticalMetrics:
        """Validate that all required libraries are importable."""
        try:
            if self.config.bleu or self.config.meteor:
                import nltk  # noqa: F401
            if self.config.rouge:
                from rouge_score import rouge_scorer  # noqa: F401
            if self.config.bertscore:
                import bert_score  # noqa: F401
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        self._built = True
        return self

    # ------------------------------------------------------------------
    # Public compute methods
    # ------------------------------------------------------------------

    def compute_bleu(self, predictions: list[str], references: list[str]) -> float:
        """Corpus-level BLEU with add-one smoothing."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        smoothie = SmoothingFunction().method1
        ref_lists = [[ref.split()] for ref in references]
        hyp_lists = [pred.split() for pred in predictions]
        return float(corpus_bleu(ref_lists, hyp_lists, smoothing_function=smoothie))

    def compute_rouge(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
        try:
            from rouge_score import rouge_scorer as rs
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        buckets: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            out = scorer.score(ref, pred)
            for key in buckets:
                buckets[key].append(out[key].fmeasure)
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def compute_bertscore(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """BERTScore precision, recall, and F1 averaged over the corpus."""
        try:
            from bert_score import score as bscore
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        P, R, F1 = bscore(predictions, references, lang="en", verbose=False)
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }

    def compute_meteor(self, predictions: list[str], references: list[str]) -> float:
        """Average METEOR score (requires NLTK WordNet data)."""
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        self._ensure_nltk_data(nltk)
        scores = [
            meteor_score([ref.split()], pred.split())
            for pred, ref in zip(predictions, references)
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def compute(self, dataset: list[dict]) -> dict[str, float]:
        """Run all enabled metrics and return a flat results dict."""
        predictions = [item["answer"] for item in dataset]
        references = [item.get("ground_truth", "") for item in dataset]
        results: dict[str, float] = {}

        if self.config.bleu:
            results["bleu"] = self.compute_bleu(predictions, references)
        if self.config.rouge:
            results.update(self.compute_rouge(predictions, references))
        if self.config.bertscore:
            results.update(self.compute_bertscore(predictions, references))
        if self.config.meteor:
            results["meteor"] = self.compute_meteor(predictions, references)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_nltk_data(nltk) -> None:
        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
