from dedupe.base import BaseDeduplicator

class NoDeduplicator(BaseDeduplicator):
    """Fallback transparent baseline: Everything goes to IDS."""
    def process_record(self, record: dict) -> bool:
        self.total_records += 1
        self.ids_evaluations += 1
        return False
