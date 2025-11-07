import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError

from config import settings


logger = logging.getLogger("elastic_logger")


class ElasticLogger:
    def __init__(self, host: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.client: Optional[Elasticsearch] = None
        if enabled:
            try:
                self.client = Elasticsearch(hosts=[host])
                # Ping to verify
                self.client.info()
                logger.info("Connected to Elasticsearch at %s", host)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Elasticsearch unavailable: %s", exc)
                self.client = None

    def index(self, index: str, doc: Dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.index(index=index, document=doc)
        except ESConnectionError:
            logger.warning("Failed to send doc to Elasticsearch index=%s", index)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected ES error: %s", exc)


def make_es_logger() -> ElasticLogger:
    return ElasticLogger(settings.ES_HOST, enabled=settings.ES_ENABLED)

