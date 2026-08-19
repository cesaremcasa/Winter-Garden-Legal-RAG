import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formata logs como JSON por linha."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Adiciona campos extras se existirem
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "query"):
            log_data["query"] = record.query
        if hasattr(record, "retrieved_chunks"):
            log_data["retrieved_chunks"] = record.retrieved_chunks
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        
        # Adiciona exception info se existir
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Retorna logger configurado com formato JSON.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo)
        level: Nível de logging (default: INFO)
    
    Returns:
        Logger configurado
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Query received", extra={"request_id": "123", "query": "test"})
    """
    logger = logging.getLogger(name)
    
    # Evita duplicar handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


def generate_request_id(header_value: Optional[str] = None) -> str:
    """
    Gera ou retorna request_id.
    
    Args:
        header_value: Valor do header X-Request-ID (se fornecido)
    
    Returns:
        request_id (do header ou UUID gerado)
    """
    if header_value:
        return header_value
    return str(uuid.uuid4())
