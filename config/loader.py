from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carrega configuração do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
    
    Returns:
        Dicionário com configurações
    
    Raises:
        FileNotFoundError: Se arquivo não existir
        yaml.YAMLError: Se arquivo estiver mal formatado
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    return config


def get_config_value(key: str, default: Any = None, config_path: str = "config/config.yaml") -> Any:
    """
    Retorna valor específico da configuração.
    
    Args:
        key: Chave da configuração
        default: Valor padrão se chave não existir
        config_path: Caminho para o arquivo de configuração
    
    Returns:
        Valor da configuração ou default
    """
    config = load_config(config_path)
    return config.get(key, default)
