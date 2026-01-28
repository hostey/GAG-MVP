# core/utils/serializers.py
import json
import pickle
import base64
import zlib
from typing import Any, Dict, Optional, Union
from datetime import datetime, date
import numpy as np
import pandas as pd
import torch

from core.utils.logger import get_logger

logger = get_logger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')

        return super().default(obj)


class Serializer:
    """Main serializer for simulation data."""

    @staticmethod
    def to_json(data: Any, indent: int = 2, compress: bool = False) -> Union[str, bytes]:
        """Serialize data to JSON."""
        try:
            json_str = json.dumps(data, cls=JSONEncoder, indent=indent)

            if compress:
                compressed = zlib.compress(json_str.encode('utf-8'))
                return base64.b64encode(compressed).decode('utf-8')

            return json_str
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            raise

    @staticmethod
    def from_json(json_data: Union[str, bytes], compressed: bool = False) -> Any:
        """Deserialize JSON data."""
        try:
            if compressed:
                if isinstance(json_data, str):
                    json_data = base64.b64decode(json_data)
                decompressed = zlib.decompress(json_data)
                json_str = decompressed.decode('utf-8')
            else:
                json_str = json_data if isinstance(json_data, str) else json_data.decode('utf-8')

            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise

    @staticmethod
    def to_pickle(data: Any, compress: bool = True) -> bytes:
        """Serialize data to pickle format."""
        try:
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            if compress:
                pickled = zlib.compress(pickled)

            return pickled
        except Exception as e:
            logger.error(f"Pickle serialization failed: {e}")
            raise

    @staticmethod
    def from_pickle(pickled_data: bytes, compressed: bool = True) -> Any:
        """Deserialize pickle data."""
        try:
            if compressed:
                pickled_data = zlib.decompress(pickled_data)

            return pickle.loads(pickled_data)
        except Exception as e:
            logger.error(f"Pickle deserialization failed: {e}")
            raise

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """Serialize data to bytes."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, dict) or isinstance(data, list):
            return Serializer.to_json(data).encode('utf-8')
        else:
            return pickle.dumps(data)

    @staticmethod
    def to_dict(data: Any) -> Dict[str, Any]:
        """Convert any object to dictionary."""
        if isinstance(data, dict):
            return data
        elif hasattr(data, 'to_dict'):
            return data.to_dict()
        elif hasattr(data, '__dict__'):
            return {k: v for k, v in data.__dict__.items() if not k.startswith('_')}
        else:
            return {'value': data}


class ModelSerializer:
    """Specialized serializer for neural network models."""

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> Dict[str, Any]:
        """Serialize PyTorch model to JSON-serializable format."""
        try:
            # Get model state
            state_dict = model.state_dict()

            # Convert tensors to lists
            serialized_state = {}
            for key, tensor in state_dict.items():
                serialized_state[key] = {
                    'dtype': str(tensor.dtype),
                    'shape': list(tensor.shape),
                    'data': tensor.cpu().numpy().tolist()
                }

            # Get model metadata
            metadata = {
                'model_class': model.__class__.__name__,
                'config': getattr(model, 'config', {}),
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': str(next(model.parameters()).device),
                'timestamp': datetime.now().isoformat()
            }

            return {
                'metadata': metadata,
                'state_dict': serialized_state
            }

        except Exception as e:
            logger.error(f"Model serialization failed: {e}")
            raise

    @staticmethod
    def deserialize_model(serialized: Dict[str, Any], model_class: type) -> torch.nn.Module:
        """Deserialize JSON-serialized model."""
        try:
            # Create model instance
            config = serialized['metadata'].get('config', {})
            model = model_class(**config)

            # Reconstruct state dict
            state_dict = {}
            for key, tensor_info in serialized['state_dict'].items():
                data = np.array(tensor_info['data'])
                tensor = torch.from_numpy(data).to(tensor_info['dtype'])
                state_dict[key] = tensor

            # Load state dict
            model.load_state_dict(state_dict)

            return model

        except Exception as e:
            logger.error(f"Model deserialization failed: {e}")
            raise


class CacheSerializer:
    """Serializer for caching purposes."""

    @staticmethod
    def encode_for_cache(data: Any) -> str:
        """Encode data for cache storage."""
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, cls=JSONEncoder)
            compressed = zlib.compress(json_str.encode('utf-8'))
            return base64.b64encode(compressed).decode('utf-8')
        elif isinstance(data, str):
            compressed = zlib.compress(data.encode('utf-8'))
            return base64.b64encode(compressed).decode('utf-8')
        else:
            pickled = pickle.dumps(data)
            compressed = zlib.compress(pickled)
            return base64.b64encode(compressed).decode('utf-8')

    @staticmethod
    def decode_from_cache(encoded: str) -> Any:
        """Decode data from cache."""
        try:
            decoded = base64.b64decode(encoded)
            decompressed = zlib.decompress(decoded)

            # Try to decode as JSON first
            try:
                return json.loads(decompressed.decode('utf-8'))
            except:
                pass

            # Try to decode as string
            try:
                return decompressed.decode('utf-8')
            except:
                pass

            # Fall back to pickle
            return pickle.loads(decompressed)

        except Exception as e:
            logger.error(f"Cache decoding failed: {e}")
            raise