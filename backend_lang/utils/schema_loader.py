import json
import os
from typing import Dict, Any, List
from pydantic import BaseModel, create_model, Field

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schemas')

class OperationSchema:
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.operation = raw['operation']
        self.description = raw.get('description', '')
        self.endpoint = raw.get('endpoint')
        self.method = raw.get('method', 'POST').upper()
        self.fields = raw.get('fields', {})

    def build_pydantic_model(self) -> BaseModel:
        field_defs = {}
        for name, meta in self.fields.items():
            py_type = str
            t = meta.get('type')
            if t == 'integer':
                py_type = int
            elif t == 'number':
                py_type = float
            elif t == 'array':
                # simplistic: assume array of strings unless specified
                item_type = str
                if isinstance(meta.get('items'), dict):
                    if meta['items'].get('type') == 'integer':
                        item_type = int
                py_type = List[item_type]  # type: ignore
            required = meta.get('required', False)
            default = ... if required else None
            field_defs[name] = (py_type, Field(default, description=meta.get('description'), examples=meta.get('examples')))
        model = create_model(f"Op_{self.operation}_Model", **field_defs)  # type: ignore
        return model


def load_operation_schemas() -> Dict[str, OperationSchema]:
    schemas: Dict[str, OperationSchema] = {}
    for fname in os.listdir(SCHEMA_DIR):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(SCHEMA_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        op_schema = OperationSchema(raw)
        schemas[op_schema.operation] = op_schema
    return schemas
