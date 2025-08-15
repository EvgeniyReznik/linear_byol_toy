from __future__ import annotations
import os, json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetLogger:
    """
    Append rows (dicts) to a parquet file with a fixed schema, creating it on first write.
    Vectors/lists are stored as JSON strings to keep schema simple/portable.
    """
    def __init__(self, path: str):
        self.path = path
        self._writer = None
        self._schema = None

    def _infer_schema(self, row: dict):
        fields = []
        for k, v in row.items():
            if isinstance(v, bool):   t = pa.bool_()
            elif isinstance(v, int):  t = pa.int64()
            elif isinstance(v, float):t = pa.float64()
            else:                     t = pa.string()
            fields.append(pa.field(k, t))
        self._schema = pa.schema(fields)

    def write_row(self, row: dict):
        # coerce lists/tuples/dicts to JSON
        norm = {}
        for k, v in row.items():
            if isinstance(v, (list, tuple, dict)):
                norm[k] = json.dumps(v)
            else:
                norm[k] = v
        if self._writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._infer_schema(norm)
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")
        arrays = [pa.array([norm[k]], type=self._schema.field(k).type) for k in self._schema.names]
        table = pa.Table.from_arrays(arrays, names=self._schema.names)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
