"""Library module for IMS reference compound library."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from .schemas import LibraryCompound
from .utils import get_library_path

class LibraryManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or get_library_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS library_compounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    family TEXT,
                    gas TEXT,
                    z INTEGER,
                    mz REAL,
                    K0 REAL,
                    notes TEXT,
                    created_utc INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
    
    def add_compound(self, compound: LibraryCompound) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO library_compounds (name, family, gas, z, mz, K0, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (compound.name, compound.family, compound.gas, 
                  compound.z, compound.mz, compound.K0, compound.notes))
            return cursor.lastrowid
    
    def get_compounds(self, family=None, gas=None, name_filter=None):
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM library_compounds WHERE 1=1"
            params = []
            
            if family:
                query += " AND family LIKE ?"
                params.append(f"%{family}%")
            
            if gas:
                query += " AND gas = ?"
                params.append(gas)
            
            if name_filter:
                query += " AND name LIKE ?"
                params.append(f"%{name_filter}%")
            
            query += " ORDER BY name"
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    
    def import_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        imported = 0
        
        for _, row in df.iterrows():
            compound = LibraryCompound(
                name=str(row.get('name', 'Unknown')),
                family=str(row.get('family', '')),
                gas=str(row.get('gas', 'N2')),
                z=int(row.get('z', 1)),
                mz=float(row.get('mz', 0)),
                K0=float(row.get('K0', 0)),
                notes=str(row.get('notes', ''))
            )
            self.add_compound(compound)
            imported += 1
        
        return imported
    
    def export_csv(self, output_path):
        compounds = self.get_compounds()
        if not compounds:
            return 0
        
        df = pd.DataFrame(compounds)
        df.to_csv(output_path, index=False)
        return len(compounds)