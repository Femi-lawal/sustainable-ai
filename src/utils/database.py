"""
Database module for storing energy logs and generating transparency reports.
Uses SQLite for local storage with support for export to various formats.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.config import DATABASE_CONFIG, APP_CONFIG
except ImportError:
    from src.utils.config import DATABASE_CONFIG, APP_CONFIG


@dataclass
class EnergyLogEntry:
    """Single energy log entry."""
    id: Optional[int]
    timestamp: str
    prompt_hash: str
    prompt_preview: str
    token_count: int
    complexity_score: float
    energy_kwh: float
    carbon_kg: float
    water_liters: float
    is_anomaly: bool
    was_optimized: bool
    energy_saved_kwh: float
    model_params: str  # JSON
    region: str


class DatabaseManager:
    """
    Manages SQLite database for energy logging and transparency reporting.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Optional custom path for the database
        """
        self.db_path = db_path or DATABASE_CONFIG.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Energy logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt_preview TEXT,
                token_count INTEGER,
                complexity_score REAL,
                energy_kwh REAL NOT NULL,
                carbon_kg REAL,
                water_liters REAL,
                is_anomaly INTEGER DEFAULT 0,
                was_optimized INTEGER DEFAULT 0,
                energy_saved_kwh REAL DEFAULT 0,
                model_params TEXT,
                region TEXT DEFAULT 'default'
            )
        ''')
        
        # Transparency reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transparency_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                generated_at TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_energy_kwh REAL,
                total_carbon_kg REAL,
                total_water_liters REAL,
                total_prompts INTEGER,
                anomaly_count INTEGER,
                optimization_count INTEGER,
                report_data TEXT,
                compliance_status TEXT
            )
        ''')
        
        # Benchmarks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_type TEXT,
                prompt_category TEXT,
                avg_energy_kwh REAL,
                avg_tokens INTEGER,
                sample_count INTEGER,
                percentile_25 REAL,
                percentile_50 REAL,
                percentile_75 REAL,
                percentile_95 REAL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON energy_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_energy ON energy_logs(energy_kwh)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reports_period ON transparency_reports(period_start, period_end)')
        
        conn.commit()
        conn.close()
    
    def log_energy(self, 
                   prompt: str,
                   token_count: int,
                   complexity_score: float,
                   energy_kwh: float,
                   carbon_kg: float = 0,
                   water_liters: float = 0,
                   is_anomaly: bool = False,
                   was_optimized: bool = False,
                   energy_saved_kwh: float = 0,
                   model_params: Dict = None,
                   region: str = "default") -> int:
        """
        Log an energy consumption event.
        
        Returns:
            ID of the inserted log entry
        """
        import hashlib
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Hash the prompt for privacy
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        cursor.execute('''
            INSERT INTO energy_logs 
            (timestamp, prompt_hash, prompt_preview, token_count, complexity_score,
             energy_kwh, carbon_kg, water_liters, is_anomaly, was_optimized,
             energy_saved_kwh, model_params, region)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prompt_hash,
            prompt_preview,
            token_count,
            complexity_score,
            energy_kwh,
            carbon_kg,
            water_liters,
            1 if is_anomaly else 0,
            1 if was_optimized else 0,
            energy_saved_kwh,
            json.dumps(model_params or {}),
            region
        ))
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return log_id
    
    def get_logs(self, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve energy logs within a date range.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM energy_logs'
        params = []
        
        if start_date or end_date:
            query += ' WHERE '
            conditions = []
            
            if start_date:
                conditions.append('timestamp >= ?')
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append('timestamp <= ?')
                params.append(end_date.isoformat())
            
            query += ' AND '.join(conditions)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate statistics for energy consumption.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query_base = '''
            SELECT 
                COUNT(*) as total_prompts,
                SUM(energy_kwh) as total_energy,
                AVG(energy_kwh) as avg_energy,
                MIN(energy_kwh) as min_energy,
                MAX(energy_kwh) as max_energy,
                SUM(carbon_kg) as total_carbon,
                SUM(water_liters) as total_water,
                SUM(is_anomaly) as anomaly_count,
                SUM(was_optimized) as optimization_count,
                SUM(energy_saved_kwh) as total_energy_saved,
                AVG(token_count) as avg_tokens,
                AVG(complexity_score) as avg_complexity
            FROM energy_logs
        '''
        
        params = []
        if start_date or end_date:
            query_base += ' WHERE '
            conditions = []
            if start_date:
                conditions.append('timestamp >= ?')
                params.append(start_date.isoformat())
            if end_date:
                conditions.append('timestamp <= ?')
                params.append(end_date.isoformat())
            query_base += ' AND '.join(conditions)
        
        cursor.execute(query_base, params)
        row = cursor.fetchone()
        conn.close()
        
        if row and row['total_prompts']:
            return {
                "total_prompts": row['total_prompts'],
                "total_energy_kwh": round(row['total_energy'] or 0, 6),
                "average_energy_kwh": round(row['avg_energy'] or 0, 6),
                "min_energy_kwh": round(row['min_energy'] or 0, 6),
                "max_energy_kwh": round(row['max_energy'] or 0, 6),
                "total_carbon_kg": round(row['total_carbon'] or 0, 4),
                "total_water_liters": round(row['total_water'] or 0, 4),
                "anomaly_count": row['anomaly_count'] or 0,
                "anomaly_rate": round((row['anomaly_count'] or 0) / row['total_prompts'] * 100, 2),
                "optimization_count": row['optimization_count'] or 0,
                "total_energy_saved_kwh": round(row['total_energy_saved'] or 0, 6),
                "average_tokens": round(row['avg_tokens'] or 0, 1),
                "average_complexity": round(row['avg_complexity'] or 0, 4)
            }
        
        return {
            "total_prompts": 0,
            "total_energy_kwh": 0,
            "message": "No data available for the specified period"
        }
    
    def generate_transparency_report(self,
                                     period_start: datetime,
                                     period_end: datetime) -> Dict[str, Any]:
        """
        Generate a comprehensive transparency report for EU compliance.
        """
        import uuid
        
        # Get statistics
        stats = self.get_statistics(period_start, period_end)
        logs = self.get_logs(period_start, period_end)
        
        # Generate report ID with UUID for uniqueness
        report_id = f"TR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
        # Calculate percentiles for energy distribution
        if logs:
            energies = [log['energy_kwh'] for log in logs]
            import numpy as np
            percentiles = {
                "p25": round(np.percentile(energies, 25), 6),
                "p50": round(np.percentile(energies, 50), 6),
                "p75": round(np.percentile(energies, 75), 6),
                "p95": round(np.percentile(energies, 95), 6)
            }
        else:
            percentiles = {"p25": 0, "p50": 0, "p75": 0, "p95": 0}
        
        # Determine compliance status
        days_until_deadline = (datetime.strptime(APP_CONFIG.eu_reporting_deadline, "%Y-%m-%d") - datetime.now()).days
        if stats.get('total_prompts', 0) > 0:
            compliance_status = "COMPLIANT"
        else:
            compliance_status = "NO_DATA"
        
        report = {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
                "duration_days": (period_end - period_start).days
            },
            "summary": stats,
            "energy_distribution": percentiles,
            "environmental_impact": {
                "carbon_footprint_kg": stats.get('total_carbon_kg', 0),
                "water_usage_liters": stats.get('total_water_liters', 0),
                "trees_equivalent": round(stats.get('total_carbon_kg', 0) / 21, 2),  # ~21kg CO2 per tree/year
                "homes_powered_hours": round(stats.get('total_energy_kwh', 0) / 1.2, 2)  # ~1.2 kWh/hour avg home
            },
            "optimization_impact": {
                "total_optimizations": stats.get('optimization_count', 0),
                "energy_saved_kwh": stats.get('total_energy_saved_kwh', 0),
                "carbon_avoided_kg": round(stats.get('total_energy_saved_kwh', 0) * 0.42, 4)
            },
            "anomalies": {
                "total_detected": stats.get('anomaly_count', 0),
                "anomaly_rate_percent": stats.get('anomaly_rate', 0)
            },
            "compliance": {
                "standard": "EU AI Act Energy Reporting Requirements",
                "deadline": APP_CONFIG.eu_reporting_deadline,
                "days_until_deadline": days_until_deadline,
                "status": compliance_status,
                "data_completeness_percent": 100 if stats.get('total_prompts', 0) > 0 else 0
            }
        }
        
        # Store the report
        self._store_report(report)
        
        return report
    
    def _store_report(self, report: Dict[str, Any]):
        """Store a transparency report in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO transparency_reports
            (report_id, generated_at, period_start, period_end, total_energy_kwh,
             total_carbon_kg, total_water_liters, total_prompts, anomaly_count,
             optimization_count, report_data, compliance_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report['report_id'],
            report['generated_at'],
            report['period']['start'],
            report['period']['end'],
            report['summary'].get('total_energy_kwh', 0),
            report['environmental_impact']['carbon_footprint_kg'],
            report['environmental_impact']['water_usage_liters'],
            report['summary'].get('total_prompts', 0),
            report['anomalies']['total_detected'],
            report['optimization_impact']['total_optimizations'],
            json.dumps(report),
            report['compliance']['status']
        ))
        
        conn.commit()
        conn.close()
    
    def get_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve stored transparency reports."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM transparency_reports
            ORDER BY generated_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        reports = []
        for row in rows:
            report = dict(row)
            if report.get('report_data'):
                report['full_report'] = json.loads(report['report_data'])
            reports.append(report)
        
        return reports
    
    def get_hourly_breakdown(self, date: datetime) -> List[Dict[str, Any]]:
        """Get hourly energy breakdown for a specific date."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as prompt_count,
                SUM(energy_kwh) as total_energy,
                AVG(energy_kwh) as avg_energy
            FROM energy_logs
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''', (start.isoformat(), end.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def export_to_csv(self, filepath: Path, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None):
        """Export logs to CSV file."""
        import csv
        
        logs = self.get_logs(start_date, end_date)
        
        if not logs:
            return
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
    
    def clear_old_logs(self, days_to_keep: int = 90):
        """Clear logs older than specified days."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute('DELETE FROM energy_logs WHERE timestamp < ?', (cutoff,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted


# Factory function
def get_database() -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager()


# Convenience functions
def log_prompt_energy(prompt: str, energy_kwh: float, **kwargs) -> int:
    """Quick function to log energy for a prompt."""
    db = DatabaseManager()
    return db.log_energy(prompt, energy_kwh=energy_kwh, **kwargs)


def get_energy_statistics(days: int = 30) -> Dict[str, Any]:
    """Get energy statistics for the last N days."""
    db = DatabaseManager()
    start_date = datetime.now() - timedelta(days=days)
    return db.get_statistics(start_date=start_date)


if __name__ == "__main__":
    # Test the database
    db = DatabaseManager()
    
    # Log some test entries
    print("Logging test entries...")
    for i in range(5):
        db.log_energy(
            prompt=f"Test prompt {i}",
            token_count=50 + i * 10,
            complexity_score=0.3 + i * 0.1,
            energy_kwh=0.1 + i * 0.05,
            carbon_kg=0.04 + i * 0.02,
            water_liters=0.2 + i * 0.1,
            is_anomaly=(i == 4),
            was_optimized=(i % 2 == 0),
            energy_saved_kwh=0.02 if i % 2 == 0 else 0
        )
    
    # Get statistics
    print("\nStatistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate report
    print("\nGenerating transparency report...")
    report = db.generate_transparency_report(
        datetime.now() - timedelta(days=7),
        datetime.now()
    )
    print(f"Report ID: {report['report_id']}")
    print(f"Compliance Status: {report['compliance']['status']}")
