"""
Database setup and connection management for Stroke Lab Backend

Provides SQLAlchemy engine, session management, and database utilities.
"""

from typing import Generator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging

from app.core.config import get_settings, get_database_settings

# Get configuration
settings = get_settings()
db_settings = get_database_settings()

# Create SQLAlchemy engine
if db_settings.is_sqlite:
    # SQLite configuration with optimizations
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args=db_settings.connect_args,
        poolclass=StaticPool,
        echo=settings.DEBUG,  # Log SQL queries in debug mode
    )
else:
    # PostgreSQL or other database configuration
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
    )

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create Base class for declarative models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

# Logger
logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_context() as db:
            # Use db session
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables():
    """Drop all database tables (use with caution!)."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def reset_database():
    """Reset the database by dropping and recreating all tables."""
    logger.warning("Resetting database - all data will be lost!")
    drop_tables()
    create_tables()


def init_database():
    """Initialize the database with default data."""
    try:
        create_tables()
        
        # Add any default data here
        with get_db_context() as db:
            # Example: Create default admin user, sample data, etc.
            pass
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_table_info(self) -> dict:
        """Get information about database tables."""
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        
        table_info = {}
        for table in tables:
            columns = inspector.get_columns(table)
            table_info[table] = {
                'columns': [col['name'] for col in columns],
                'column_count': len(columns)
            }
        
        return table_info
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database (SQLite only)."""
        if not db_settings.is_sqlite:
            raise NotImplementedError("Backup is only supported for SQLite databases")
        
        import shutil
        from pathlib import Path
        
        try:
            # Extract database file path from URL
            db_path = settings.DATABASE_URL.replace("sqlite:///", "")
            if Path(db_path).exists():
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
            else:
                logger.warning("Database file not found for backup")
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def restore_database(self, backup_path: str):
        """Restore database from backup (SQLite only)."""
        if not db_settings.is_sqlite:
            raise NotImplementedError("Restore is only supported for SQLite databases")
        
        import shutil
        from pathlib import Path
        
        try:
            db_path = settings.DATABASE_URL.replace("sqlite:///", "")
            if Path(backup_path).exists():
                shutil.copy2(backup_path, db_path)
                logger.info(f"Database restored from {backup_path}")
            else:
                logger.error(f"Backup file not found: {backup_path}")
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


# Health check functions
def health_check() -> dict:
    """Perform a health check on the database."""
    try:
        is_connected = db_manager.check_connection()
        table_info = db_manager.get_table_info() if is_connected else {}
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "database_type": "sqlite" if db_settings.is_sqlite else "postgresql",
            "tables": list(table_info.keys()),
            "table_count": len(table_info)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False
        }


# Utility functions for common operations
def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query and return results."""
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, params or {})
            return result.fetchall()
    except Exception as e:
        logger.error(f"Error executing raw SQL: {e}")
        raise


def get_table_row_count(table_name: str) -> int:
    """Get the number of rows in a table."""
    try:
        sql = f"SELECT COUNT(*) FROM {table_name}"
        result = execute_raw_sql(sql)
        return result[0][0] if result else 0
    except Exception as e:
        logger.error(f"Error getting row count for table {table_name}: {e}")
        return 0


# Migration helpers
def check_migration_needed() -> bool:
    """Check if database migration is needed."""
    # This would typically check against Alembic migration versions
    # For now, we'll do a simple table existence check
    try:
        table_info = db_manager.get_table_info()
        expected_tables = ['stroke_data', 'trained_models', 'training_sessions']
        
        for table in expected_tables:
            if table not in table_info:
                return True
        
        return False
    except Exception:
        return True


def perform_migration():
    """Perform database migration if needed."""
    if check_migration_needed():
        logger.info("Database migration needed, creating tables...")
        create_tables()
    else:
        logger.info("Database schema is up to date")
