from platform import architecture

import psycopg2
from config import POSTGRES
import logging
import json
logger = logging.getLogger("uvicorn.error")
class PostgresClient:

    def __init__(self):
        self.conn = psycopg2.connect(**POSTGRES)
        self.init_tables()

    def init_tables(self):
        cur = self.conn.cursor()
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id VARCHAR(100) PRIMARY KEY,
                architecture_type VARCHAR(50),
                facts JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_facts_gin ON user_profile USING GIN(facts);
            """
            cur.execute(sql)
            self.conn.commit()
            logger.info("Tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing tables: {e}")
            self.conn.rollback()

        finally:
            cur.close()

    def upsert_dict(self, user_id, facts):
        try:
            architecture_type = facts.pop("architecture_type", None)
            sql = """
            INSERT INTO user_profile(user_id, architecture_type, facts, updated_at)
            VALUES (%s, %s, %s::jsonb, NOW())
            ON CONFLICT(user_id)
            DO UPDATE SET 
            architecture_type = COALESCE(EXCLUDED.architecture_type, user_profile.architecture_type),
            facts = user_profile.facts || EXCLUDED.facts,
            updated_at = NOW()
            """
            facts_json = json.dumps(facts)
            cur = self.conn.cursor()
            cur.execute(sql, (user_id, architecture_type, facts_json))
            self.conn.commit()
            logger.info(f"Storge {facts} result for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing hybird result: {e}")
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def upsert(self, user_id, key, value):

        sql = f"""
        INSERT INTO user_profile(user_id,{key})
        VALUES (%s,%s)
        ON CONFLICT(user_id)
        DO UPDATE SET {key}=EXCLUDED.{key},
        updated_at=NOW()
        """
        logger.info("upsert:", sql)
        cur = self.conn.cursor()
        cur.execute(sql, (user_id, value))
        self.conn.commit()

    def get(self, user_id):

        cur = self.conn.cursor()
        sql = f"""
        SELECT facts
        FROM user_profile
        WHERE user_id=%s
        """
        cur.execute(sql, (user_id,))

        row = cur.fetchone()
        #logger.info("row:..............", row)
        if not row:
            return {}

        return row[0]