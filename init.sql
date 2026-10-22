CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    gpu TEXT,
    os TEXT,
    mpi TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);