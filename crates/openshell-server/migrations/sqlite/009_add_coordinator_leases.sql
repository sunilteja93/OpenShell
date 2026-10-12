-- Elected coordinator leases. One row per lease name; never deleted, so the
-- term only increases. Fenced writes check holder, term and expiry in the
-- same transaction as the guarded write.
CREATE TABLE IF NOT EXISTS coordinator_leases (
    name          TEXT PRIMARY KEY,
    holder        TEXT NOT NULL,
    term          BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL
);
