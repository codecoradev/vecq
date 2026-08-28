# Storing a vecq index inside SQLite (BLOB pattern)

vecq's first consumers are SQLite-native projects (uteke's memory engine,
cora-code's symbol index). Keeping the index **inside** the database — instead
of a `.vecq` file next to it — means backups, migrations, `ATTACH`, and
per-project multi-tenancy see the index as ordinary data. This page is the
canonical pattern so downstream projects converge on one approach.

## When to embed vs standalone file

| consideration | BLOB in SQLite | standalone file |
|---|---|---|
| backup/migration tooling | included automatically (`sqlite3 .backup`, VACUUM INTO) | must be handled separately — easy to miss |
| multiple indexes in one app | one row per index/project | one file per index, manual registry |
| remote DBs (Postgres/Redis) | readable via any SQL client | needs file transport |
| very large indexes (> ~100 MB) | DB file grows fast, VACUUM gets expensive | same cost, plus mmap-friendly random access |
| concurrent writers from multiple processes | serialized by SQLite | needs your own file locking |

**Rule of thumb:** embed up to a few tens of MB (≈ 100k vectors at dim 768);
beyond that, prefer the standalone file and register its path in the DB.

## Schema

Two shapes, pick one:

```sql
-- Simple: single-row table, one index per database (or per tenant via WHERE).
CREATE TABLE vecq_index (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    dim  INTEGER NOT NULL,
    seed INTEGER NOT NULL,          -- must be persisted (from index.seed())
    blob BLOB NOT NULL              -- output of VecqIndex::to_bytes()
);

-- Scalable: per-shard rows (e.g. shard by key prefix or time bucket), so an
-- incremental update only rewrites the affected shard's BLOB.
CREATE TABLE vecq_shard (
    shard TEXT PRIMARY KEY,
    dim   INTEGER NOT NULL,
    seed  INTEGER NOT NULL,
    blob  BLOB NOT NULL
);
```

Keys from the keyed API (`add_keyed`/`remove_keyed`) are **not** part of the
file format — persist your own mapping table next to the index:

```sql
CREATE TABLE vecq_keys (
    key  INTEGER PRIMARY KEY,       -- the u64 key used in add_keyed()
    slot INTEGER NOT NULL           -- slot index, valid until compact()
);
```

Because `to_bytes()` drops tombstones and re-serializes live slots in order,
**slot indices change whenever you save-then-reload after a `compact()`** —
rewrite `vecq_keys` in the same transaction whenever you save the index (see
below), or simply resolve keys → search results instead of storing slots.

## Canonical save/load (Rust + rusqlite)

```rust
use rusqlite::Connection;
use vecq_core::VecqIndex;

fn save_index(db: &Connection, index: &VecqIndex) -> rusqlite::Result<()> {
    let blob = index.to_bytes();
    db.execute_batch("BEGIN IMMEDIATE")?;
    db.execute(
        "INSERT INTO vecq_index (id, dim, seed, blob) VALUES (1, ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET dim = ?1, seed = ?2, blob = ?3",
        rusqlite::params![index.dim(), index.seed(), blob],
    )?;
    db.execute_batch("COMMIT")?;
    Ok(())
}

fn load_index(db: &Connection) -> rusqlite::Result<Option<VecqIndex>> {
    let mut stmt = db.prepare("SELECT dim, seed, blob FROM vecq_index WHERE id = 1")?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        let blob: Vec<u8> = row.get(2)?;
        let index = VecqIndex::from_bytes(&blob)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        return Ok(Some(index));
    }
    Ok(None)
}
```

## Atomicity: keep it boring

A single transactional `UPDATE` of the BLOB is already atomic and crash-safe —
SQLite journaling (rollback or WAL) guarantees readers see either the old or
the new index, never a torn file. **No manual write-ahead-BLOB dance or
`wal_checkpoint` choreography is needed for correctness**; checkpointing only
controls the WAL file's size.

Recommended pragmas:

```sql
PRAGMA journal_mode = WAL;       -- readers don't block the save
PRAGMA synchronous = NORMAL;     -- safe under WAL; FULL is rarely needed here
```

Save at most once per batch of mutations (not per add/remove): `to_bytes()` is
O(n) over the live vectors.

## Size and latency

Bytes per vector = `padded_dim/2 + 2` (nibble codes + f16 scale, format v1.1).
For dim 768 (padded 1024): **514 B/vector**.

| vectors | dim 768 BLOB | save (update+commit) | load (read BLOB) |
|---|---|---|---|
| 1,000 | 0.5 MB | ~0.3 ms | ~0.1 ms |
| 10,000 | 5.0 MB | ~2 ms | ~3 ms |
| 50,000 | 25 MB | ~28 ms | ~10 ms |

Measured on an M-series MacBook (Python `sqlite3`, WAL, `synchronous=NORMAL`)
— treat as order-of-magnitude for commodity hardware. The point: for the
10k-scale workloads vecq targets, "save the whole index in one BLOB update per
batch" is comfortably fast, and per-shard rows are only worth it past ~50 MB.

## Pitfalls

- **Memory duplication on load**: `row.get::<_, Vec<u8>>` copies the BLOB,
  then `from_bytes` copies into the index's codes/scales. Transient 2x is
  fine at 5 MB; load once at startup, not per query, for large indexes.
- **Slot instability across save/reload**: in-memory slot indices stay stable
  across tombstones (issue #10 design), but a save→reload produces a fresh
  dense index — any stored slot references must be rewritten in the same
  transaction as the BLOB save.
- **Churn-heavy workloads**: every save rewrites the full BLOB. If you mutate
  thousands of times per second, shard the index (per-shard rows) or debounce
  saves.
- **`PRAGMA integrity_check` cost**: giant BLOBs make integrity checks and
  `VACUUM` slow; another reason for the ~100 MB embed threshold.
- **Seed discipline**: always store `seed` next to the BLOB; a rebuilt index
  with a different seed produces a different rotation and incompatible scores.
