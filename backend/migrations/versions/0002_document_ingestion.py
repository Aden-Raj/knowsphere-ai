"""Add document ingestion metadata and lifecycle statuses.

Revision ID: 0002_document_ingestion
Revises: 0001_initial
"""
from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0002_document_ingestion"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE TYPE document_status_new AS ENUM ('uploaded', 'processing', 'completed', 'failed')")
    op.execute("ALTER TABLE knowledge_documents ALTER COLUMN status DROP DEFAULT")
    op.execute(
        "ALTER TABLE knowledge_documents ALTER COLUMN status TYPE document_status_new "
        "USING (CASE status::text WHEN 'pending' THEN 'uploaded' "
        "WHEN 'ready' THEN 'completed' ELSE status::text END)::document_status_new"
    )
    op.execute("DROP TYPE document_status")
    op.execute("ALTER TYPE document_status_new RENAME TO document_status")
    op.execute("ALTER TABLE knowledge_documents ALTER COLUMN status SET DEFAULT 'uploaded'")

    op.add_column("knowledge_documents", sa.Column("original_filename", sa.String(512), nullable=True))
    op.add_column("knowledge_documents", sa.Column("storage_path", sa.Text(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("file_extension", sa.String(20), nullable=True))
    op.add_column("knowledge_documents", sa.Column("file_size_bytes", sa.BigInteger(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False))
    op.add_column("knowledge_documents", sa.Column("embedding_model", sa.String(255), nullable=True))
    op.add_column(
        "knowledge_documents",
        sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.alter_column("knowledge_documents", "failure_reason", new_column_name="error_message")


def downgrade() -> None:
    op.alter_column("knowledge_documents", "error_message", new_column_name="failure_reason")
    op.drop_column("knowledge_documents", "processing_completed_at")
    op.drop_column("knowledge_documents", "embedding_model")
    op.drop_column("knowledge_documents", "chunk_count")
    op.drop_column("knowledge_documents", "file_size_bytes")
    op.drop_column("knowledge_documents", "file_extension")
    op.drop_column("knowledge_documents", "storage_path")
    op.drop_column("knowledge_documents", "original_filename")

    op.execute("CREATE TYPE document_status_old AS ENUM ('pending', 'processing', 'ready', 'failed')")
    op.execute("ALTER TABLE knowledge_documents ALTER COLUMN status DROP DEFAULT")
    op.execute(
        "ALTER TABLE knowledge_documents ALTER COLUMN status TYPE document_status_old "
        "USING (CASE status::text WHEN 'uploaded' THEN 'pending' "
        "WHEN 'completed' THEN 'ready' ELSE status::text END)::document_status_old"
    )
    op.execute("DROP TYPE document_status")
    op.execute("ALTER TYPE document_status_old RENAME TO document_status")
    op.execute("ALTER TABLE knowledge_documents ALTER COLUMN status SET DEFAULT 'pending'")
