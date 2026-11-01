"""users, runs, reports

Revision ID: 0001
Revises:
Create Date: 2026-11-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("n_rows", sa.Integer()),
        sa.Column("n_columns", sa.Integer()),
        sa.Column("target", sa.String(255)),
        sa.Column("narrative_source", sa.String(16)),
        sa.Column("verification_status", sa.String(32)),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("key_source", sa.String(16)),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_runs_user_id", "runs", ["user_id"])
    op.create_index("ix_runs_created_at", "runs", ["created_at"])

    op.create_table(
        "reports",
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("data", JSONType, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_index("ix_runs_user_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
