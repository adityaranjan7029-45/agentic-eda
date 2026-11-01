"""runs.stage: latest progress stage (step 12, live progress)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("stage", sa.String(32)))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("stage")
