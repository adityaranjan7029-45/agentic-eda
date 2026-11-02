"""users: identify by login provider + account id (step 13, Auth.js)

Existing users came from the dev header; they become provider "dev" with their
email as the subject. Email is no longer unique (an address can move between
accounts) and may be missing (private GitHub emails).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-02
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("auth_provider", sa.String(32)))
        batch.add_column(sa.Column("auth_subject", sa.String(255)))
        batch.add_column(sa.Column("name", sa.String(255)))
    op.execute("UPDATE users SET auth_provider = 'dev', auth_subject = email")
    with op.batch_alter_table("users") as batch:
        batch.alter_column("auth_provider", existing_type=sa.String(32), nullable=False)
        batch.alter_column("auth_subject", existing_type=sa.String(255), nullable=False)
        batch.alter_column("email", existing_type=sa.String(320), nullable=True)
        batch.drop_index("ix_users_email")
        batch.create_index("ix_users_email", ["email"], unique=False)
        batch.create_unique_constraint("uq_users_identity", ["auth_provider", "auth_subject"])


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("uq_users_identity", type_="unique")
        batch.drop_index("ix_users_email")
        batch.create_index("ix_users_email", ["email"], unique=True)
        batch.alter_column("email", existing_type=sa.String(320), nullable=False)
        batch.drop_column("name")
        batch.drop_column("auth_subject")
        batch.drop_column("auth_provider")
