import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship

from langchain.vectorstores.pgvector import BaseModel, CollectionStore


class EmbeddingStore(BaseModel):
    __tablename__ = "langchain_pg_embedding"

    collection_id = sqlalchemy.Column(
        UUID(as_uuid=True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete="CASCADE",
        ),
    )
    collection = relationship(CollectionStore, back_populates="embeddings")

    embedding: Vector = sqlalchemy.Column(Vector(None))
    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
