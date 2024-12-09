import uuid
from typing import Any, List, Optional, Tuple

import sqlalchemy
from sqlalchemy.orm import Session, declarative_base, relationship


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ModelFactory:
    """Provide SQLAlchemy model objects at runtime."""

    def __init__(self, dimensions: Optional[int] = None):
        from sqlalchemy_cratedb import FloatVector, ObjectType

        # While it does not have any function here, you will still need to supply a
        # dummy dimension size value for operations like deleting records.
        self.dimensions = dimensions or 1024

        Base: Any = declarative_base()

        # Optional: Use a custom schema for the langchain tables.
        # Base = declarative_base(metadata=MetaData(schema="langchain"))  # type: Any

        class BaseModel(Base):
            """Base model for the SQL stores."""

            __abstract__ = True
            uuid = sqlalchemy.Column(
                sqlalchemy.String, primary_key=True, default=generate_uuid
            )

        class CollectionStore(BaseModel):
            """Collection store."""

            __tablename__ = "collection"
            __table_args__ = {"keep_existing": True}

            name = sqlalchemy.Column(sqlalchemy.String)
            cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType)

            embeddings = relationship(
                "EmbeddingStore",
                back_populates="collection",
                cascade="all, delete-orphan",
                passive_deletes=False,
            )

            @classmethod
            def get_by_name(
                cls, session: Session, name: str
            ) -> Optional["CollectionStore"]:
                return session.query(cls).filter(cls.name == name).first()  # type: ignore[attr-defined]

            @classmethod
            def get_by_names(
                cls, session: Session, names: List[str]
            ) -> List["CollectionStore"]:
                return session.query(cls).filter(cls.name.in_(names)).all()  # type: ignore[attr-defined]

            @classmethod
            def get_or_create(
                cls,
                session: Session,
                name: str,
                cmetadata: Optional[dict] = None,
            ) -> Tuple["CollectionStore", bool]:
                """
                Get or create a collection.
                Returns [Collection, bool] where the bool is True
                if the collection was created.
                """
                created = False
                collection = cls.get_by_name(session, name)
                if collection:
                    return collection, created

                collection = cls(name=name, cmetadata=cmetadata)
                session.add(collection)
                session.commit()
                created = True
                return collection, created

        class EmbeddingStore(BaseModel):
            """Embedding store."""

            __tablename__ = "embedding"
            __table_args__ = {"keep_existing": True}

            collection_id = sqlalchemy.Column(
                sqlalchemy.String,
                sqlalchemy.ForeignKey(
                    f"{CollectionStore.__tablename__}.uuid",
                    ondelete="CASCADE",
                ),
            )
            collection = relationship("CollectionStore", back_populates="embeddings")

            embedding: sqlalchemy.Column = sqlalchemy.Column(
                FloatVector(self.dimensions)
            )
            document: sqlalchemy.Column = sqlalchemy.Column(
                sqlalchemy.String, nullable=True
            )
            cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType, nullable=True)

            # custom_id : any user defined id
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

        self.Base = Base
        self.BaseModel = BaseModel
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
