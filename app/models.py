from sqlalchemy import Column, String, Float, Integer, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True, index=True)
    product_name = Column(String, index=True)
    category = Column(String)  # Raw category string
    discounted_price = Column(Float)
    actual_price = Column(Float)
    discount_percentage = Column(Float)
    rating = Column(Float)
    rating_count = Column(Integer)
    about_product = Column(Text)
    img_link = Column(String)
    product_link = Column(String)
    
    # Relationships
    wishlist_entries = relationship("Wishlist", back_populates="product")
    recently_viewed = relationship("RecentlyViewed", back_populates="product")


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    
    # Allow hierarchical category lookups
    parent = relationship("Category", remote_side=[id])


class Brand(Base):
    __tablename__ = "brands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)


class TrainingHistory(Base):
    __tablename__ = "training_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True, nullable=True)
    training_date = Column(DateTime, default=func.now())
    duration_seconds = Column(Float)
    vocab_size = Column(Integer)
    dataset_rows = Column(Integer)
    metrics_precision_at_5 = Column(Float, nullable=True)
    metrics_coverage = Column(Float, nullable=True)
    model_version = Column(String)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    source_product_id = Column(String, ForeignKey("products.product_id"), nullable=True)
    recommended_product_id = Column(String, ForeignKey("products.product_id"))
    engine = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    clicked = Column(Integer, default=0)  # 0 = False, 1 = True
    purchased = Column(Integer, default=0)


class Wishlist(Base):
    __tablename__ = "wishlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    product_id = Column(String, ForeignKey("products.product_id"))
    added_at = Column(DateTime, default=func.now())

    product = relationship("Product", back_populates="wishlist_entries")


class RecentlyViewed(Base):
    __tablename__ = "recently_viewed"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    product_id = Column(String, ForeignKey("products.product_id"))
    viewed_at = Column(DateTime, default=func.now())

    product = relationship("Product", back_populates="recently_viewed")
