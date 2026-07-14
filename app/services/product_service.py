from sqlalchemy.orm import Session
from app.models import Product
from typing import List, Optional

class ProductService:
    def __init__(self, db: Session):
        self.db = db

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.db.query(Product).filter(Product.product_id == product_id).first()

    def get_products(self, skip: int = 0, limit: int = 50, category: Optional[str] = None) -> List[Product]:
        query = self.db.query(Product)
        if category:
            query = query.filter(Product.category.contains(category))
        return query.offset(skip).limit(limit).all()

    def get_product_categories(self) -> List[str]:
        # Using simple raw categories for the API
        categories = self.db.query(Product.category).distinct().all()
        
        # Clean up and split the hierarchical category strings
        unique_cats = set()
        for cat in categories:
            if cat[0]:
                parts = str(cat[0]).split('|')
                for part in parts:
                    unique_cats.add(part.strip())
                    
        return sorted(list(unique_cats))
