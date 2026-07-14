import sys
import os
from pathlib import Path
import pandas as pd

# Ensure project root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import engine, Base, SessionLocal
from app.models import Product, Category, Brand
from src.logger.database_logger import get_database_logger

logger = get_database_logger()

def init_db(seed_data: bool = True):
    logger.info("Initializing SQLite database...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Created database tables successfully.")
    
    if not seed_data:
        return

    db = SessionLocal()
    
    try:
        # Check if products already exist
        existing_count = db.query(Product).count()
        if existing_count > 0:
            msg = f"Database already seeded with {existing_count} products. Skipping seed."
            logger.info(msg)
            print(f"  [init_db] {msg}")
            return

        # Resolve paths relative to project root
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / "data/raw/amazon.csv"
        
        if not csv_path.exists():
            csv_path = project_root / "data/processed/cleaned.csv"
            
        if not csv_path.exists():
            msg = f"Could not find dataset at {csv_path} for seeding."
            logger.warning(msg)
            print(f"  [init_db] {msg}")
            return
            
        msg = f"Seeding database from {csv_path}..."
        logger.info(msg)
        print(f"  [init_db] {msg}")
        
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"  [init_db] Read {len(df)} rows from CSV.")
        
        products = []
        seen_ids = set()
        
        for _, row in df.iterrows():
            product_id = str(row.get('product_id'))
            
            # Basic validation and duplicate check
            if pd.isna(row.get('product_id')) or pd.isna(row.get('product_name')) or product_id in seen_ids:
                continue
                
            seen_ids.add(product_id)
                
            product = Product(
                product_id=product_id,
                product_name=str(row.get('product_name', '')),
                category=str(row.get('category', '')),
                discounted_price=float(str(row.get('discounted_price', '0')).replace('₹', '').replace(',', '')) if pd.notna(row.get('discounted_price')) else 0.0,
                actual_price=float(str(row.get('actual_price', '0')).replace('₹', '').replace(',', '')) if pd.notna(row.get('actual_price')) else 0.0,
                discount_percentage=float(str(row.get('discount_percentage', '0')).replace('%', '')) if pd.notna(row.get('discount_percentage')) else 0.0,
                rating=float(row.get('rating', 0.0)) if pd.notna(row.get('rating')) and str(row.get('rating')).replace('.','',1).isdigit() else 0.0,
                rating_count=int(str(row.get('rating_count', '0')).replace(',', '')) if pd.notna(row.get('rating_count')) and str(row.get('rating_count')).replace(',','').isdigit() else 0,
                about_product=str(row.get('about_product', '')),
                img_link=str(row.get('img_link', '')),
                product_link=str(row.get('product_link', ''))
            )
            products.append(product)
            
        # Bulk insert
        msg = f"Inserting {len(products)} unique products into database..."
        logger.info(msg)
        print(f"  [init_db] {msg}")
        
        db.bulk_save_objects(products)
        db.commit()
        
        final_count = db.query(Product).count()
        msg = f"Database seeding completed successfully. Final count: {final_count} products."
        logger.info(msg)
        print(f"  [init_db] {msg}")

    except Exception as e:
        msg = f"Error seeding database: {e}"
        logger.error(msg)
        print(f"  [init_db] {msg}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db(seed_data=True)
