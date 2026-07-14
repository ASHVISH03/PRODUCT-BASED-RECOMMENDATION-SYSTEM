import sys
import os
from pathlib import Path

# Ensure project root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.database import engine, Base, SessionLocal
from app.models import Product

# Print formatting
def header(msg: str):
    print(f"\n{'-'*60}\n  {msg}\n{'-'*60}")

def check(condition: bool, message: str) -> bool:
    if condition:
        print(f"  [PASS] {message}")
        return True
    else:
        print(f"  [FAIL] {message}")
        return False

client = TestClient(app)

def test_health():
    header("TEST 1: API Health Check")
    response = client.get("/api/v1/health")
    success = check(response.status_code == 200, "Health endpoint returns 200 OK")
    if success:
        data = response.json()
        check(data.get("status") == "ok", "Status is 'ok'")

    return success

def test_database():
    header("TEST 2: Database and ORM")
    success = True
    try:
        db = SessionLocal()
        product_count = db.query(Product).count()
        success &= check(product_count > 0, f"Database seeded with {product_count} products")
    except Exception as e:
        success = check(False, f"Database query failed: {e}")
    finally:
        db.close()
    return success

def test_products_endpoints():
    header("TEST 3: Products API Endpoints")
    success = True
    
    # Test GET /products
    response = client.get("/api/v1/products?limit=5")
    success &= check(response.status_code == 200, "GET /products returns 200 OK")
    
    if response.status_code == 200:
        data = response.json()
        success &= check(data.get("count") == 5, "Pagination limit works")
        success &= check(len(data.get("data", [])) > 0, "Products data returned")
        
        # Test GET /products/{id}
        if data.get("data"):
            test_product = data["data"][0]
            prod_id = test_product["product_id"]
            resp2 = client.get(f"/api/v1/products/{prod_id}")
            success &= check(resp2.status_code == 200, "GET /products/{id} returns 200 OK")
            
    # Test GET /products/categories
    response = client.get("/api/v1/products/categories")
    success &= check(response.status_code == 200, "GET /products/categories returns 200 OK")
    
    return success

def test_search_endpoint():
    header("TEST 4: Search API Endpoint")
    success = True
    
    response = client.get("/api/v1/search?q=samsung&limit=3")
    success &= check(response.status_code == 200, "GET /search returns 200 OK")
    if response.status_code == 200:
        data = response.json()
        success &= check("data" in data, "Search returns data array")
        
    return success

def test_recommendations_endpoint():
    header("TEST 5: Recommendations API Endpoint")
    success = True
    
    # First get a valid product ID
    db = SessionLocal()
    try:
        product = db.query(Product).first()
        if not product:
            return check(False, "No product found to test recommendations")
            
        prod_id = product.product_id
    finally:
        db.close()
        
    response = client.get(f"/api/v1/recommendations/{prod_id}?strategy=hybrid&k=3")
    success &= check(response.status_code == 200, "GET /recommendations/{id} returns 200 OK")
    if response.status_code == 200:
        data = response.json()
        success &= check(data.get("count") == 3, "Recommendations count matches 'k' parameter")
        success &= check("reason" in data["data"][0], "Explainability reasoning is included")
        
    return success

if __name__ == "__main__":
    print("\n============================================================")
    print("  PHASE 4 VERIFICATION SUITE")
    print("  Backend API")
    print("============================================================")
    
    # Make sure DB is seeded before testing
    from scripts.init_db import init_db
    print("\nInitializing database for tests...")
    init_db(seed_data=True)
    
    tests = [
        test_health,
        test_database,
        test_products_endpoints,
        test_search_endpoint,
        test_recommendations_endpoint
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
            
    print("\n============================================================")
    print("  PHASE 4 VERIFICATION SUMMARY")
    print("============================================================")
    print(f"  {passed}/{len(tests)} tests passed")
    print("============================================================\n")
    
    if passed == len(tests):
        sys.exit(0)
    else:
        sys.exit(1)
