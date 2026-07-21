import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.recommendation_service import RecommendationService

REAL_BROWSER_HISTORY_IDS = [
    "B0B4HJNPV4", "B0B1YZX72F", "B08L879JSN", "B0B9959XF3", "B0BC9BW512",
    "B0B2RBP83P", "B09PTT8DZF", "B08LW31NQ6", "B09P22HXH6", "B0B1YY6JJL",
    "B0B997FBZT", "B00EDJJ7FS", "B08BJN4MP3", "B09XXZXQC1", "B096MSW6CT"
]

db = SessionLocal()
try:
    print("Testing RecommendationService.get_personalized_recommendations...")
    svc = RecommendationService(db)
    res = svc.get_personalized_recommendations(
        history_ids=REAL_BROWSER_HISTORY_IDS,
        k=8,
        session_id="debug_session"
    )
    print("SUCCESS! Output:")
    print("Mode:", res.get("mode"))
    print("Count:", res.get("count"))
    print("History used:", len(res.get("history_used", [])))
    print("Data count:", len(res.get("data", [])))
    for item in res.get("data", []):
        print(" -", item["product_id"], item["product_name"][:40], "score:", item["score"])
except Exception as e:
    print("EXCEPTION CAUGHT:")
    print(e)
    traceback.print_exc()
finally:
    db.close()
