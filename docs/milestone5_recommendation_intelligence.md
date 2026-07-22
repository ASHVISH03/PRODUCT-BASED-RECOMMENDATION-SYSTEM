# Milestone 5 — Multi-Signal Recommendation Intelligence & MLflow Experiment Tracking

## Executive Summary

Milestone 5 upgrades the Product Recommendation System from a single-signal view-based personalization system into a **Multi-Signal Recommendation Intelligence System**.

User actions now carry distinct configurable weights reflecting purchase intent:
- **`view`**: $1.0$ (standard browsing)
- **`repeat_view`**: $1.5$ (repeated interest)
- **`wishlist`**: $2.0$ (explicit save)
- **`cart`**: $3.0$ (high-intent add to cart)
- **`purchase`**: $5.0$ (strongest purchase signal)

---

## 1. System Architecture Evolution

```
[ Frontend Interactions ] ──> [ localStorage: amazclone_interactions ]
                                      │
                                      ▼
                        [ POST /api/v1/recommendations/personalized ]
                                      │
                                      ▼
                      [ RecommendationService (v2) ]
                                      │
                                      ▼
                     [ Multi-Signal PersonalizedEngine ]
                     ├── 1. Recency Decay (0.85^pos)
                     ├── 2. Interaction Weighting (1.0 - 5.0)
                     ├── 3. Weighted TF-IDF Profile Vector
                     ├── 4. Cosine Catalog Scoring
                     ├── 5. History & Excluded Item Filtering
                     └── 6. MMR Diversity Reranking (λ = 0.60)
                                      │
                                      ▼
                     [ Dynamic Explainable Reasons & Tags ]
                                      │
                                      ▼
                        [ MLflow Experiment Tracker ]
```

---

## 2. Mathematical Profile Construction

For a sequence of $N$ user interactions $(p_i, e_i)$ where $p_i$ is the product ID and $e_i$ is the event type:

1. **Base Interaction Weight:**
   $$W_{\text{base}}(e_i) \in \{1.0, 1.5, 2.0, 3.0, 5.0\}$$

2. **Recency Decay:**
   $$\text{decay}(i) = 0.85^i \quad (i = 0 \text{ is newest})$$

3. **Effective Item Weight:**
   $$w_i = W_{\text{base}}(e_i) \times 0.85^i$$

4. **Normalized User Profile Vector:**
   $$\vec{u}_{\text{user}} = \frac{\sum_{i=0}^{N-1} w_i \cdot \vec{v}_{\text{tfidf}}(p_i)}{\sum_{i=0}^{N-1} w_i}$$
   $$\vec{u}_{\text{unit}} = \frac{\vec{u}_{\text{user}}}{\|\vec{u}_{\text{user}}\|_2}$$

5. **Cosine Similarity Catalog Scoring & MMR Reranking:**
   $$\text{score}(c) = \text{cosine\_similarity}(\vec{u}_{\text{unit}}, \vec{v}_{\text{tfidf}}(c))$$
   $$\text{MMR}(c) = \lambda \cdot \text{score}(c) - (1 - \lambda) \max_{s \in S} \text{sim}(c, s) \quad (\lambda = 0.60)$$

---

## 3. Pydantic API Specifications

### POST `/api/v1/recommendations/personalized`

**Request Body:**
```json
{
  "interactions": [
    {
      "product_id": "B09WNJ6LXZ",
      "event_type": "cart",
      "timestamp": 1784660000000,
      "quantity": 1
    },
    {
      "product_id": "B08L879JSN",
      "event_type": "wishlist",
      "timestamp": 1784659000000
    }
  ],
  "k": 8,
  "session_id": "session_demo"
}
```

**Response Body:**
```json
{
  "status": "success",
  "mode": "personalized",
  "history_used": ["B09WNJ6LXZ", "B08L879JSN"],
  "dominant_signal": "cart",
  "count": 8,
  "data": [
    {
      "product_id": "B08TV3B8D4",
      "product_name": "Realme Smart TV Stick 4K",
      "score": 0.541,
      "confidence_grade": "Medium",
      "recommendation_reason": "Related to items in your cart (Electronics)",
      "reason_tags": ["Personalized", "Cart Affinity", "Top Rated"]
    }
  ]
}
```

---

## 4. Controlled Scenario Results

| Scenario | Trigger Interactions | Mode | Dominant Signal | Top Recommendation Category | History Leakage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Empty history | `cold_start` | `None` | Trending Products | $0$ |
| **Scenario B** | 3 Monitor views | `personalized` | `view` | Display / Monitor Accessories | $0$ |
| **Scenario C** | Monitors viewed + Gaming mouse in cart | `personalized` | `cart` | Gaming Accessories & Keyboards | $0$ |
| **Scenario D** | Purchased Streaming TV Stick | `personalized` | `purchase` | HDMI Cables & Adapters | $0$ |
| **Scenario E** | Old Electronics views + New Kitchen cart | `personalized` | `cart` | Home & Kitchen Appliances | $0$ |

---

## 5. MLflow Tracking Integration

- **Experiment Name:** `Product_Recommendation_Intelligence_M5`
- **Logged Parameters:** `recommendation_strategy`, `decay_factor`, `mmr_lambda`, `interaction_weights`, `catalog_size`, `vocab_size`.
- **Logged Metrics:** `recommendation_latency_ms`, `history_leakage`, `intra_list_diversity`, `catalog_coverage`.
- **Logged Artifacts:** Scenario evaluation snapshots saved to `artifacts/experiments/*.json`.
- **Resilience:** All MLflow calls are wrapped in exception handlers to prevent tracking offline state from breaking API responses.
