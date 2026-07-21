# Dataset Audit

## Overview
This document provides a detailed audit of the current Amazon e-commerce product dataset used in Milestone 4. It includes numerical statistics regarding product distribution, missing data, and review coverage, establishing a baseline for the Recommendation Model Validation and future Dataset V2 migrations.

## Numerical Statistics

- **Total Unique Products**: 1,465
- **Top-Level Categories**: 23
- **Subcategories**: ~1,300 unique hierarchical paths
- **Duplicate IDs**: 0

### Top-Level Category Distribution
The dataset exhibits a significant imbalance, heavily favoring electronics accessories and cables.

1. **Computers & Accessories**: ~610 (41.6%)
2. **Electronics**: ~485 (33.1%)
3. **Home & Kitchen**: ~255 (17.4%)
4. **Office Products**: ~35 (2.4%)
5. **Musical Instruments**: ~15 (1.0%)
6. **Other**: ~65 (4.5%)

*Approximately 74.7% of the dataset is concentrated in just two major categories.*

### Data Completeness & Coverage
- **Missing Images**: 0 (0%)
- **Missing Prices (discounted_price)**: ~12 (0.8%)
- **Missing Ratings**: 0 (0%)
- **Review Coverage**: ~100% (Products consistently contain delimited strings for `user_name`, `review_title`, and `review_content`).

## Implications for Recommendation Models
Due to the heavy concentration in `Computers & Accessories` (primarily cables and chargers), collaborative filtering and content-based models will inevitably bias toward these items. TF-IDF artifacts will heavily prioritize and overweigh terms like "USB", "Type-C", "Braided", and "Fast Charging" which may reduce recommendation diversity for broader categories.

## Strategy for Dataset V2 Migration
When migrating to a larger dataset (e.g., 50k+ items) in a future milestone, the following constraints must be met to preserve system stability:

1. **Canonical IDs**: Canonical `product_id` fields must be preserved. If moving away from ASINs (B0...), the new format must not clash.
2. **Schema Integrity**: The pipeline relies on hierarchical `category` fields separated by `|`, and review arrays separated by `,`. Preprocessing scripts (`src/data/preprocessing.py`) must normalize any new dataset into this exact schema.
3. **Graceful Client Failures**: The frontend `localStorage` (Cart, Wishlist, Recently Viewed) stores canonical IDs. If a user returns and their stored ID no longer exists in Dataset V2, the UI and API must safely drop or ignore the 404 rather than breaking the rendering loop.
