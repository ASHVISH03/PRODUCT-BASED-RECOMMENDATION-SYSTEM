import sqlite3
import pandas as pd
import json

conn = sqlite3.connect('app/data/ecommerce.db')
df = pd.read_sql_query('SELECT * FROM products', conn)

categories = df['category'].str.split('|').str[0]
stats = {
    'total': len(df),
    'unique_categories': df['category'].nunique(),
    'top_level_count': categories.nunique(),
    'top_levels': categories.value_counts().to_dict(),
    'missing_images': int(df['img_link'].isna().sum()),
    'missing_prices': int(df['discounted_price'].isna().sum()),
    'missing_ratings': int(df['rating'].isna().sum())
}

with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f)
