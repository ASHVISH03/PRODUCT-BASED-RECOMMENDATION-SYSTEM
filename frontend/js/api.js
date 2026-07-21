/**
 * API Client for FastAPI Backend
 * Base URL: http://127.0.0.1:8000/api/v1
 */

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    let result = { data: null, error: true, status: 0 };
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        result.status = response.status;

        if (response.ok) {
            result = await response.json();
            result.error = false;
            result.status = response.status;
        } else {
            console.error(`[API Error] GET ${url} returned HTTP ${response.status} ${response.statusText}`);
            result.errorText = `HTTP ${response.status} ${response.statusText}`;
        }
    } catch (error) {
        console.error(`[API Network Error] GET ${url}:`, error);
        result.errorText = error.message;
    } finally {
        return result;
    }
}

const cache = {
    health: null,
    metrics: null,
    categories: null,
    healthLastFetch: 0,
    products: {},
    recommendations: {}
};

const CACHE_TTL = 30000; // 30 seconds for health/metrics

const api = {
    /**
     * Check backend health
     */
    async getHealth() {
        const now = Date.now();
        if (cache.health && (now - cache.healthLastFetch) < CACHE_TTL) {
            return cache.health;
        }
        const res = await fetchAPI('/health');
        cache.health = res;
        cache.healthLastFetch = now;
        return res;
    },

    /**
     * Get system metrics (Cached)
     */
    async getMetrics() {
        if (cache.metrics) return cache.metrics;
        const res = await fetchAPI('/metrics');
        if (!res.error) cache.metrics = res;
        return res;
    },

    /**
     * Get paginated products
     * @param {number} skip 
     * @param {number} limit 
     * @param {string} category 
     */
    async getProducts(skip = 0, limit = 20, category = '') {
        let url = `/products?skip=${skip}&limit=${limit}`;
        if (category) {
            url += `&category=${encodeURIComponent(category)}`;
        }
        return fetchAPI(url);
    },

    /**
     * Get single product details (Cached)
     * @param {string} productId 
     */
    async getProduct(productId) {
        if (cache.products[productId]) return cache.products[productId];
        const res = await fetchAPI(`/products/${encodeURIComponent(productId)}`);
        if (!res.error) cache.products[productId] = res;
        return res;
    },

    /**
     * Get list of all categories (Cached)
     */
    async getCategories() {
        if (cache.categories) return cache.categories;
        const res = await fetchAPI('/products/categories');
        cache.categories = res;
        return res;
    },

    /**
     * Search products with typo correction
     * @param {string} query 
     * @param {number} limit 
     */
    async searchProducts(query, limit = 20) {
        return fetchAPI(`/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    },

    /**
     * Get AI Recommendations (Cached)
     * @param {string} productId 
     * @param {string} strategy 
     */
    async getRecommendations(productId, strategy = 'hybrid', k = 10) {
        const cacheKey = `${productId}_${strategy}_${k}`;
        if (cache.recommendations[cacheKey]) return cache.recommendations[cacheKey];
        const res = await fetchAPI(`/recommendations/${encodeURIComponent(productId)}?strategy=${strategy}&k=${k}`);
        if (!res.error) cache.recommendations[cacheKey] = res;
        return res;
    },

    /**
     * Get personalized AI Picks for a session — NOT cached (always fresh)
     * @param {string[]} historyIds - Product IDs viewed, newest first
     * @param {number} k - Number of recommendations
     */
    async getPersonalizedRecommendations(historyIds, k = 8) {
        let endpoint = `/recommendations/personalized?k=${k}`;
        if (historyIds && historyIds.length > 0) {
            const historyParam = historyIds.map(id => encodeURIComponent(id)).join(',');
            endpoint = `/recommendations/personalized?history=${historyParam}&k=${k}`;
        }
        const fullUrl = `${API_BASE_URL}${endpoint}`;
        console.info(`[Personalized API Request] Requesting URL: ${fullUrl}`);
        const res = await fetchAPI(endpoint);
        console.info(`[Personalized API Response] HTTP Status: ${res.status || 'OK'}, mode: ${res.mode}, count: ${res.count}, data:`, res);
        return res;
    },

    /**
     * Invalidate the recommendation cache
     * Call this before loading AI Picks to ensure fresh recommendations
     */
    clearRecommendationCache() {
        cache.recommendations = {};
    },

    /**
     * Trigger background ML training
     */
    async trainModel() {
        return fetchAPI('/train', { method: 'POST' });
    }
};

window.api = api;
