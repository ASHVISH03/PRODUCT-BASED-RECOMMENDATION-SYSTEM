/**
 * Main Application Logic
 * Fetches data from backend and populates the Homepage
 */

document.addEventListener('DOMContentLoaded', async () => {
    
    // 1. Check backend connection & populate metrics
    try {
        const health = await api.getHealth();
        if (health.status === 'ok') {
            document.getElementById('stat-status').innerHTML = '<i class="ph-fill ph-check-circle"></i>';
        }
        
        // Fetch some basic metrics (this might fail if /metrics endpoint isn't exactly as expected, 
        // but we'll safely try-catch it)
        try {
            const metrics = await api.getMetrics();
            if (metrics.system_metrics) {
                document.getElementById('stat-products').innerText = metrics.system_metrics.total_products || '1351';
                // Placeholder for categories since backend metrics might not have it
                document.getElementById('stat-categories').innerText = '10+';
            }
        } catch (e) {
            console.warn("Could not fetch metrics, using fallbacks.");
            document.getElementById('stat-products').innerText = '1,351';
            document.getElementById('stat-categories').innerText = '15';
        }
        
    } catch (err) {
        console.error('Backend offline:', err);
        document.getElementById('stat-status').innerHTML = '<i class="ph-fill ph-warning-circle" style="color:#cf222e"></i>';
    }

    // 2. Fetch Categories for the Popular Categories Section
    try {
        const catRes = await api.getCategories();
        let categories = catRes.data || [];
        
        const cleanCats = [...new Set(categories.map(c => c.split('|')[0].trim()))].filter(c => c);
        
        const gridContainer = document.getElementById('categories-grid');
        if (gridContainer) {
            gridContainer.innerHTML = '';
            cleanCats.slice(0, 8).forEach(cat => {
                gridContainer.innerHTML += UI.renderCategoryCard(cat);
            });
        }
        
        document.getElementById('stat-categories').innerText = cleanCats.length || '15';
    } catch (err) {
        console.error('Failed to load categories:', err);
    }

    // 3. Fetch Products for Grids
    try {
        const productRes = await api.getProducts(0, 56);
        const products = productRes.data || [];
        
        // Update stats
        document.getElementById('stat-products').innerText = '1,465'; // Kaggle dataset size
        
        if (products.length > 0) {
            // AI Picks: PERSONALIZED — powered by TF-IDF user profile + MMR
            // Loaded separately below via getPersonalizedRecommendations()
            const deals = products.slice(0, 8);
            const trending = products.slice(8, 16);
            const recommended = products.slice(16, 24);
            const newArrivals = products.slice(24, 32);
            const topRated = products.slice(32, 40);
            const bestSellers = products.slice(40, 48);

            const grids = [
                { id: 'deals-grid', data: deals, ctx: 'deals' },
                { id: 'trending-grid', data: trending, ctx: 'trending' },
                { id: 'recommended-grid', data: recommended, ctx: 'default' },
                { id: 'new-arrivals-grid', data: newArrivals, ctx: 'new' },
                { id: 'top-rated-grid', data: topRated, ctx: 'default' },
                { id: 'bestsellers-grid', data: bestSellers, ctx: 'default' }
            ];

            grids.forEach(grid => {
                const el = document.getElementById(grid.id);
                if (el) el.innerHTML = grid.data.map(p => UI.renderProductCard(p, grid.ctx)).join('');
            });

            // ── PERSONALIZED AI PICKS ─────────────────────────────────────────
            // Reads the user's browsing history from localStorage, sends it to the
            // backend personalization endpoint, and renders the results as AI Picks.
            // Cold start (no history) → backend returns trending products instead.
            const loadPersonalizedPicks = async () => {
                const aiPicksGrid = document.getElementById('ai-picks-grid');
                const aiPicksSubtitle = document.getElementById('ai-picks-subtitle');
                if (!aiPicksGrid) return;

                // 1. Audit localStorage history
                const rawHistory = localStorage.getItem('amazclone_recently_viewed');
                let recentIds = [];
                try {
                    recentIds = JSON.parse(rawHistory || '[]');
                } catch (e) {
                    console.error('[AI Picks Audit] Failed to parse amazclone_recently_viewed:', e);
                }

                // Filter to valid canonical ID strings
                recentIds = recentIds.filter(id => typeof id === 'string' && id.trim().length > 0);

                console.info(`[AI Picks Audit] Raw localStorage amazclone_recently_viewed:`, rawHistory);
                console.info(`[AI Picks Audit] Parsed canonical history IDs (${recentIds.length}):`, recentIds);

                // Dynamic Subtitle update before request
                if (recentIds.length > 0) {
                    if (aiPicksSubtitle) aiPicksSubtitle.innerText = 'Personalized from your recent activity';
                } else {
                    if (aiPicksSubtitle) aiPicksSubtitle.innerText = 'Trending recommendations for you';
                }

                // Invalidate frontend cache
                if (api.clearRecommendationCache) api.clearRecommendationCache();

                try {
                    const res = await api.getPersonalizedRecommendations(recentIds, 8);
                    
                    if (res.error || !res.data || res.data.length === 0) {
                        console.error('[AI Picks ERROR] Personalized API request failed or returned empty:', res);
                        aiPicksGrid.innerHTML = `
                            <div class="empty-state" style="grid-column: 1 / -1; padding: 3rem; text-align: center;">
                                <i class="ph ph-warning-circle empty-icon" style="color:var(--brand-primary); font-size: 2.5rem;"></i>
                                <h3 style="margin-top: 1rem;">Failed to load AI Picks</h3>
                                <p style="color:var(--text-secondary);">Backend returned HTTP ${res.status || 'Error'}. Check console for details.</p>
                            </div>
                        `;
                        return;
                    }

                    // Update Subtitle based on backend mode
                    if (res.mode === 'personalized' && aiPicksSubtitle) {
                        aiPicksSubtitle.innerText = 'Personalized from your recent activity';
                    } else if (res.mode === 'cold_start' && aiPicksSubtitle) {
                        aiPicksSubtitle.innerText = 'Trending recommendations for you';
                    }

                    const picks = res.data;
                    console.info(
                        `[AI Picks Success] Mode: ${res.mode} | History Used: [${(res.history_used || []).join(', ')}] | ` +
                        `Rendered IDs: [${picks.map(p => p.product_id).join(', ')}]`
                    );

                    aiPicksGrid.innerHTML = picks.map(p => UI.renderProductCard(p, 'ai')).join('');
                    if (window.UI && window.UI.initAnimations) window.UI.initAnimations();

                } catch (err) {
                    console.error('[AI Picks ERROR] Exception in loadPersonalizedPicks:', err);
                    aiPicksGrid.innerHTML = `
                        <div class="empty-state" style="grid-column: 1 / -1; padding: 3rem; text-align: center;">
                            <i class="ph ph-warning-circle empty-icon" style="color:var(--brand-primary); font-size: 2.5rem;"></i>
                            <h3 style="margin-top: 1rem;">Recommendation Error</h3>
                            <p style="color:var(--text-secondary);">${err.message}</p>
                        </div>
                    `;
                }
            };
            loadPersonalizedPicks();

            // ── CONTINUE VIEWING (Recently Viewed) ────────────────────────────
            // This is NOT a recommendation algorithm — it displays raw browsing history.
            const rvSection = document.getElementById('recently-viewed-section');
            const rvGrid = document.getElementById('recently-viewed-grid');
            if (rvSection && rvGrid) {
                const recent = JSON.parse(localStorage.getItem('amazclone_recently_viewed') || '[]');
                if (recent.length > 0) {
                    rvSection.style.display = 'block';
                    const renderRV = async () => {
                        let html = '';
                        for (let pid of recent.slice(0, 5)) { // Show max 5 on homepage
                            let product = window.globalProductCache[pid];
                            if (!product && window.api) {
                                const res = await window.api.getProduct(pid);
                                if (!res.error && res.data) product = res.data;
                            }
                            if (product) {
                                html += UI.renderProductCard(product, 'default');
                            }
                        }
                        rvGrid.innerHTML = html;
                        if (window.UI.initAnimations) window.UI.initAnimations();
                    };
                    renderRV();
                } else {
                    rvSection.style.display = 'none';
                }
            }

        } else {
            console.warn("Backend returned 0 products.");
            const msg = "Database is empty. Please run the backend seeder.";
            document.querySelectorAll('.product-grid').forEach(el => UI.showError(el.id, msg));
        }
    } catch (err) {
        console.error('Failed to load products:', err);
        const msg = "Could not connect to FastAPI backend. Is it running on port 8000?";
        document.querySelectorAll('.product-grid').forEach(el => UI.showError(el.id, msg));
    }

    // Intersection Observer for scroll animations
    UI.initAnimations();
});
