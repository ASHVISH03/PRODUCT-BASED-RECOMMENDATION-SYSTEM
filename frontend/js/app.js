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

            // ── MULTI-SIGNAL INTERACTION STORAGE MANAGER ──────────────────────
            window.Interactions = {
                get() {
                    try {
                        return JSON.parse(localStorage.getItem('amazclone_interactions') || '[]');
                    } catch (e) {
                        return [];
                    }
                },

                add(productId, eventType = 'view', quantity = 1) {
                    if (!productId || typeof productId !== 'string') return;
                    let interactions = this.get();
                    const now = Date.now();
                    const cleanType = String(eventType).trim().lowerCase ? eventType.toLowerCase() : 'view';

                    // Rapid repeat view check (within 60 seconds)
                    if (cleanType === 'view' && interactions.length > 0) {
                        const last = interactions[0];
                        if (last.product_id === productId && (now - (last.timestamp || 0)) < 60000) {
                            last.event_type = 'repeat_view';
                            last.timestamp = now;
                            localStorage.setItem('amazclone_interactions', JSON.stringify(interactions));
                            return;
                        }
                    }

                    // Prepend new interaction event (newest first)
                    interactions.unshift({
                        product_id: productId,
                        event_type: cleanType,
                        timestamp: now,
                        quantity: Number(quantity) || 1
                    });

                    // Cap to 50 items
                    if (interactions.length > 50) interactions = interactions.slice(0, 50);

                    localStorage.setItem('amazclone_interactions', JSON.stringify(interactions));
                    console.info(`[Interactions] Recorded event '${cleanType}' for product ${productId}`);
                },

                removeWishlist(productId) {
                    let interactions = this.get();
                    interactions = interactions.filter(item => !(item.product_id === productId && item.event_type === 'wishlist'));
                    localStorage.setItem('amazclone_interactions', JSON.stringify(interactions));
                }
            };

            // ── PERSONALIZED AI PICKS ─────────────────────────────────────────
            const loadPersonalizedPicks = async () => {
                const aiPicksGrid = document.getElementById('ai-picks-grid');
                const aiPicksSubtitle = document.getElementById('ai-picks-subtitle');
                if (!aiPicksGrid) return;

                // 1. Audit localStorage history & interactions
                const rawHistory = localStorage.getItem('amazclone_recently_viewed');
                let recentIds = [];
                try {
                    recentIds = JSON.parse(rawHistory || '[]');
                } catch (e) {}

                recentIds = recentIds.filter(id => typeof id === 'string' && id.trim().length > 0);
                const interactions = window.Interactions.get();

                console.info(`[AI Picks Audit] Raw amazclone_recently_viewed:`, recentIds);
                console.info(`[AI Picks Audit] Raw amazclone_interactions (${interactions.length}):`, interactions);

                // Dynamic Subtitle update before request
                if (interactions.length > 0 || recentIds.length > 0) {
                    if (aiPicksSubtitle) aiPicksSubtitle.innerText = 'Personalized from your recent activity';
                } else {
                    if (aiPicksSubtitle) aiPicksSubtitle.innerText = 'Trending recommendations for you';
                }

                if (api.clearRecommendationCache) api.clearRecommendationCache();

                try {
                    // Call POST multi-signal personalized endpoint
                    let res = await api.postMultiSignalPersonalizedRecommendations(interactions, recentIds, 8);
                    
                    if (res.error || !res.data || res.data.length === 0) {
                        console.warn('[AI Picks Fallback] POST request returned error/empty, trying GET fallback...');
                        res = await api.getPersonalizedRecommendations(recentIds, 8);
                    }

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

                    // Update Subtitle based on dominant signal
                    if (aiPicksSubtitle) {
                        if (res.dominant_signal === 'wishlist') {
                            aiPicksSubtitle.innerText = 'Personalized from your wishlist & browsing activity';
                        } else if (res.dominant_signal === 'cart') {
                            aiPicksSubtitle.innerText = 'Personalized from your cart & browsing activity';
                        } else if (res.dominant_signal === 'purchase') {
                            aiPicksSubtitle.innerText = 'Personalized from your purchases & browsing activity';
                        } else if (res.mode === 'personalized') {
                            aiPicksSubtitle.innerText = 'Personalized from your recent activity';
                        } else {
                            aiPicksSubtitle.innerText = 'Trending recommendations for you';
                        }
                    }

                    const picks = res.data;
                    console.info(
                        `[AI Picks Success] Mode: ${res.mode} | Dominant Signal: ${res.dominant_signal || 'N/A'} | ` +
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
