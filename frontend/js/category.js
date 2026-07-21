document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize UI components
    if (window.UI && window.UI.initNavbar) {
        window.UI.initNavbar();
    }

    const grid = document.getElementById('category-grid');
    const titleEl = document.getElementById('category-title');
    const countEl = document.getElementById('category-count');
    
    if (!grid) return;

    // 2. Parse Category from URL
    const params = new URLSearchParams(window.location.search);
    const category = params.get('cat');

    if (!category) {
        titleEl.innerHTML = `<i class="ph-fill ph-warning" style="color:var(--brand-primary);"></i> Category Not Found`;
        countEl.textContent = "Please select a valid category from the menu.";
        grid.innerHTML = '';
        return;
    }

    const formattedCategory = window.UI ? window.UI.formatCategoryName(category) : category;
    titleEl.innerHTML = `<i class="ph-fill ph-tag" style="color:var(--brand-secondary);"></i> ${formattedCategory}`;

    try {
        // 3. Fetch Category Data from Backend
        // Our backend API accepts the category param and handles filtering natively via ProductService
        const res = await window.api.getProducts(0, 100, category);
        
        if (res.error || !res.data || res.data.length === 0) {
            countEl.textContent = "0 products found.";
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1; padding: 4rem 0;">
                    <i class="ph ph-package empty-icon"></i>
                    <h3>No products found</h3>
                    <p>Try exploring other categories.</p>
                </div>
            `;
            return;
        }

        // 4. Render Grid
        const products = res.data;
        countEl.textContent = `Showing ${products.length} products`;
        grid.innerHTML = products.map(p => window.UI.renderProductCard(p, 'default')).join('');
        
        if (window.UI.initAnimations) window.UI.initAnimations();

    } catch (err) {
        console.error('Failed to load category products:', err);
        countEl.textContent = "Error loading products.";
        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1; padding: 4rem 0;">
                <i class="ph ph-warning-circle empty-icon" style="color:#ef4444"></i>
                <h3>Connection Error</h3>
                <p>Could not load products for this category.</p>
            </div>
        `;
    }
});
