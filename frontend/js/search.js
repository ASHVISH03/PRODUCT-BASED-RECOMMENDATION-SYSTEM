/**
 * Search & Discovery Engine
 * Handles client-side filtering, sorting, and pagination for search results.
 */

document.addEventListener('DOMContentLoaded', async () => {
    // Elements
    const grid = document.getElementById('search-results-grid');
    const titleEl = document.getElementById('search-title');
    const countEl = document.getElementById('search-count');
    const emptyState = document.getElementById('search-empty-state');
    
    const filterBrands = document.getElementById('filter-brands');
    const filterCategories = document.getElementById('filter-categories');
    const priceSlider = document.getElementById('price-slider');
    const priceVal = document.getElementById('price-slider-val');
    const sortSelect = document.getElementById('sort-select');
    const clearBtn = document.getElementById('clear-filters-btn');
    
    // View Toggles
    const viewGrid = document.getElementById('view-grid');
    const viewList = document.getElementById('view-list');

    // State
    let allResults = []; // Raw fetched data
    let filteredResults = []; // Data after filters/sort applied
    
    // Parse URL params
    const params = new URLSearchParams(window.location.search);
    const query = params.get('q');
    const cat = params.get('category') || params.get('cat');

    // Init View
    const setView = (view) => {
        if (view === 'list') {
            grid.classList.add('list-view');
            viewList.classList.add('active');
            viewGrid.classList.remove('active');
        } else {
            grid.classList.remove('list-view');
            viewGrid.classList.add('active');
            viewList.classList.remove('active');
        }
    };
    
    viewGrid.addEventListener('click', () => setView('grid'));
    viewList.addEventListener('click', () => setView('list'));

    // Price Slider UI Update
    priceSlider.addEventListener('input', (e) => {
        priceVal.innerText = `₹${parseInt(e.target.value).toLocaleString()}`;
    });

    /**
     * 1. Fetch Data
     */
    try {
        if (query) {
            titleEl.innerText = `Results for "${query}"`;
            document.title = `Search: ${query} - AmazClone`;
            // Fetch top 50 highly relevant ML results
            const res = await api.searchProducts(query, 50);
            allResults = res.data || [];
        } else if (cat) {
            const formattedCat = window.UI ? window.UI.formatCategoryName(cat) : cat;
            titleEl.innerText = `Category: ${formattedCat}`;
            document.title = `${formattedCat} - AmazClone`;
            // Fetch up to 100 products in category
            const res = await api.getProducts(0, 100, cat);
            allResults = res.data || [];
        } else {
            titleEl.innerText = `All Products`;
            // Discovery Mode
            const res = await api.getProducts(0, 100);
            allResults = res.data || [];
        }
    } catch (err) {
        console.error("Search fetch failed:", err);
        titleEl.innerText = "Error loading results";
        grid.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }

    /**
     * 2. Build Sidebar Filters
     */
    const extractFilters = () => {
        const brands = new Set();
        const categories = new Set();
        let maxPrice = 0;

        allResults.forEach(p => {
            if (p.brand) brands.add(p.brand.split('|')[0].trim());
            if (p.category) categories.add(p.category.split('|')[0].trim());
            
            const price = parseFloat(String(p.discounted_price || p.price).replace(/[^0-9.]/g, '')) || 0;
            if (price > maxPrice) maxPrice = price;
        });

        // Populate Categories
        if (filterCategories) {
            filterCategories.innerHTML = Array.from(categories).slice(0, 8).map(c => `
                <label class="filter-item">
                    <input type="checkbox" value="${c}" class="filter-checkbox filter-cat-cb">
                    ${c}
                </label>
            `).join('');
        }

        // Populate Brands
        if (filterBrands) {
            filterBrands.innerHTML = Array.from(brands).slice(0, 8).map(b => `
                <label class="filter-item">
                    <input type="checkbox" value="${b}" class="filter-checkbox filter-brand-cb">
                    ${b}
                </label>
            `).join('');
        }

        // Setup Price Slider
        if (maxPrice > 0) {
            priceSlider.max = Math.ceil(maxPrice / 100) * 100;
            priceSlider.value = priceSlider.max;
            priceVal.innerText = `₹${parseInt(priceSlider.max).toLocaleString()}`;
        }
    };
    
    extractFilters();

    /**
     * 3. Apply Filters & Sorting
     */
    const applyFilters = () => {
        const selectedCats = Array.from(document.querySelectorAll('.filter-cat-cb:checked')).map(cb => cb.value);
        const selectedBrands = Array.from(document.querySelectorAll('.filter-brand-cb:checked')).map(cb => cb.value);
        const maxP = parseInt(priceSlider.value);
        const minRating = parseFloat(document.querySelector('input[name="rating"]:checked').value);
        
        const sortMode = sortSelect.value;

        // Filter
        filteredResults = allResults.filter(p => {
            const price = parseFloat(String(p.discounted_price || p.price).replace(/[^0-9.]/g, '')) || 0;
            const rating = parseFloat(p.rating) || 0;
            const b = p.brand ? p.brand.split('|')[0].trim() : '';
            const c = p.category ? p.category.split('|')[0].trim() : '';

            if (price > maxP) return false;
            if (rating < minRating) return false;
            if (selectedCats.length > 0 && !selectedCats.includes(c)) return false;
            if (selectedBrands.length > 0 && !selectedBrands.includes(b)) return false;
            
            return true;
        });

        // Sort
        if (sortMode === 'price_asc') {
            filteredResults.sort((a, b) => {
                const pa = parseFloat(String(a.discounted_price || a.price).replace(/[^0-9.]/g, '')) || 0;
                const pb = parseFloat(String(b.discounted_price || b.price).replace(/[^0-9.]/g, '')) || 0;
                return pa - pb;
            });
        } else if (sortMode === 'price_desc') {
            filteredResults.sort((a, b) => {
                const pa = parseFloat(String(a.discounted_price || a.price).replace(/[^0-9.]/g, '')) || 0;
                const pb = parseFloat(String(b.discounted_price || b.price).replace(/[^0-9.]/g, '')) || 0;
                return pb - pa;
            });
        } else if (sortMode === 'rating') {
            filteredResults.sort((a, b) => {
                return (parseFloat(b.rating) || 0) - (parseFloat(a.rating) || 0);
            });
        }
        // If 'relevance', leave as original array order (which was sorted by ML backend)

        renderResults();
    };

    /**
     * 4. Render Grid
     */
    const renderResults = () => {
        countEl.innerText = `(${filteredResults.length} items)`;
        
        if (filteredResults.length === 0) {
            grid.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }

        emptyState.style.display = 'none';
        
        // Render all matching (up to 50 is fine for DOM without complex pagination)
        grid.innerHTML = filteredResults.map(p => UI.renderProductCard(p, 'default')).join('');
        
        // Trigger animations
        if (window.UI.initAnimations) window.UI.initAnimations();
    };

    // Attach Event Listeners to Filters
    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('filter-checkbox') || e.target.name === 'rating') {
            applyFilters();
        }
    });
    priceSlider.addEventListener('change', applyFilters);
    sortSelect.addEventListener('change', applyFilters);

    clearBtn.addEventListener('click', () => {
        document.querySelectorAll('.filter-checkbox').forEach(cb => cb.checked = false);
        document.querySelector('input[name="rating"][value="0"]').checked = true;
        priceSlider.value = priceSlider.max;
        priceVal.innerText = `₹${parseInt(priceSlider.max).toLocaleString()}`;
        sortSelect.value = 'relevance';
        applyFilters();
    });

    // Initial Render
    applyFilters();
});
