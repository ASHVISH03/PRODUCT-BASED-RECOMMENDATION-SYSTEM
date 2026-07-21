document.addEventListener('DOMContentLoaded', async () => {
    const params = UI.getQueryParams();
    const productId = params.id;
    
    if (!productId) {
        document.getElementById('main-content').innerHTML = `
            <div class="empty-state">
                <i class="ph ph-warning-circle empty-icon" style="color:var(--brand-primary)"></i>
                <h3>Product Not Found</h3>
                <p>No product ID was provided.</p>
                <a href="index.html" class="btn btn-primary" style="margin-top:1rem">Back to Home</a>
            </div>
        `;
        return;
    }

    try {
        const productRes = await window.api.getProduct(productId);
        if (productRes.error || !productRes.data) throw new Error("Product not found");
        
        const product = productRes.data;
        renderProductPage(product);
        
        // Fetch recommendations asynchronously so it doesn't block main render
        fetchAndRenderRecommendations(productId);

    } catch (e) {
        document.getElementById('main-content').innerHTML = `
            <div class="empty-state">
                <i class="ph ph-warning-circle empty-icon" style="color:var(--brand-primary)"></i>
                <h3>Error Loading Product</h3>
                <p>We couldn't load the details for this item.</p>
                <a href="index.html" class="btn btn-primary" style="margin-top:1rem">Back to Home</a>
            </div>
        `;
    }
});

function renderProductPage(product) {
    document.title = `${product.product_name} | Amazclone`;
    
    // Breadcrumb
    const categories = product.category ? product.category.split('|') : ['Category'];
    const breadcrumbHtml = `
        <a href="index.html">Home</a> 
        <span><i class="ph ph-caret-right"></i></span> 
        <a href="category.html?cat=${encodeURIComponent(categories[0])}">${window.UI ? window.UI.formatCategoryName(categories[0]) : categories[0]}</a>
        <span><i class="ph ph-caret-right"></i></span> 
        <span style="color:var(--text-primary)">${product.brand || 'Premium Brand'}</span>
    `;
    document.getElementById('product-breadcrumb').innerHTML = breadcrumbHtml;

    // Format pricing
    const cleanPrice = (val) => {
        if (!val) return 0;
        if (typeof val === 'number') return val;
        const parsed = parseFloat(String(val).replace(/[^0-9.]/g, ''));
        return isNaN(parsed) ? 0 : parsed;
    };
    const formatPrice = (price) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(price);
    
    let currentPriceNum = cleanPrice(product.discounted_price) || cleanPrice(product.price);
    let originalPriceNum = cleanPrice(product.actual_price) || currentPriceNum;
    if (currentPriceNum === 0) currentPriceNum = originalPriceNum || 999;
    
    const currentPrice = formatPrice(currentPriceNum);
    const originalPrice = (originalPriceNum > currentPriceNum) ? formatPrice(originalPriceNum) : '';
    
    let discount = '';
    if (originalPriceNum > currentPriceNum) {
        const discountPercent = Math.round(((originalPriceNum - currentPriceNum) / originalPriceNum) * 100);
        if (discountPercent > 0 && discountPercent < 100) {
            discount = `-${discountPercent}%`;
        }
    }

    // Images
    const category = categories[0] || '';
    let imgLink = UI.getPlaceholderSVG(category);
    if (product.img_link) {
        imgLink = product.img_link.split('|')[0].split(',')[0];
        if (imgLink.includes('m.media-amazon.com/images/W/')) {
            imgLink = imgLink.replace(/\/images\/W\/[^/]+\//, '/');
        }
    }
    const fallbackImg = UI.getPlaceholderSVG(category);

    // Gallery
    document.getElementById('product-gallery').innerHTML = `
        <div class="main-image-container" onclick="window.open('${imgLink}', '_blank')">
            <img id="main-product-img" src="${fallbackImg}" alt="${product.product_name}" 
                 style="opacity: 0.8; transition: opacity 0.5s ease-in;"
                 onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }"
                 onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }">
        </div>
        <div class="thumbnail-strip">
            <div class="thumbnail active" onclick="changeMainImage(this, '${imgLink}')">
                <img src="${fallbackImg}" onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }" onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }">
            </div>
            <!-- Mock additional thumbnails -->
            <div class="thumbnail" onclick="changeMainImage(this, '${imgLink}')">
                <img src="${fallbackImg}" onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }" onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }">
            </div>
            <div class="thumbnail" onclick="changeMainImage(this, '${imgLink}')">
                <img src="${fallbackImg}" onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }" onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }">
            </div>
        </div>
    `;

    // Info
    document.getElementById('product-info').innerHTML = `
        <span class="product-brand-tag">${product.brand || 'Premium Brand'}</span>
        <h1>${product.product_name}</h1>
        <div class="product-rating-overview">
            <div class="stars" style="color:#F59E0B; font-size:1.2rem;">${UI.renderStars(product.rating || 4.5)}</div>
            <span style="color:var(--brand-primary); font-weight:600">${product.rating || 4.5}</span>
            <span style="color:var(--text-tertiary)">(${product.rating_count ? product.rating_count.toLocaleString() : '1,234'} ratings)</span>
        </div>
        <div class="product-short-desc">
            ${formatAboutProduct(product.about_product)}
        </div>
    `;

    // Purchase Panel
    document.getElementById('purchase-panel').innerHTML = `
        <div class="price-block">
            ${discount ? `<div style="color:var(--brand-primary); font-size:1.2rem; font-weight:700; margin-bottom:0.5rem">${discount}</div>` : ''}
            <div class="price-current">${currentPrice}</div>
            ${originalPrice ? `<div class="price-original">M.R.P.: ${originalPrice}</div>` : ''}
        </div>
        <div class="stock-status">
            <i class="ph-fill ph-check-circle"></i> In Stock
        </div>
        <div style="font-size:0.9rem; color:var(--text-secondary); margin-bottom:1.5rem">
            <i class="ph ph-truck"></i> FREE Premium Delivery by Tomorrow
        </div>
        <div class="panel-actions">
            <button class="btn btn-primary" style="width:100%" onclick="window.addToCart('${product.product_id}')">
                <i class="ph ph-shopping-cart"></i> Add to Cart
            </button>
            <button class="btn btn-outline" style="width:100%" onclick="window.addToCart('${product.product_id}'); window.location.href='checkout.html'">
                <i class="ph ph-lightning"></i> Buy Now
            </button>
            <button class="btn btn-ghost" style="width:100%; border:1px solid var(--border-subtle)" onclick="window.toggleWishlist('${product.product_id}', this)">
                <i class="${window.wishlist && window.wishlist.includes(product.product_id) ? 'ph-fill' : 'ph'} ph-heart"></i> Add to Wishlist
            </button>
        </div>
    `;

    // Tabs Content
    document.getElementById('tab-desc').innerHTML = `
        <h3 style="margin-bottom:1rem; color:var(--text-primary)">About this item</h3>
        <ul style="list-style-type:disc; padding-left:1.5rem; color:var(--text-secondary); line-height:1.8">
            ${product.about_product ? product.about_product.split('|').map(item => `<li>${item.trim()}</li>`).join('') : '<li>No description available.</li>'}
        </ul>
    `;
    
    document.getElementById('tab-specs').innerHTML = `
        <h3 style="margin-bottom:1rem; color:var(--text-primary)">Technical Details</h3>
        <table style="width:100%; text-align:left; border-collapse:collapse; color:var(--text-secondary)">
            <tr style="border-bottom:1px solid var(--border-subtle)">
                <td style="padding:1rem 0; font-weight:600; width:40%">Brand</td>
                <td style="padding:1rem 0">${product.brand || 'N/A'}</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border-subtle)">
                <td style="padding:1rem 0; font-weight:600">Category</td>
                <td style="padding:1rem 0">${categories.join(' > ')}</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border-subtle)">
                <td style="padding:1rem 0; font-weight:600">Product ID</td>
                <td style="padding:1rem 0">${product.product_id}</td>
            </tr>
        </table>
    `;

    // Generate Authentic Reviews
    const generateReview = (name, title, content, rating, i) => {
        const stars = UI.renderStars(rating);
        return `
            <div class="review-card fade-up" style="animation-delay: ${0.1 * i}s;">
                <div class="review-header">
                    <div style="display:flex; align-items:center; gap:0.5rem">
                        <div style="width:32px; height:32px; border-radius:50%; background:var(--border-strong); display:flex; align-items:center; justify-content:center">
                            <i class="ph-fill ph-user" style="color:var(--text-secondary)"></i>
                        </div>
                        <span style="font-weight:600; color:var(--text-primary)">${name || 'Amazon Customer'}</span>
                    </div>
                    <span style="color:#10B981; font-size:0.8rem; font-weight:600"><i class="ph-fill ph-seal-check"></i> Verified Purchase</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem">
                    <div class="stars" style="color:#F59E0B; font-size:0.9rem">${stars}</div>
                    <span class="review-title" style="font-weight:600;">${title || 'Review'}</span>
                </div>
                <p class="review-text">${content || 'No review content provided.'}</p>
            </div>
        `;
    };
    
    // Parse real dataset reviews (usually separated by ',' or '|')
    let reviewNames = [], reviewTitles = [], reviewContents = [];
    if (product.user_name) reviewNames = String(product.user_name).split(/[,|]/);
    if (product.review_title) reviewTitles = String(product.review_title).split(/[,|]/);
    if (product.review_content) reviewContents = String(product.review_content).split(/[,|]/);

    const numReviews = Math.min(Math.max(reviewNames.length, 1), 5); // Show max 5 reviews
    let reviewsHtml = '';
    
    if (reviewNames.length === 0 && reviewTitles.length === 0 && reviewContents.length === 0) {
        reviewsHtml = `<p style="color:var(--text-secondary);">No customer reviews available for this product.</p>`;
    } else {
        for (let i = 0; i < numReviews; i++) {
            const name = reviewNames[i] ? reviewNames[i].trim() : 'Amazon Customer';
            const title = reviewTitles[i] ? reviewTitles[i].trim() : 'Review';
            const content = reviewContents[i] ? reviewContents[i].trim() : 'No review content provided.';
            reviewsHtml += generateReview(name, title, content, Math.floor(product.rating || 4), i);
        }
    }

    document.getElementById('tab-reviews').innerHTML = `
        <h3 style="margin-bottom:1rem; color:var(--text-primary)">Customer Reviews</h3>
        ${reviewsHtml}
    `;

    // AI Insights Tab
    document.getElementById('tab-ai').innerHTML = `
        <div style="background:var(--bg-base); padding:2rem; border-radius:var(--radius-md); border:1px solid var(--border-strong)">
            <h3 style="margin-bottom:1rem; color:var(--text-primary); display:flex; align-items:center; gap:0.5rem">
                <i class="ph-fill ph-sparkle" style="color:#a855f7"></i> Recommendation Engine Insights
            </h3>
            <p style="color:var(--text-secondary); margin-bottom:1rem">This product is analyzed by our Hybrid MLOps pipeline. It is clustered in the <strong>${categories[0]}</strong> space and shares a high TF-IDF similarity with ${product.rating_count ? product.rating_count.toLocaleString() : 'thousands of'} user interaction patterns.</p>
            <div style="display:flex; gap:1rem; margin-top:2rem">
                <div style="flex:1; padding:1.5rem; background:var(--bg-surface); border-radius:var(--radius-sm); border:1px solid var(--border-subtle); text-align:center">
                    <div style="font-size:2rem; color:var(--brand-primary); font-family:var(--font-display); font-weight:700">${product.discount_percentage ? Math.round(parseFloat(String(product.discount_percentage).replace(/[^0-9.]/g, '')) || 15) : '15'}%</div>
                    <div style="font-size:0.85rem; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:1px; margin-top:0.5rem">Price Competitiveness</div>
                </div>
                <div style="flex:1; padding:1.5rem; background:var(--bg-surface); border-radius:var(--radius-sm); border:1px solid var(--border-subtle); text-align:center">
                    <div style="font-size:2rem; color:var(--brand-primary); font-family:var(--font-display); font-weight:700">${product.rating || 4.5}</div>
                    <div style="font-size:0.85rem; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:1px; margin-top:0.5rem">Sentiment Score</div>
                </div>
            </div>
        </div>
    `;

    setupTabs();
    
    // Feature 3: Record Recently Viewed
    let recent = JSON.parse(localStorage.getItem('amazclone_recently_viewed') || '[]');
    recent = recent.filter(id => id !== product.product_id);
    recent.unshift(product.product_id);
    if (recent.length > 15) recent = recent.slice(0, 15);
    localStorage.setItem('amazclone_recently_viewed', JSON.stringify(recent));

    // Bug 1: Trigger IntersectionObserver so elements fade in!
    if (window.UI && window.UI.initAnimations) {
        window.UI.initAnimations();
    }
}

function formatAboutProduct(text) {
    if (!text) return 'No description available.';
    const points = text.split('|');
    return points[0] + (points.length > 1 ? ' ' + points[1] : '') + '... <a href="#" onclick="event.preventDefault(); document.querySelector(\'[data-target=tab-desc]\').click(); window.scrollBy(0, 300)" style="color:var(--brand-primary)">Read more</a>';
}

// Global Image switcher for thumbnails
window.changeMainImage = (thumbnailEl, imgSrc) => {
    document.querySelectorAll('.thumbnail').forEach(el => el.classList.remove('active'));
    thumbnailEl.classList.add('active');
    document.getElementById('main-product-img').src = imgSrc;
};

// Setup tabs
function setupTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.getAttribute('data-target')).classList.add('active');
        });
    });
}

// Fetch Recommendations
async function fetchAndRenderRecommendations(productId) {
    try {
        // AI Picks (Hybrid strategy)
        const hybridRes = await window.api.getRecommendations(productId, 'hybrid', 10);
        const hybridContainer = document.getElementById('ai-recommendations-slider');
        if (!hybridRes.error && hybridRes.data && hybridRes.data.length > 0) {
            const reasons = ["Highly similar description", "Top rated in category", "Frequently bought together", "High TF-IDF Match", "Trending hybrid pick"];
            hybridContainer.innerHTML = hybridRes.data.map((p, i) => UI.renderRecommendationCard(p, reasons[i % reasons.length])).join('');
        } else {
            hybridContainer.innerHTML = '<p style="color:var(--text-secondary)">No AI recommendations found.</p>';
        }

        // Customers Also Viewed (Category strategy as fallback)
        const catRes = await window.api.getRecommendations(productId, 'category_similarity', 10);
        const catContainer = document.getElementById('also-viewed-slider');
        if (!catRes.error && catRes.data && catRes.data.length > 0) {
            catContainer.innerHTML = catRes.data.map(p => UI.renderRecommendationCard(p, null)).join('');
        } else {
            catContainer.innerHTML = '<p style="color:var(--text-secondary)">No similar items found.</p>';
        }

        if (window.UI && window.UI.initAnimations) {
            window.UI.initAnimations();
        }
    } catch(e) {
        console.error("Failed to load recommendations", e);
        document.getElementById('ai-recommendations-slider').innerHTML = '<p style="color:var(--text-secondary)">Error loading recommendations.</p>';
        document.getElementById('also-viewed-slider').innerHTML = '<p style="color:var(--text-secondary)">Error loading recommendations.</p>';
    }
}
