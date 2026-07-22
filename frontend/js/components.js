/**
 * Shared UI Components
 * Methods to generate HTML for reusable elements
 */

const UI = {
    /**
     * Silently test an image in memory to prevent 400 console errors in the DOM.
     */
    preloadImage(imgElement, src, fallbackSrc) {
        if (!src) {
            if (imgElement) imgElement.src = fallbackSrc;
            return;
        }
        const img = new Image();
        img.onload = () => {
            if (imgElement) {
                imgElement.src = src;
                imgElement.style.opacity = 1;
            }
        };
        img.onerror = () => {
            if (imgElement) {
                imgElement.src = fallbackSrc;
                imgElement.style.opacity = 1;
            }
        };
        img.src = src;
    },

    /**
     * Initialize IntersectionObserver for scroll animations (.fade-up, .scale-in)
     */
    initAnimations() {
        if (!window._animationObserver) {
            window._animationObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, { threshold: 0.1 });
        }
        document.querySelectorAll('.fade-up:not(.visible), .scale-in:not(.visible)').forEach(el => {
            window._animationObserver.observe(el);
        });
    },

    /**
     * Professional Toast Notification
     */
    showToast(message, icon = 'ph-check-circle', type = 'success') {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.style.cssText = `
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                z-index: 9999;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        toast.className = `toast-notification toast-${type}`;
        toast.style.cssText = `
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-md);
            padding: 1rem 1.5rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            pointer-events: auto;
            min-width: 300px;
            overflow: hidden;
            position: relative;
        `;

        const iconColor = type === 'success' ? '#10B981' : 'var(--brand-primary)';

        toast.innerHTML = `
            <i class="ph-fill ${icon}" style="color:${iconColor}; font-size:1.5rem"></i>
            <span style="font-weight:600; font-size:0.95rem">${message}</span>
            <div class="toast-progress" style="
                position: absolute;
                bottom: 0;
                left: 0;
                height: 3px;
                background: ${iconColor};
                width: 100%;
                animation: toastProgress 3s linear forwards;
            "></div>
        `;

        container.appendChild(toast);

        // Slide in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateY(0)';
            toast.style.opacity = '1';
        });

        // Auto dismiss
        setTimeout(() => {
            toast.style.transform = 'translateY(20px)';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 400);
        }, 3000);
    },

    /**
     * Generate dynamic premium SVG placeholder based on category
     */
    getPlaceholderSVG(category) {
        const catLower = (category || '').toLowerCase();
        let color1 = '#2a2a2a', color2 = '#111111', accent = '#555555';
        
        if(catLower.includes('phone') || catLower.includes('mobile')) { color1 = '#1e3a8a'; color2 = '#172554'; accent = '#60a5fa'; }
        else if(catLower.includes('laptop') || catLower.includes('computer')) { color1 = '#064e3b'; color2 = '#022c22'; accent = '#34d399'; }
        else if(catLower.includes('cable') || catLower.includes('usb')) { color1 = '#701a75'; color2 = '#4a044e'; accent = '#e879f9'; }
        else if(catLower.includes('gaming')) { color1 = '#7f1d1d'; color2 = '#450a0a'; accent = '#f87171'; }
        else if(catLower.includes('keyboard')) { color1 = '#831843'; color2 = '#4c0519'; accent = '#f472b6'; }
        else if(catLower.includes('television') || catLower.includes('tv')) { color1 = '#0f766e'; color2 = '#042f2e'; accent = '#2dd4bf'; }
        else if(catLower.includes('kitchen')) { color1 = '#854d0e'; color2 = '#422006'; accent = '#facc15'; }
        else if(catLower.includes('speaker') || catLower.includes('audio')) { color1 = '#5b21b6'; color2 = '#2e1065'; accent = '#c084fc'; }
        else if(catLower.includes('headphone')) { color1 = '#3730a3'; color2 = '#1e1b4b'; accent = '#818cf8'; }
        else if(catLower.includes('camera')) { color1 = '#075985'; color2 = '#082f49'; accent = '#38bdf8'; }
        else if(catLower.includes('fashion') || catLower.includes('clothing')) { color1 = '#9f1239'; color2 = '#4c0519'; accent = '#fb7185'; }
        else if(catLower.includes('accessori')) { color1 = '#115e59'; color2 = '#042f2e'; accent = '#2dd4bf'; }

        let title = (category || 'Premium Product').split('|')[0].substring(0, 20).toUpperCase();
        title = title.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        const svg = `
            <svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
                <defs>
                    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="${color1}" />
                        <stop offset="100%" stop-color="${color2}" />
                    </linearGradient>
                </defs>
                <rect width="400" height="400" fill="url(#g)" />
                <circle cx="200" cy="180" r="60" fill="none" stroke="${accent}" stroke-width="2" stroke-dasharray="4 4" opacity="0.3"/>
                <rect x="175" y="155" width="50" height="50" rx="12" fill="none" stroke="${accent}" stroke-width="4" opacity="0.8"/>
                <text x="200" y="270" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="600" fill="#ffffff" text-anchor="middle" letter-spacing="3" opacity="0.9">
                    ${title}
                </text>
            </svg>
        `;
        return `data:image/svg+xml;base64,${btoa(svg)}`;
    },

    /**
     * Render rating stars HTML
     */
    renderStars(rating = 4.5) {
        const fullStars = Math.round(parseFloat(rating) || 4.5);
        const emptyStars = Math.max(0, 5 - fullStars);
        return '<i class="ph-fill ph-star"></i>'.repeat(fullStars) + 
               '<i class="ph ph-star"></i>'.repeat(emptyStars);
    },

    /**
     * Generate HTML for a premium product card
     */
    renderProductCard(product, context = 'default') {
        window.globalProductCache = window.globalProductCache || {};
        window.globalProductCache[product.product_id] = product;

        const category = product.category ? product.category.split('|')[0] : '';
        let imgLink = UI.getPlaceholderSVG(category);
        if (product.img_link) {
            imgLink = product.img_link.split('|')[0].split(',')[0];
            if (imgLink.includes('m.media-amazon.com/images/W/')) {
                imgLink = imgLink.replace(/\/images\/W\/[^/]+\//, '/');
            }
        }
        
        let discountBadge = product.discount_percentage > 0 ? `<div class="badge badge-discount">-${Math.round(product.discount_percentage)}%</div>` : '';
        
        // Clean price string helper
        const cleanPrice = (val) => {
            if (!val) return 0;
            if (typeof val === 'number') return val;
            const parsed = parseFloat(String(val).replace(/[^0-9.]/g, ''));
            return isNaN(parsed) ? 0 : parsed;
        };
        
        // Resolve prices
        let price = cleanPrice(product.discounted_price) || cleanPrice(product.price);
        let originalPrice = cleanPrice(product.actual_price) || price;
        
        // Fallback for 0 prices (dataset edge cases)
        if (price === 0) price = originalPrice || 999;
        if (originalPrice === 0) originalPrice = price;

        let actualPriceStr = '';
        
        let displayPrice = `₹${price.toLocaleString()}`;
        
        if (originalPrice > price) {
            actualPriceStr = `<span class="product-price-actual">₹${originalPrice.toLocaleString()}</span>`;
            const discountPercent = Math.round(((originalPrice - price) / originalPrice) * 100);
            if (discountPercent > 0 && discountPercent < 100) {
                discountBadge = `<div class="badge badge-discount">-${discountPercent}%</div>`;
            }
        }

        // Ratings
        const rating = parseFloat(product.rating) || 4.0;
        const ratingCount = product.rating_count ? product.rating_count.toLocaleString() : '12';
        
        const fullStars = Math.round(rating);
        const emptyStars = 5 - fullStars;
        const starsHTML = 
            '<i class="ph-fill ph-star"></i>'.repeat(fullStars) + 
            '<i class="ph ph-star"></i>'.repeat(emptyStars);

        // Badges Logic
        let aiBadge = '';
        if (context === 'ai') {
            const grade = product.confidence_grade || 'High';
            const gradeClass = `badge-${grade.toLowerCase().replace(' ', '-')}`;
            aiBadge = `
                <div class="badge badge-ai">
                    <i class="ph-fill ph-sparkle" style="color:var(--brand-secondary)"></i> AI Pick
                </div>
            `;
        } else if (context === 'trending') {
            aiBadge = `
                <div class="badge badge-ai">
                    <i class="ph-fill ph-fire" style="color:#ff4500"></i> Trending
                </div>
            `;
        } else if (context === 'deals') {
            aiBadge = `
                <div class="badge badge-ai" style="color:#ff0080">
                    <i class="ph-fill ph-lightning"></i> Flash Deal
                </div>
            `;
        } else if (context === 'new') {
            aiBadge = `
                <div class="badge badge-ai" style="color:#3291ff">
                    <i class="ph-fill ph-star-four"></i> New Arrival
                </div>
            `;
        }

        const brand = product.brand ? product.brand : 'Premium Brand';
        const fallbackImg = UI.getPlaceholderSVG(category);

        return `
            <div class="product-card" onclick="if(!event.target.closest('.btn-icon') && !event.target.closest('.btn-action')) window.location.href='product.html?id=${encodeURIComponent(product.product_id)}'" style="cursor:pointer">
                ${discountBadge}
                ${aiBadge}
                <div class="product-image">
                    <img 
                        src="${fallbackImg}" 
                        alt="${product.product_name}" 
                        loading="lazy"
                        decoding="async"
                        style="opacity: 0.8; transition: opacity 0.5s ease-in;"
                        onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }"
                        onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }"
                    >
                </div>
                
                <div class="product-meta">
                    <span class="product-brand">${brand}</span>
                    ${category ? `<span style="font-size:0.8rem; color:var(--text-tertiary)">• ${category}</span>` : ''}
                </div>
                
                <a href="product.html?id=${encodeURIComponent(product.product_id)}" class="product-title" title="${product.product_name}">
                    ${product.product_name}
                </a>
                
                <div class="product-rating">
                    <span class="stars">${starsHTML}</span>
                    <span>${rating.toFixed(1)}</span>
                    <span style="color:var(--text-tertiary)">(${ratingCount})</span>
                </div>
                
                <div class="product-footer">
                    <div class="product-price">
                        ${displayPrice}
                        ${actualPriceStr}
                    </div>
                </div>

                <div class="product-actions">
                    <button class="btn-action primary" onclick="event.stopPropagation(); window.addToCart('${product.product_id}')">
                        <i class="ph ph-shopping-cart"></i> Add to Cart
                    </button>
                    <button class="btn-action" title="Quick View" onclick="event.stopPropagation(); window.openQuickView('${product.product_id}')">
                        <i class="ph ph-eye"></i> Quick View
                    </button>
                    <button class="btn-icon" title="Compare" onclick="event.stopPropagation(); alert('Compare logic pending')">
                        <i class="ph ph-arrows-left-right"></i>
                    </button>
                    <button class="btn-icon wishlist-btn" id="wishlist-btn-${product.product_id}" title="Wishlist" onclick="event.stopPropagation(); window.toggleWishlist('${product.product_id}')">
                        <i class="ph ph-heart"></i>
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Show a loading spinner inside a container
     */
    showLoading(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display:flex; justify-content:center; padding:4rem;">
                    <div class="spinner"></div>
                </div>
            `;
        }
    },

    /**
     * Show an error message inside a container
     */
    showError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="ph ph-warning empty-icon"></i>
                    <h3>Oops! Something went wrong.</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    },
    
    /**
     * Format Canonical Category String to human readable form
     */
    formatCategoryName(canonicalName) {
        if (!canonicalName) return '';
        return canonicalName
            .replace(/([a-z])([A-Z])/g, '$1 $2') // e.g. AirConditioners -> Air Conditioners
            .replace(/([A-Z])([A-Z][a-z])/g, '$1 $2') // e.g. 3DGlasses -> 3D Glasses
            .replace(/&/g, ' & ') // Add spaces around &
            .trim();
    },

    /**
     * Generate HTML for Category Card
     */
    renderCategoryCard(categoryName) {
        const formattedName = UI.formatCategoryName(categoryName);

        return `
            <a href="category.html?cat=${encodeURIComponent(categoryName)}" 
               style="display:flex; align-items:center; gap:1rem; background:var(--bg-surface); padding:1.25rem; border-radius:var(--radius-lg); border:1px solid var(--border-subtle); transition:all var(--duration-fast);"
               onmouseover="this.style.borderColor='var(--brand-secondary)'; this.style.transform='translateY(-4px)'; this.style.boxShadow='0 10px 20px rgba(0,0,0,0.2)';"
               onmouseout="this.style.borderColor='var(--border-subtle)'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                <div style="width:48px; height:48px; flex-shrink:0; border-radius:var(--radius-md); background:var(--bg-base); display:flex; align-items:center; justify-content:center; color:var(--brand-secondary);">
                    <i class="ph-fill ph-tag" style="font-size:1.5rem;"></i>
                </div>
                <span style="font-weight:600; font-size:0.95rem; line-height:1.3; color:var(--text-primary); word-break:break-word;">
                    ${formattedName}
                </span>
            </a>
        `;
    },

    /**
     * Generate HTML for an AI Recommendation Card with explainability badge
     */
    renderRecommendationCard(product, reason) {
        window.globalProductCache = window.globalProductCache || {};
        window.globalProductCache[product.product_id] = product;

        const category = product.category ? product.category.split('|')[0] : '';
        let imgLink = UI.getPlaceholderSVG(category);
        if (product.img_link) {
            imgLink = product.img_link.split('|')[0].split(',')[0];
            if (imgLink.includes('m.media-amazon.com/images/W/')) {
                imgLink = imgLink.replace(/\/images\/W\/[^/]+\//, '/');
            }
        }
        
        const fallbackImg = UI.getPlaceholderSVG(category);
        const discountBadge = product.discount_percentage > 0 ? `<div class="badge badge-discount">-${Math.round(product.discount_percentage)}%</div>` : '';
        const brand = product.brand ? product.brand : 'Premium Brand';
        
        const formatPrice = (price) => {
            return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(price);
        };

        const cleanPrice = (val) => {
            if (!val) return 0;
            if (typeof val === 'number') return val;
            const parsed = parseFloat(String(val).replace(/[^0-9.]/g, ''));
            return isNaN(parsed) ? 0 : parsed;
        };

        let rawPrice = cleanPrice(product.discounted_price) || cleanPrice(product.price) || cleanPrice(product.actual_price);
        if (rawPrice === 0 && window.globalProductCache && window.globalProductCache[product.product_id]) {
            const cached = window.globalProductCache[product.product_id];
            rawPrice = cleanPrice(cached.discounted_price) || cleanPrice(cached.price) || cleanPrice(cached.actual_price);
        }
        const currentPrice = rawPrice > 0 ? formatPrice(rawPrice) : '₹999';

        const ratingVal = parseFloat(product.rating) || (window.globalProductCache && window.globalProductCache[product.product_id] && window.globalProductCache[product.product_id].rating) || 4.2;
        const ratingCountRaw = product.rating_count || (window.globalProductCache && window.globalProductCache[product.product_id] && window.globalProductCache[product.product_id].rating_count) || 12;
        const ratingCountStr = typeof ratingCountRaw === 'number' ? ratingCountRaw.toLocaleString() : String(ratingCountRaw);

        const aiReasonBadge = reason ? `
            <div class="ai-reasoning-badge">
                <i class="ph-fill ph-sparkle"></i> ${reason}
            </div>
        ` : '';

        return `
            <div class="product-card" onclick="if(!event.target.closest('.btn-icon') && !event.target.closest('.btn-action')) window.location.href='product.html?id=${encodeURIComponent(product.product_id)}'" style="cursor:pointer; min-width: 280px; scroll-snap-align: start;">
                ${discountBadge}
                ${aiReasonBadge}
                <div class="product-image">
                    <img 
                        src="${fallbackImg}" 
                        alt="${product.product_name}" 
                        loading="lazy"
                        decoding="async"
                        style="opacity: 0.8; transition: opacity 0.5s ease-in;"
                        onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }"
                    >
                </div>
                
                <div class="product-meta">
                    <span class="product-brand">${brand}</span>
                </div>
                
                <a href="product.html?id=${encodeURIComponent(product.product_id)}" class="product-title" title="${product.product_name}">
                    ${product.product_name}
                </a>
                
                <div class="product-rating">
                    <div class="stars">${UI.renderStars(ratingVal)}</div>
                    <span>(${ratingCountStr})</span>
                </div>
                
                <div class="product-footer">
                    <div class="product-price">
                        ${currentPrice}
                    </div>
                </div>
                
                <div class="product-actions">
                    <button class="btn-action primary" onclick="event.stopPropagation(); window.addToCart('${product.product_id}')">
                        <i class="ph ph-shopping-cart"></i> Add to Cart
                    </button>
                    <button class="btn-icon wishlist-btn" id="wishlist-btn-${product.product_id}" title="Wishlist" onclick="event.stopPropagation(); window.toggleWishlist('${product.product_id}')">
                        <i class="${window.wishlist && window.wishlist.includes(product.product_id) ? 'ph-fill' : 'ph'} ph-heart" style="${window.wishlist && window.wishlist.includes(product.product_id) ? 'color:var(--brand-primary)' : ''}"></i>
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Parse query parameters from URL
     */
    getQueryParams() {
        const params = new URLSearchParams(window.location.search);
        const entries = {};
        for (const [key, value] of params.entries()) {
            entries[key] = value;
        }
        return entries;
    },

    /**
     * Universal Navbar Initialization
     */
    async initNavbar() {
        if (window._navbarInitialized) return; // Idempotent

        // 1. Ensure Cart & Wishlist Buttons work
        const cartBtn = document.querySelector('#nav-cart-btn, [data-action="open-cart"]');
        if (cartBtn) {
            cartBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.openCartDrawer) window.openCartDrawer();
            });
        }
        
        const wishlistBtn = document.querySelector('#nav-wishlist-btn, [data-action="open-wishlist"]');
        if (wishlistBtn) {
            wishlistBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.openWishlistDrawer) window.openWishlistDrawer();
            });
        }

        // 2. Ensure Search Autocomplete Container exists
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            let searchDropdown = document.getElementById('search-dropdown');
            if (!searchDropdown) {
                searchDropdown = document.createElement('div');
                searchDropdown.id = 'search-dropdown';
                searchDropdown.className = 'search-dropdown';
                searchInput.parentElement.appendChild(searchDropdown);
            }
        }
        
        // 3. Dynamically Populate Global Category Navigation
        const navContainer = document.getElementById('category-nav-container');
        if (navContainer && window.api) {
            try {
                const catRes = await window.api.getCategories();
                const categories = catRes.data || [];
                const cleanCats = [...new Set(categories.map(c => c.split('|')[0].trim()))].filter(c => c);
                
                let navHtml = '<a href="index.html" class="cat-link">All Products</a>';
                
                // Get current category from URL if any
                const params = new URLSearchParams(window.location.search);
                const currentCat = params.get('cat') || params.get('category');
                
                cleanCats.slice(0, 8).forEach(cat => {
                    const isActive = currentCat === cat ? 'active' : '';
                    navHtml += `<a href="category.html?cat=${encodeURIComponent(cat)}" class="cat-link ${isActive}">${UI.formatCategoryName(cat)}</a>`;
                });
                
                navContainer.innerHTML = navHtml;
                navContainer.classList.add('visible'); // If it uses a fade-in class
            } catch (err) {
                console.error('Failed to load global category nav:', err);
                navContainer.innerHTML = '<a href="index.html" class="cat-link">All Products</a>';
            }
        }
        
        window._navbarInitialized = true;
    }
};

window.UI = UI;

/**
 * Universal Navbar Autocomplete Logic
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize standard buttons
    UI.initNavbar();

    const searchInput = document.getElementById('search-input');
    const searchDropdown = document.getElementById('search-dropdown');
    const voiceIcon = document.querySelector('.voice-icon');
    
    if (!searchInput || !searchDropdown) return;

    let debounceTimer;
    let currentFocus = -1;
    const suggestionCache = {};

    // Voice Search
    if (voiceIcon) {
        voiceIcon.onclick = null; // Remove inline onclick
        voiceIcon.addEventListener('click', () => {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {
                const recognition = new SpeechRecognition();
                const originalPlaceholder = searchInput.placeholder;
                
                recognition.onstart = () => {
                    voiceIcon.classList.add('ph-waveform');
                    voiceIcon.classList.remove('ph-microphone');
                    voiceIcon.style.color = '#ff0000';
                    searchInput.placeholder = "Listening...";
                    searchInput.value = "";
                };
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    searchInput.value = transcript;
                    window.location.href = `search.html?q=${encodeURIComponent(transcript)}`;
                };
                recognition.onend = () => {
                    voiceIcon.classList.remove('ph-waveform');
                    voiceIcon.classList.add('ph-microphone');
                    voiceIcon.style.color = 'var(--brand-secondary)';
                    searchInput.placeholder = originalPlaceholder;
                };
                recognition.start();
            } else {
                alert('Voice search is not supported by your browser.');
            }
        });
    }

    const rankResults = (items, query) => {
        const q = query.toLowerCase();
        return items.sort((a, b) => {
            const aName = (a.product_name || '').toLowerCase();
            const bName = (b.product_name || '').toLowerCase();
            const aBrand = (a.brand || '').toLowerCase();
            const bBrand = (b.brand || '').toLowerCase();
            const aCat = (a.category || '').toLowerCase();
            const bCat = (b.category || '').toLowerCase();

            // Scoring system: lower is better
            const getScore = (name, brand, cat) => {
                if (name.startsWith(q)) return 1; // Prefix match
                if (brand === q || brand.startsWith(q)) return 2; // Brand match
                if (name.includes(` ${q}`)) return 3; // Word match in title
                if (cat.includes(q)) return 4; // Category match
                if (name.includes(q)) return 5; // Substring match
                return 6;
            };

            return getScore(aName, aBrand, aCat) - getScore(bName, bBrand, bCat);
        });
    };

    const openSearchDropdown = () => {
        searchDropdown.classList.add('active');
        document.body.classList.add('search-open');
        const navSearch = searchInput.closest('.nav-search');
        if (navSearch) navSearch.classList.add('search-active');
    };

    const closeSearchDropdown = () => {
        searchDropdown.classList.remove('active');
        document.body.classList.remove('search-open');
        const navSearch = searchInput.closest('.nav-search');
        if (navSearch) navSearch.classList.remove('search-active');
        currentFocus = -1;
    };

    // Stop event propagation on dropdown container to prevent clicks from leaking to underlying category links
    ['click', 'mousedown', 'pointerdown', 'touchstart'].forEach(evt => {
        searchDropdown.addEventListener(evt, (e) => {
            e.stopPropagation();
        });
    });

    const renderDropdown = (items, query) => {
        if (!items || items.length === 0) {
            closeSearchDropdown();
            return;
        }

        const ranked = rankResults(items, query);
        currentFocus = -1;

        let html = `<div class="dropdown-section-title">Search Suggestions</div>`;
        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        ranked.slice(0, 6).forEach((item, index) => {
            const highlighted = item.product_name.replace(new RegExp(`(${escapedQuery})`, 'gi'), '<strong style="color:var(--brand-primary);">$1</strong>');
            html += `
                <a href="product.html?id=${encodeURIComponent(item.product_id)}" class="dropdown-item" id="suggestion-${index}" onclick="event.stopPropagation();">
                    <i class="ph ph-magnifying-glass dropdown-item-icon"></i>
                    <span class="dropdown-item-text">
                        ${highlighted}
                    </span>
                </a>
            `;
        });
        searchDropdown.innerHTML = html;
        openSearchDropdown();
    };

    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const query = e.target.value.trim();
        
        if (query.length < 2) {
            closeSearchDropdown();
            return;
        }

        if (suggestionCache[query.toLowerCase()]) {
            renderDropdown(suggestionCache[query.toLowerCase()], query);
            return;
        }

        debounceTimer = setTimeout(async () => {
            try {
                const res = await api.searchProducts(query, 20);
                suggestionCache[query.toLowerCase()] = res.data;
                renderDropdown(res.data, query);
            } catch (err) {
                console.error("Autocomplete failed:", err);
            }
        }, 300);
    });

    searchInput.addEventListener('keydown', (e) => {
        const items = searchDropdown.querySelectorAll('.dropdown-item');
        if (!searchDropdown.classList.contains('active') || items.length === 0) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = searchInput.value.trim();
                if (query) window.location.href = `search.html?q=${encodeURIComponent(query)}`;
            }
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            currentFocus++;
            if (currentFocus >= items.length) currentFocus = 0;
            updateActiveStatus(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            currentFocus--;
            if (currentFocus < 0) currentFocus = items.length - 1;
            updateActiveStatus(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (currentFocus > -1 && items[currentFocus]) {
                window.location.href = items[currentFocus].getAttribute('href');
            } else {
                const query = searchInput.value.trim();
                if (query) window.location.href = `search.html?q=${encodeURIComponent(query)}`;
            }
        } else if (e.key === 'Escape') {
            closeSearchDropdown();
        }
    });

    const updateActiveStatus = (items) => {
        items.forEach(item => item.classList.remove('focused'));
        if (currentFocus >= 0 && currentFocus < items.length) {
            items[currentFocus].classList.add('focused');
            items[currentFocus].scrollIntoView({ block: 'nearest' });
        }
    };

    // Close dropdown on click outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchDropdown.contains(e.target)) {
            closeSearchDropdown();
        }
    });
});

/**
 * Cart & Wishlist Logic
 */
window.wishlist = JSON.parse(localStorage.getItem('amazclone_wishlist')) || [];
window.cart = JSON.parse(localStorage.getItem('amazclone_cart')) || [];

window.toggleWishlist = (productId) => {
    if (!productId) return;
    window.wishlist = JSON.parse(localStorage.getItem('amazclone_wishlist')) || [];
    const idx = window.wishlist.indexOf(productId);
    let isWishlisted = false;
    if (idx > -1) {
        window.wishlist = window.wishlist.filter(id => id !== productId);
        UI.showToast('Removed from Wishlist', 'ph-heart-break', 'success');
        if (window.Interactions && window.Interactions.removeWishlist) {
            window.Interactions.removeWishlist(productId);
        }
    } else {
        if (!window.wishlist.includes(productId)) {
            window.wishlist.push(productId);
        }
        isWishlisted = true;
        UI.showToast('Added to Wishlist', 'ph-heart', 'success');
        if (window.Interactions && window.Interactions.add) {
            window.Interactions.add(productId, 'wishlist');
        }
    }
    localStorage.setItem('amazclone_wishlist', JSON.stringify(window.wishlist));
    
    // Update button UI if currently visible
    const btn = document.getElementById(`wishlist-btn-${productId}`);
    if (btn) {
        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = isWishlisted ? 'ph-fill ph-heart' : 'ph ph-heart';
            icon.style.color = isWishlisted ? 'var(--brand-primary)' : 'inherit';
        }
        if (btn.classList.contains('btn-ghost')) {
            btn.innerHTML = `<i class="${isWishlisted ? 'ph-fill' : 'ph'} ph-heart" style="${isWishlisted ? 'color:var(--brand-primary)' : ''}"></i> ${isWishlisted ? 'Saved to Wishlist' : 'Add to Wishlist'}`;
        }
    }
};

window.addToCart = (productId) => {
    if (!window.cart.includes(productId)) {
        window.cart.push(productId);
        localStorage.setItem('amazclone_cart', JSON.stringify(window.cart));
    }
    UI.showToast('Added to Cart', 'ph-shopping-cart', 'success');
    if (window.Interactions && window.Interactions.add) {
        window.Interactions.add(productId, 'cart');
    }

    // Bounce animation on the cart button in navbar
    const cartBtns = Array.from(document.querySelectorAll('a')).filter(a => a.innerHTML.includes('Cart'));
    cartBtns.forEach(btn => {
        btn.classList.add('badge-animating');
        setTimeout(() => btn.classList.remove('badge-animating'), 400);
    });
};

const createDrawer = (id, title, items, emptyText) => {
    let drawer = document.getElementById(id);
    if (!drawer) {
        drawer = document.createElement('div');
        drawer.id = id;
        drawer.style.cssText = `
            position: fixed; top: 0; right: -400px; width: 400px; height: 100%;
            background: var(--bg-surface); z-index: 10000; display: flex; flex-direction: column;
            box-shadow: -5px 0 25px rgba(0,0,0,0.5); transition: right 0.3s ease; border-left: 1px solid var(--border-subtle);
        `;
        document.body.appendChild(drawer);

        const overlay = document.createElement('div');
        overlay.id = `${id}-overlay`;
        overlay.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); z-index: 9999; opacity: 0; pointer-events: none;
            transition: opacity 0.3s ease; backdrop-filter: blur(2px);
        `;
        document.body.appendChild(overlay);

        overlay.addEventListener('click', () => {
            drawer.style.right = '-400px';
            overlay.style.opacity = '0';
            overlay.style.pointerEvents = 'none';
        });
    }

    const overlay = document.getElementById(`${id}-overlay`);
    
    let itemsHTML = '';
    let subtotal = 0;

    if (items.length === 0) {
        itemsHTML = `
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-tertiary);">
                <i class="ph-duotone ph-package" style="font-size: 4rem; margin-bottom: 1rem;"></i>
                <p>${emptyText}</p>
            </div>
        `;
    } else {
        itemsHTML = `<div style="flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1.5rem;">`;
        items.forEach((pid, index) => {
            const product = window.globalProductCache[pid];
            if (product) {
                const category = product.category ? product.category.split('|')[0] : '';
                let imgLink = UI.getPlaceholderSVG(category);
                if (product.img_link) {
                    imgLink = product.img_link.split('|')[0].split(',')[0];
                    if (imgLink.includes('m.media-amazon.com/images/W/')) {
                        imgLink = imgLink.replace(/\/images\/W\/[^/]+\//, '/');
                    }
                }
                const fallbackImg = UI.getPlaceholderSVG(category);
                const priceNum = parseFloat(String(product.discounted_price || product.actual_price || 999).replace(/[^0-9.]/g, ''));
                subtotal += priceNum;

                itemsHTML += `
                    <div style="display: flex; gap: 1rem; background: var(--bg-base); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); position: relative;">
                        <img src="${fallbackImg}" onload="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }" onerror="if(!this.dataset.loaded){ this.dataset.loaded='true'; UI.preloadImage(this, '${imgLink}', '${fallbackImg}'); }" style="width: 80px; height: 80px; object-fit: contain; border-radius: var(--radius-sm); background: #ffffff; padding: 0.25rem;">
                        <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.3;">${product.product_name}</div>
                            <div class="stars" style="color:#F59E0B; font-size: 0.8rem; margin-bottom: 0.5rem;">${UI.renderStars(product.rating || 4.5)}</div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="color: var(--brand-secondary); font-weight: 700; font-size: 1.1rem;">₹${priceNum.toLocaleString()}</div>
                                <div style="display: flex; align-items: center; gap: 0.5rem; background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 0.2rem;">
                                    <button style="background:none; border:none; color:var(--text-primary); cursor:pointer; padding: 0.2rem 0.5rem;"><i class="ph ph-minus"></i></button>
                                    <span style="font-size:0.9rem; font-weight:600; min-width: 1rem; text-align:center;">1</span>
                                    <button style="background:none; border:none; color:var(--text-primary); cursor:pointer; padding: 0.2rem 0.5rem;"><i class="ph ph-plus"></i></button>
                                </div>
                            </div>
                        </div>
                        <button onclick="${id === 'cart-drawer' ? `window.removeFromCart('${pid}')` : `window.removeFromWishlist('${pid}')`}" style="position: absolute; top: -0.5rem; right: -0.5rem; background: var(--bg-surface); border: 1px solid var(--border-strong); border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; color: var(--text-tertiary); transition: all 0.2s;"><i class="ph ph-x"></i></button>
                    </div>
                `;
            } else {
                itemsHTML += `<div style="padding: 1rem; background: var(--bg-base); border-radius: var(--radius-md);">Product ${pid}</div>`;
            }
        });
        
        if (id === 'cart-drawer') {
            itemsHTML += `
                <div style="background: var(--bg-surface); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: var(--text-secondary);">
                        <span>Subtotal</span>
                        <span style="font-weight: 600; color: var(--text-primary);">₹${subtotal.toLocaleString()}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: var(--text-secondary);">
                        <span>Shipping</span>
                        <span style="font-weight: 600; color: #10B981;">FREE</span>
                    </div>
                    <div style="border-top: 1px dashed var(--border-strong); margin: 0.5rem 0; padding-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 1.1rem; color: var(--text-primary);">Total</span>
                        <span style="font-weight: 700; font-size: 1.25rem; color: var(--brand-secondary);">₹${subtotal.toLocaleString()}</span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #10B981; display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-top: 0.5rem;">
                    <i class="ph-fill ph-truck"></i> Estimated Delivery: Tomorrow by 9 PM
                </div>
            `;
        }
        
        itemsHTML += `</div>`;
    }

    drawer.innerHTML = `
        <div style="padding: 1.5rem; border-bottom: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem;">
                ${title} <span class="badge" style="background: var(--brand-secondary); color: var(--bg-base); padding: 0.2rem 0.6rem;">${items.length}</span>
            </h3>
            <button onclick="document.getElementById('${id}').style.right = '-400px'; document.getElementById('${id}-overlay').style.opacity = '0'; document.getElementById('${id}-overlay').style.pointerEvents = 'none';" style="background: none; border: none; font-size: 1.5rem; color: var(--text-secondary); cursor: pointer;"><i class="ph ph-x"></i></button>
        </div>
        ${itemsHTML}
        ${items.length > 0 && id === 'cart-drawer' ? `
            <div style="padding: 1.5rem; border-top: 1px solid var(--border-subtle); background: var(--bg-surface);">
                <button class="btn btn-primary" style="width: 100%; padding: 1rem; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.5px;" onclick="window.location.href='checkout.html'">
                    Proceed to Checkout
                </button>
            </div>
        ` : ''}
    `;

    overlay.style.opacity = '1';
    overlay.style.pointerEvents = 'auto';
    setTimeout(() => drawer.style.right = '0', 50);
};

window.removeFromWishlist = (productId) => {
    if (!productId) return;
    window.wishlist = (JSON.parse(localStorage.getItem('amazclone_wishlist')) || []).filter(id => id !== productId);
    localStorage.setItem('amazclone_wishlist', JSON.stringify(window.wishlist));
    if (window.Interactions && window.Interactions.removeWishlist) {
        window.Interactions.removeWishlist(productId);
    }
    const btn = document.getElementById(`wishlist-btn-${productId}`);
    if (btn) {
        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = 'ph ph-heart';
            icon.style.color = 'inherit';
        }
    }
    if (window.openWishlistDrawer) {
        window.openWishlistDrawer();
    }
};

window.removeFromCart = (productId) => {
    if (!productId) return;
    window.cart = (JSON.parse(localStorage.getItem('amazclone_cart')) || []).filter(id => id !== productId);
    localStorage.setItem('amazclone_cart', JSON.stringify(window.cart));
    if (window.openCartDrawer) {
        window.openCartDrawer();
    }
};

window.openWishlistDrawer = async () => {
    window.wishlist = JSON.parse(localStorage.getItem('amazclone_wishlist')) || [];
    const missingPids = (window.wishlist || []).filter(pid => typeof pid === 'string' && pid && (!window.globalProductCache || !window.globalProductCache[pid]));
    if (missingPids.length > 0 && window.api && window.api.getProduct) {
        await Promise.all(missingPids.map(async (pid) => {
            try {
                const res = await window.api.getProduct(pid);
                if (!res.error && res.data) {
                    window.globalProductCache = window.globalProductCache || {};
                    window.globalProductCache[pid] = res.data;
                }
            } catch (e) {}
        }));
    }
    createDrawer('wishlist-drawer', '<i class="ph-fill ph-heart" style="color: var(--brand-primary);"></i> Wishlist', window.wishlist, 'No saved products yet.');
};

window.openCartDrawer = async () => {
    window.cart = JSON.parse(localStorage.getItem('amazclone_cart')) || [];
    const missingPids = (window.cart || []).filter(pid => typeof pid === 'string' && pid && (!window.globalProductCache || !window.globalProductCache[pid]));
    if (missingPids.length > 0 && window.api && window.api.getProduct) {
        await Promise.all(missingPids.map(async (pid) => {
            try {
                const res = await window.api.getProduct(pid);
                if (!res.error && res.data) {
                    window.globalProductCache = window.globalProductCache || {};
                    window.globalProductCache[pid] = res.data;
                }
            } catch (e) {}
        }));
    }
    createDrawer('cart-drawer', '<i class="ph-fill ph-shopping-cart"></i> Cart', window.cart, 'Your cart is empty.');
};

/**
 * Quick View Logic
 */
window.openQuickView = (productId) => {
    const product = window.globalProductCache[productId];
    if (!product) return;

    let modal = document.getElementById('quick-view-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'quick-view-modal';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); z-index: 9999; display: flex;
            align-items: center; justify-content: center; opacity: 0;
            pointer-events: none; transition: opacity 0.3s ease; backdrop-filter: blur(5px);
        `;
        document.body.appendChild(modal);
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) window.closeQuickView();
        });
    }

    const category = product.category ? product.category.split('|')[0] : '';
    const imgLink = product.img_link || UI.getPlaceholderSVG(category);
    const price = product.discounted_price || product.price || 999;

    modal.innerHTML = `
        <div style="background: var(--bg-surface); padding: 2rem; border-radius: var(--radius-lg); max-width: 800px; width: 90%; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; position: relative; border: 1px solid var(--border-subtle);">
            <button onclick="window.closeQuickView()" style="position: absolute; top: 1rem; right: 1rem; background: none; border: none; font-size: 1.5rem; color: var(--text-secondary); cursor: pointer;"><i class="ph ph-x"></i></button>
            
            <div style="background: var(--bg-base); border-radius: var(--radius-md); padding: 1rem; display: flex; align-items: center; justify-content: center;">
                <img src="${imgLink}" alt="Product" style="max-width: 100%; max-height: 300px; object-fit: contain;">
            </div>
            
            <div style="display: flex; flex-direction: column; justify-content: center;">
                <span style="color: var(--brand-secondary); font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">${category}</span>
                <h2 style="font-size: 1.5rem; margin: 0.5rem 0; color: var(--text-primary); line-height: 1.3;">${product.product_name}</h2>
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <span style="font-size: 2rem; font-weight: 700; color: var(--text-primary);">₹${parseFloat(String(price).replace(/[^0-9.]/g, '')).toLocaleString()}</span>
                    <span style="color: var(--text-secondary);"><i class="ph-fill ph-star" style="color: #F59E0B;"></i> ${parseFloat(product.rating || 4.0).toFixed(1)}</span>
                </div>
                
                <p style="color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6; margin-bottom: 2rem; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;">
                    ${product.about_product || 'Premium product with excellent features and quality.'}
                </p>
                
                <div style="display: flex; gap: 1rem;">
                    <button class="btn btn-primary" style="flex: 1;" onclick="window.addToCart('${productId}')">
                        <i class="ph ph-shopping-cart"></i> Add to Cart
                    </button>
                    <button class="btn btn-secondary" onclick="window.toggleWishlist('${productId}')">
                        <i class="ph ph-heart"></i>
                    </button>
                </div>
            </div>
        </div>
    `;

    modal.style.opacity = '1';
    modal.style.pointerEvents = 'auto';
};

window.closeQuickView = () => {
    const modal = document.getElementById('quick-view-modal');
    if (modal) {
        modal.style.opacity = '0';
        modal.style.pointerEvents = 'none';
    }
};

