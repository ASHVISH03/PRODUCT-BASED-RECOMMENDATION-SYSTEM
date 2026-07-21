/**
 * Theme Management
 * Handles Light/Dark mode toggling and localStorage persistence
 */

const ThemeManager = {
    init() {
        // Check saved preference or system preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            this.setTheme(savedTheme);
        } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            this.setTheme('dark');
        } else {
            this.setTheme('light');
        }

        // Attach listener to toggle button if it exists
        document.addEventListener('DOMContentLoaded', () => {
            const toggleBtn = document.getElementById('theme-toggle');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => this.toggleTheme());
            }
            this.updateIcon();
        });
    },

    setTheme(themeName) {
        document.documentElement.setAttribute('data-theme', themeName);
        localStorage.setItem('theme', themeName);
        this.updateIcon();
    },

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    },

    updateIcon() {
        const toggleBtn = document.getElementById('theme-toggle');
        if (!toggleBtn) return;
        
        const currentTheme = document.documentElement.getAttribute('data-theme');
        toggleBtn.innerHTML = currentTheme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    }
};

// Initialize immediately to prevent FOUC (Flash of Unstyled Content)
ThemeManager.init();
