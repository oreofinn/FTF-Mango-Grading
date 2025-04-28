// navigation.js - Handles loading and managing the navigation menu

document.addEventListener("DOMContentLoaded", function () {
    loadNavigationComponent();
});

function loadNavigationComponent() {
    fetch('navigation.html')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.text();
        })
        .then(html => {
            const navContainer = document.getElementById('navigation-container');
            if (navContainer) {
                navContainer.innerHTML = html;
                setupNavigationFeatures();
            }
        })
        .catch(error => {
            console.error('Failed to load navigation:', error);
            setupBasicNavigation();
        });
}

function setupNavigationFeatures() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item a').forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath.endsWith(linkPath)) {
            link.parentElement.classList.add('active');
        } else {
            link.parentElement.classList.remove('active');
        }
    });
}

function setupBasicNavigation() {
    const navContainer = document.getElementById('navigation-container');
    if (navContainer) {
        navContainer.innerHTML = `
            <div class="nav-header">
                <h1 class="nav-title">Mango Grading</h1>
            </div>
            <nav class="main-nav">
                <ul>
                    <li class="nav-item"><a href="/dashboard">Dashboard</a></li>
                    <li class="nav-item"><a href="/status">Grading Status</a></li>
                    <li class="nav-item"><a href="/report">Reports</a></li>
                    <li class="nav-item"><a href="/settings">Settings</a></li>
                </ul>
            </nav>
        `;
    }
}
