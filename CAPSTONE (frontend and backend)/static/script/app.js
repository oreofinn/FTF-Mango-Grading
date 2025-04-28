// Mango Grading System - Main Application Script

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Load navigation component
    loadNavigationComponent();

    // Initialize page-specific functionality
    initPageFeatures();
});

// Load and setup navigation
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
            // Fallback to basic navigation if needed
            setupBasicNavigation();
        });
}

// Setup navigation features
function setupNavigationFeatures() {
    // Highlight the active navigation item based on the current page
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item a').forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath.endsWith(linkPath)) {
            link.parentElement.classList.add('active');
        } else {
            link.parentElement.classList.remove('active');
        }
    });

    // Add event listener for sidebar toggle (if applicable)
    const toggleBtn = document.getElementById('toggleSidebar');
    const sidebar = document.getElementById('sidebar');
    if (toggleBtn && sidebar) {
        // Restore saved state
        if (localStorage.getItem('sidebarCollapsed') === 'true') {
            sidebar.classList.add('collapsed');
        }

        // Setup click handler
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
        });
    }
}

// Fallback basic navigation
function setupBasicNavigation() {
    console.warn('Using basic navigation fallback');
    const navContainer = document.getElementById('navigation-container');
    if (navContainer) {
        navContainer.innerHTML = `
            <div class="nav-header">
                <h1 class="nav-title">Mango Grading</h1>
            </div>
            <nav class="main-nav">
                <ul>
                    <li class="nav-item"><a href="index.html">Dashboard</a></li>
                    <li class="nav-item"><a href="status.html">Grading Status</a></li>
                    <li class="nav-item"><a href="report.html">Reports</a></li>
                    <li class="nav-item"><a href="settings.html">Settings</a></li>
                </ul>
            </nav>
        `;
    }
}

// Initialize page-specific features
function initPageFeatures() {
    // Dashboard page charts
    if (document.getElementById('stockChart')) {
        initDashboardCharts();
    }

    // Status page functionality
    if (document.getElementById('date-display')) {
        document.getElementById('date-display').textContent = new Date().toLocaleDateString();
    }

    // Report page charts
    if (document.getElementById('gradeChart')) {
        initReportCharts();
    }
}

// Dashboard charts initialization
function initDashboardCharts() {
    // Stock distribution chart
    new Chart(document.getElementById('stockChart').getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Grade A', 'Grade B', 'Grade C', 'Rejected'],
            datasets: [{
                data: [1250, 850, 320, 180],
                backgroundColor: ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
            }]
        }
    });

    // Weekly trend chart
    new Chart(document.getElementById('trendChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Grade A',
                data: [120, 190, 170, 210, 180, 150, 200],
                borderColor: '#10B981'
            }]
        }
    });
}


document.addEventListener("DOMContentLoaded", function () {
    const cards = document.querySelectorAll(".card");
    const modals = {
        a: document.getElementById("modal-a"),
        b: document.getElementById("modal-b"),
        c: document.getElementById("modal-c"),
        rejected: document.getElementById("modal-rejected")
    };

    const closeButtons = document.querySelectorAll(".close-btn");

    // Price simulation data (can be dynamic)
    const gradePrices = {
        a: 50,
        b: 35,
        c: 20,
        rejected: -5 // loss
    };
    const gradeQuantities = {
        a: 120,
        b: 850,
        c: 320,
        rejected: 180
    };

    // Show modals
    cards.forEach(card => {
        card.addEventListener("click", () => {
            const grade = card.dataset.grade;
            const modal = modals[grade];
            if (modal) {
                modal.style.display = "block";
                const profitSpan = document.getElementById(`profit-${grade}`);
                const total = gradePrices[grade] * gradeQuantities[grade];
                profitSpan.textContent = total.toLocaleString();
            }
        });
    });

    // Close modals
    closeButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            btn.closest(".modal").style.display = "none";
        });
    });

    // Close modal when clicking outside
    window.addEventListener("click", function (e) {
        Object.values(modals).forEach(modal => {
            if (e.target === modal) {
                modal.style.display = "none";
            }
        });
    });
});
// Initialize Weekly Grading Trend Chart
const trendChartCtx = document.getElementById('trendChart').getContext('2d');
const trendChart = new Chart(trendChartCtx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'], // Example weekly labels
        datasets: [
            {
                label: 'Grade A',
                data: [1200, 1300, 1100, 1150], // Data for Grade A
                borderColor: '#10B981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.3
            },
            {
                label: 'Grade B',
                data: [800, 900, 850, 950], // Data for Grade B
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3
            },
            {
                label: 'Grade C',
                data: [300, 400, 350, 320], // Data for Grade C
                borderColor: '#F59E0B',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                fill: true,
                tension: 0.3
            },
            {
                label: 'Rejected',
                data: [150, 180, 130, 170], // Data for Rejected
                borderColor: '#EF4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.3
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top', // Display the legend at the top of the chart
            }
        },
        scales: {
            x: {
                beginAtZero: true
            },
            y: {
                beginAtZero: true
            }
        }
    }
});