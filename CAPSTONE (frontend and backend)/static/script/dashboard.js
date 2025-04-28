// dashboard.js - Handles charts on the index.html Dashboard page

document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById('stockChart')) {
        initDashboardCharts();
    }
});

function initDashboardCharts() {
    // Stock Distribution Chart
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

    // Weekly Grading Trend Chart
    new Chart(document.getElementById('trendChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            datasets: [
                {
                    label: 'Grade A',
                    data: [1200, 1300, 1100, 1150],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Grade B',
                    data: [800, 900, 850, 950],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Grade C',
                    data: [300, 400, 350, 320],
                    borderColor: '#F59E0B',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Rejected',
                    data: [150, 180, 130, 170],
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
                    position: 'top',
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

    // Seasonal Performance Chart
    new Chart(document.getElementById('seasonalChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Spring', 'Summer', 'Fall', 'Winter'],
            datasets: [
                {
                    label: 'Grade A',
                    data: [300, 500, 200, 400],
                    backgroundColor: '#10B981'
                },
                {
                    label: 'Grade B',
                    data: [200, 300, 150, 250],
                    backgroundColor: '#3B82F6'
                },
                {
                    label: 'Grade C',
                    data: [100, 150, 80, 100],
                    backgroundColor: '#F59E0B'
                },
                {
                    label: 'Rejected',
                    data: [50, 60, 40, 30],
                    backgroundColor: '#EF4444'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
