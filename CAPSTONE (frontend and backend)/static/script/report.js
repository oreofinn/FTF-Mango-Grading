// report.js - Handles tab switching and (future) export/search for report.html

document.addEventListener("DOMContentLoaded", function () {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", function () {
            // Deactivate all tabs
            tabButtons.forEach(t => t.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            // Activate current tab
            const target = btn.getAttribute("onclick").match(/'(.*?)'/)[1];
            btn.classList.add("active");
            document.getElementById(target).classList.add("active");
        });
    });

    // Placeholder for export functionality
    document.querySelectorAll(".export-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            alert("Export functionality not implemented yet.");
        });
    });

    // Placeholder search logic
    const searchBtn = document.getElementById("search-btn");
    if (searchBtn) {
        searchBtn.addEventListener("click", function () {
            const searchTerm = document.getElementById("search-input").value.toLowerCase();
            const rows = document.querySelectorAll("#grading-table tbody tr");
            rows.forEach(row => {
                row.style.display = row.textContent.toLowerCase().includes(searchTerm) ? "" : "none";
            });
        });
    }
});
