// status.js - Handles logic for status.html page

document.addEventListener("DOMContentLoaded", function () {
    const dateDisplay = document.getElementById("date-display");
    if (dateDisplay) {
        dateDisplay.textContent = new Date().toLocaleDateString();
    }

    const form = document.querySelector(".grading-form");
    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            alert("Form submitted! (Add your logic to handle this)");
        });
    }
});
