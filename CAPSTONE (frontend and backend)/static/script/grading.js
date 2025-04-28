// grading.js
document.addEventListener('DOMContentLoaded', () => {
  const cards   = document.querySelectorAll('.card');
  const modals  = {
    a:        document.getElementById('modal-a'),
    b:        document.getElementById('modal-b'),
    c:        document.getElementById('modal-c'),
    rejected: document.getElementById('modal-rejected')
  };
  const closeBtns = document.querySelectorAll('.close-btn');

  cards.forEach(card => {
    card.addEventListener('click', () => {
      // hide any open modal
      Object.values(modals).forEach(m => m.classList.remove('show'));
      // show the one you clicked
      const grade = card.dataset.grade.toLowerCase();
      const modal = modals[grade];
      if (modal) modal.classList.add('show');
    });
  });

  // clicking ×
  closeBtns.forEach(btn =>
    btn.addEventListener('click', () => {
      btn.closest('.modal').classList.remove('show');
    })
  );

  // clicking outside
  window.addEventListener('click', e => {
    Object.values(modals).forEach(m => {
      if (e.target === m) m.classList.remove('show');
    });
  });

  // video controls (unchanged)
  const video   = document.getElementById('grading-video');
  const start   = document.getElementById('startBtn');
  const pause   = document.getElementById('pauseBtn');
  const stop    = document.getElementById('stopBtn');
  if (video && start && pause && stop) {
    start.addEventListener('click', () => video.play());
    pause.addEventListener('click', () => video.pause());
    stop.addEventListener('click', () => { video.pause(); video.currentTime = 0; });
  }
});
