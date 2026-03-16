/**
 * SLAMAdverserialLab - Interactive Components
 * Image comparison sliders and utility functions
 */

document.addEventListener('DOMContentLoaded', function() {
  initImageCompareSliders();
});

/**
 * Initialize all image comparison sliders on the page
 */
function initImageCompareSliders() {
  const compareContainers = document.querySelectorAll('.compare-container');

  compareContainers.forEach(container => {
    const afterImage = container.querySelector('.compare-after');
    const slider = container.querySelector('.compare-slider');

    if (!afterImage || !slider) return;

    let isDragging = false;

    // Set initial position to 50%
    updateSliderPosition(container, afterImage, slider, 50);

    // Prevent default image drag behavior
    const images = container.querySelectorAll('img');
    images.forEach(img => {
      img.addEventListener('dragstart', (e) => e.preventDefault());
      img.style.userSelect = 'none';
      img.style.pointerEvents = 'none';
    });

    // Mouse events
    container.addEventListener('mousedown', (e) => {
      e.preventDefault();
      isDragging = true;
      updatePosition(e, container, afterImage, slider);
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      e.preventDefault();
      updatePosition(e, container, afterImage, slider);
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Touch events
    container.addEventListener('touchstart', (e) => {
      isDragging = true;
      updatePosition(e.touches[0], container, afterImage, slider);
    });

    container.addEventListener('touchmove', (e) => {
      if (!isDragging) return;
      e.preventDefault();
      updatePosition(e.touches[0], container, afterImage, slider);
    });

    container.addEventListener('touchend', () => {
      isDragging = false;
    });

    // Click to move slider
    container.addEventListener('click', (e) => {
      updatePosition(e, container, afterImage, slider);
    });
  });
}

/**
 * Update slider position based on mouse/touch event
 */
function updatePosition(e, container, afterImage, slider) {
  const rect = container.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

  updateSliderPosition(container, afterImage, slider, percentage);
}

/**
 * Update the visual position of the slider and clipped image
 */
function updateSliderPosition(container, afterImage, slider, percentage) {
  // Update the clip-path of the "after" image (shows original/before on left)
  afterImage.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;

  // Update slider position
  slider.style.left = percentage + '%';
}

/**
 * Copy BibTeX citation to clipboard
 */
function copyBibtex() {
  const bibtex = `@article{hefny2024slamadverseriallab,
  title={SLAMAdverserialLab: An Extensible Framework for Visual SLAM
         Robustness Evaluation under Adverse Conditions},
  author={Hefny, Mohamed and Ko, Steven and Dantu, Karthik},
  journal={arXiv preprint},
  year={2024}
}`;

  navigator.clipboard.writeText(bibtex).then(() => {
    const btn = document.querySelector('.copy-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    btn.style.background = '#28a745';

    setTimeout(() => {
      btn.innerHTML = originalText;
      btn.style.background = '';
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy: ', err);
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = bibtex;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);

    const btn = document.querySelector('.copy-btn');
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    setTimeout(() => {
      btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
    }, 2000);
  });
}

/**
 * Lazy load GIFs when they come into view
 */
function initLazyLoading() {
  const images = document.querySelectorAll('img[loading="lazy"]');

  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          if (img.dataset.src) {
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
          }
          observer.unobserve(img);
        }
      });
    }, {
      rootMargin: '50px 0px'
    });

    images.forEach(img => imageObserver.observe(img));
  }
}

/**
 * Smooth scroll for anchor links
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    if (href === '#') return;

    e.preventDefault();
    const target = document.querySelector(href);
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});
