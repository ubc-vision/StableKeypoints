window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation";
var NUM_INTERP_FRAMES = 280;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path =
      INTERP_BASE + "/" + "keypoint_" + String(i).padStart(5, "0") + ".jpg";
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function () {
    return false;
  };
  image.oncontextmenu = function () {
    return false;
  };
  $("#interpolation-image-wrapper").empty().append(image);
}

// Global variables for the auto-scroll timer, dragging, and hovering states
var autoScrollTimer;
var isDragging = false;
var isHovering = false;

function resetAutoScrollTimer() {
  clearInterval(autoScrollTimer); // Clear existing timer
  autoScrollTimer = setInterval(autoScrollSlider, 30); // Set new timer
}

function autoScrollSlider() {
  if (!isDragging && !isHovering) {
    // Only auto-scroll if not dragging and not hovering
    var slider = $("#interpolation-slider");
    var currentValue = parseInt(slider.val());
    var nextValue = currentValue + 1;

    if (nextValue >= NUM_INTERP_FRAMES) {
      // Reached the end, pause for 3 seconds, then loop back to start
      clearInterval(autoScrollTimer);
      setTimeout(function () {
        slider.val(0).trigger("input");
        resetAutoScrollTimer(); // Reset timer with standard interval
      }, 3000);
    } else {
      // Not yet at the end, just increment
      slider.val(nextValue).trigger("input");
    }
  }
}

$(document).ready(function () {
  resetAutoScrollTimer(); // Start auto-scrolling the slider

  // Event handler for starting slider interaction
  $("#interpolation-slider").on("mousedown touchstart", function () {
    clearInterval(autoScrollTimer);
    isDragging = true; // Start dragging
  });

  // Event handler for ending slider interaction
  $("#interpolation-slider").on("mouseup touchend", function () {
    isDragging = false; // Stop dragging
    if (!isHovering) {
      resetAutoScrollTimer(); // Reset timer if not hovering
    }
  });

  // Event handlers for mouse hover
  $("#interpolation-slider").on("mouseenter", function () {
    clearInterval(autoScrollTimer); // Pause on hover
    isHovering = true; // Start hovering
  });

  $("#interpolation-slider").on("mouseleave", function () {
    isHovering = false; // Stop hovering
    if (!isDragging) {
      // Resume only if not dragging
      resetAutoScrollTimer();
    }
  });

  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Options for your existing carousels
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
    pagination: false,
  };

  // Initialize all div with carousel class (for existing carousels)
  var carousels = bulmaCarousel.attach(".carousel", options);

  // New options for the two-row carousel
  var twoRowOptions = {
    // You might need different options here
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 20000,
    pagination: false,
  };

  // Initialize the new two-row carousel
  var twoRowCarousel = bulmaCarousel.attach("#two-row-carousel", twoRowOptions);

  // Assuming you want similar event listeners for the new carousel
  twoRowCarousel.forEach((carousel) => {
    carousel.on("before:show", (state) => {
      console.log(state);
    });
  });

  // Access to bulmaCarousel instance of an element
  var element = document.querySelector("#my-element");
  if (element && element.bulmaCarousel) {
    // bulmaCarousel instance is available as element.bulmaCarousel
    element.bulmaCarousel.on("before-show", function (state) {
      console.log(state);
    });
  }

  /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 50 * this.value;
      })
    }, false);*/
  preloadInterpolationImages();

  $("#interpolation-slider").on("input", function (event) {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $("#interpolation-slider").prop("max", NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();
});
