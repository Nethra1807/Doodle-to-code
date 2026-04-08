/* Canvas drawing logic */

let canvas, ctx;
let isDrawing = false;
let undoStack = [];
let drawingMode = 'freedraw'; // freedraw, line, rect, circle
let startX = 0;
let startY = 0;

// Initialize canvas
export function initCanvas() {
  canvas = document.getElementById('drawingCanvas');
  if (!canvas) return;

  // Set internal resolution
  canvas.width = 600;
  canvas.height = 450;
  
  ctx = canvas.getContext('2d');
  
  // Fill initial background white
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Save initial state
  saveState();

  // Mouse events
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', endDrawing);
  canvas.addEventListener('mouseout', endDrawing);

  // Touch events (for mobile)
  canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
  canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
  canvas.addEventListener('touchend', endDrawing);
}

export function setDrawingMode(mode) {
  drawingMode = mode;
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function startDrawing(e) {
  isDrawing = true;
  const pos = getPos(e);
  startX = pos.x;
  startY = pos.y;
  
  if (drawingMode === 'freedraw') {
    ctx.beginPath();
    ctx.moveTo(startX, startY);
  }
}

function draw(e) {
  if (!isDrawing) return;
  const pos = getPos(e);
  
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000000';

  if (drawingMode === 'freedraw') {
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  } else {
    // For shapes, we restore the previous state and draw the new shape on top
    restoreLastState();
    ctx.beginPath();
    
    if (drawingMode === 'line') {
      ctx.moveTo(startX, startY);
      ctx.lineTo(pos.x, pos.y);
    } else if (drawingMode === 'rect') {
      ctx.rect(startX, Math.min(startY, pos.y), pos.x - startX, Math.abs(pos.y - startY));
    } else if (drawingMode === 'circle') {
      const radius = Math.sqrt(Math.pow(pos.x - startX, 2) + Math.pow(pos.y - startY, 2));
      ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
    }
    ctx.stroke();
  }
}

function restoreLastState() {
  if (undoStack.length > 0) {
    const previousState = undoStack[undoStack.length - 1];
    ctx.putImageData(previousState, 0, 0);
  }
}

function handleTouchStart(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const mouseEvent = new MouseEvent('mousedown', {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}

function handleTouchMove(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const mouseEvent = new MouseEvent('mousemove', {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}

function endDrawing(e) {
  if (isDrawing) {
    // Finalize shape if we are outside canvas and need pos
    if (e.type === 'mouseout') {
      const pos = getPos(e);
      // It's already drawn in `draw`, we just need to save state.
    }
    isDrawing = false;
    ctx.closePath();
    saveState();
  }
}

export function saveState() {
  if (!canvas) return;
  undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

export function undo() {
  if (undoStack.length > 1) {
    undoStack.pop(); // Remove current state
    const previousState = undoStack[undoStack.length - 1]; // Get previous
    ctx.putImageData(previousState, 0, 0);
  } else if (undoStack.length === 1) {
    clearCanvas();
  }
}

export function clearCanvas() {
  if (!canvas) return;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  undoStack = [];
  saveState();
}

export function captureCanvas() {
  if (!canvas) return null;
  return canvas.toDataURL('image/png');
}

export function uploadImage(fileUrl) {
  if (!canvas) return;
  const img = new Image();
  img.onload = () => {
    // Fill background white first
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Calculate aspect ratio fit
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const x = (canvas.width / 2) - (img.width / 2) * scale;
    const y = (canvas.height / 2) - (img.height / 2) * scale;
    
    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
    saveState();
  };
  img.src = fileUrl;
}
