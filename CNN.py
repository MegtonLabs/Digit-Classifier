#Digit classification using neural network
import tkinter as tk
from tkinter import Canvas, Button, Frame, Label
from PIL import Image, ImageDraw, ImageOps # No ImageTk needed currently
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
# import math # Not explicitly used currently
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import time # Keep for potential future use (e.g., performance timing)
import traceback # For detailed error printing

# ==============================================================================
#                           CONFIGURATION PARAMETERS
# ==============================================================================

# --- File Paths ---
MODEL_WEIGHTS_FILE = './checkpoint/mnist_cnn.weights.h5' # Where model weights are saved/loaded

# --- Model & Training ---
NUM_CLASSES = 10
IMG_SIZE = 28      # Standard MNIST image size (pixels)
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1) # Model input: height, width, channels
EPOCHS = 400         # Number of training epochs (if training is needed)
BATCH_SIZE = 128    # Batch size for training

# --- GUI Layout Sizes ---
CANVAS_SIZE = 280              # Pixel size of the drawing canvas
INPUT_GRID_VIS_SIZE = 140      # Pixel size of the input visualization grid
NETWORK_VIS_WIDTH = 500        # Initial width for the network visualization area
NETWORK_VIS_HEIGHT = CANVAS_SIZE + 120 # Initial height for network area
INITIAL_WINDOW_WIDTH_PADDING = 80  # Extra window width padding
INITIAL_WINDOW_HEIGHT_PADDING = 50 # Extra window height padding

# --- Drawing ---
LINE_WIDTH = int(CANVAS_SIZE / 22) # Thickness of the drawing line

# --- Real-time Prediction ---
PREDICTION_DELAY_MS = 450       # Delay (ms) after drawing stops before predicting
PROCESSING_CUE_DURATION_MS = 250 # How long (ms) network lines flash bright

# --- Colors ---
BG_COLOR = 'black'
NETWORK_BG_COLOR = BG_COLOR        # Background for canvases (Added definition)
DRAWING_COLOR = 'white'
TEXT_COLOR = '#00FF00'             # Color for labels
# Network Nodes
NETWORK_NODE_COLOR = '#00FF00'     # Fill color for "active" hidden nodes
NETWORK_NODE_OUTLINE = '#005000'   # Border color for nodes
NETWORK_ACTIVATED_COLOR_MAP = plt.get_cmap('Greens') # Colormap for output node activation
# Network Lines
NETWORK_LINE_COLOR = '#004400'        # Default connection line color
NETWORK_LINE_ACTIVE_COLOR = '#00AA00' # Brighter color during "processing" cue
# Input Grid Visualization
GRID_PIXEL_COLOR_ACTIVE_MAX = '#55FF55' # Color for max intensity pixel in grid vis
GRID_PIXEL_COLOR_INACTIVE = '#050505'   # Color for zero intensity pixel in grid vis
GRID_BORDER_COLOR = '#003300'           # Border color for grid vis box
DRAWING_GRID_COLOR = '#002000'          # Color for faint grid lines on drawing canvas

# --- Network Visualization Tuning ---
# Note: These sizes are for VISUALIZATION ONLY and don't have to match the actual CNN hidden layers exactly.
# It aims to represent the flow: Input -> Conv/Pool -> Conv/Pool -> Flatten -> Dense -> Output
# We simulate this with: InputGrid -> VisLayer1 -> VisLayer2 -> OutputLayer
VIS_LAYER_SIZES = [16, 16, NUM_CLASSES] # Simulated structure: H1(16) -> H2(16) -> Output(10)
NODE_BASE_RADIUS = 8        # Base target radius for nodes
NODE_MAX_RADIUS = 10        # Maximum node radius
NODE_MIN_RADIUS = 5         # Minimum node radius
NODE_V_SPACE_MULT = 3.5     # Multiplier for vertical spacing based on radius
NODE_BORDER_WIDTH = 1.5     # Width of the node border
LABEL_X_OFFSET = 6          # Horizontal offset for output labels from node edge
LABEL_FONT_SIZE_MULT = 1.5  # Multiplier for label font size based on radius
LABEL_MIN_FONT_SIZE = 10    # Minimum font size for output labels
NETWORK_PADDING_X = 50      # Horizontal padding inside network canvas
NETWORK_PADDING_Y = 40      # Vertical padding inside network canvas

# --- Image Preprocessing ---
PREPROCESS_TARGET_SIZE = 20 # Size to fit the drawn digit into (within 28x28)

# ==============================================================================
#                           CNN MODEL DEFINITION & TRAINING
# ==============================================================================

def create_cnn_model():
    """Defines the Keras CNN model architecture."""
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", name="dense_1"), # Actual hidden dense layer
        layers.Dense(NUM_CLASSES, activation="softmax", name="output"), # Output layer
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_or_load_model():
    """Loads pre-trained model weights or trains the model if weights file not found."""
    model = create_cnn_model()
    if os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"Loading weights from {MODEL_WEIGHTS_FILE}")
        try:
            # Use expect_partial() to handle potential mismatches if the model
            # architecture changed slightly but core layers are compatible.
            # For exact loading, just use model.load_weights(MODEL_WEIGHTS_FILE)
            model.load_weights(MODEL_WEIGHTS_FILE).expect_partial()
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Proceeding without loaded weights (will train if data available).")
            # If loading fails critically, you might want to return None or exit
            # return None
    else:
        print(f"Weights file '{MODEL_WEIGHTS_FILE}' not found.")
        print("Attempting to train model (requires MNIST dataset download)...")
        try:
            # Load MNIST data
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            # Preprocess MNIST data
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            # Convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
            y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

            # Train the model
            print(f"Starting training for {EPOCHS} epochs...")
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

            # Evaluate the trained model
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Training Complete.")
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

            # Save weights
            print(f"Saving weights to {MODEL_WEIGHTS_FILE}")
            model.save_weights(MODEL_WEIGHTS_FILE)

        except ImportError:
             print("\nError: `matplotlib` or `tensorflow` might not be installed.")
             print("Please install required packages: pip install tensorflow matplotlib Pillow numpy")
             return None
        except Exception as e:
            print(f"\nError during model training: {e}")
            print("Possible issues:")
            print("- Ensure you have an active internet connection to download the MNIST dataset.")
            print("- Check available disk space.")
            print("- Verify TensorFlow installation and compatibility.")
            traceback.print_exc()
            return None # Indicate failure

    return model

# ==============================================================================
#                              GUI APPLICATION CLASS
# ==============================================================================

class DigitRecognizerApp:
    """Main application class for the digit recognizer GUI."""

    def __init__(self, root, model):
        """Initializes the application UI and state."""
        if model is None:
             raise ValueError("Model cannot be None. Training or loading might have failed.")

        self.root = root
        self.model = model
        self.root.title("Digit Recognizer with Visualization")
        self.root.configure(bg=BG_COLOR)

        # Calculate initial window size based on components
        initial_width = CANVAS_SIZE + INPUT_GRID_VIS_SIZE + NETWORK_VIS_WIDTH + INITIAL_WINDOW_WIDTH_PADDING
        initial_height = NETWORK_VIS_HEIGHT + INITIAL_WINDOW_HEIGHT_PADDING
        self.root.geometry(f"{initial_width}x{initial_height}")
        # Set minimum size to prevent excessive shrinking
        self.root.minsize(width=400, height=400)


        # Drawing state
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black") # PIL Image for processing
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        # Real-time prediction control
        self.predict_job = None # Stores the 'after' job ID for debouncing

        # Network visualization state
        self.num_vis_layers = len(VIS_LAYER_SIZES)
        self.node_positions = {} # Stores {(layer_idx, node_idx): (x, y, radius)}
        self.node_objects = {}   # Stores {(layer_idx, node_idx): canvas_id} for nodes
        self.line_objects = []   # Stores line canvas IDs for color changes
        self.layout_calculated = False # Flag: True when network layout is ready
        self.initial_draw_done = False # Flag: True after first successful network draw

        # --- Setup UI Panels ---
        self._setup_ui_layout()
        self._setup_drawing_panel()
        self._setup_grid_panel()
        self._setup_network_panel()

        # Trigger initial layout calculation and drawing after mainloop starts
        # This ensures widgets have their initial sizes computed.
        self.root.after(150, self.force_initial_draw)

    # --------------------------------------------------------------------------
    # UI Setup Methods
    # --------------------------------------------------------------------------
    def _setup_ui_layout(self):
        """Creates the main frames for the UI layout using packing."""
        # Use packing for simplicity in this three-column layout
        self.drawing_frame = Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        self.drawing_frame.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(10, 5))

        self.grid_frame = Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        self.grid_frame.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=5)

        self.network_frame = Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        self.network_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10))

    def _setup_drawing_panel(self):
        """Sets up the widgets in the drawing panel."""
        lbl = Label(self.drawing_frame, text="Draw Digit Here", fg=TEXT_COLOR, bg=BG_COLOR, anchor='center', font=('Helvetica', 11))
        lbl.pack(pady=(0, 5), fill=tk.X)

        # Canvas for drawing
        self.canvas = Canvas(self.drawing_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                             bg=NETWORK_BG_COLOR, highlightthickness=1, highlightbackground=NETWORK_NODE_OUTLINE)
        self.canvas.pack()
        self._draw_drawing_grid() # Add faint grid lines to drawing canvas

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self._start_paint) # Register start position
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_drawing_stop)

        # Frame for buttons below canvas
        btn_frame = Frame(self.drawing_frame, bg=BG_COLOR)
        btn_frame.pack(pady=15, fill=tk.X, anchor='center')
        btn_frame.columnconfigure(0, weight=1) # Allow button to center if needed

        clear_btn = Button(btn_frame, text="Clear", command=self.clear_all,
                           fg=BG_COLOR, bg=NETWORK_NODE_COLOR, width=10, font=('Helvetica', 12, 'bold'), relief=tk.RAISED, borderwidth=2)
        clear_btn.grid(row=0, column=0, padx=5) # Use grid within frame for centering

    def _setup_grid_panel(self):
        """Sets up the widgets for the input grid visualization."""
        lbl = Label(self.grid_frame, text=f"Input ({IMG_SIZE}x{IMG_SIZE})", fg=TEXT_COLOR, bg=BG_COLOR, anchor='center', font=('Helvetica', 11))
        lbl.pack(pady=(0, 5), fill=tk.X)

        # Canvas for the 28x28 grid visualization
        self.grid_canvas = Canvas(self.grid_frame, width=INPUT_GRID_VIS_SIZE, height=INPUT_GRID_VIS_SIZE,
                                  bg=NETWORK_BG_COLOR, highlightthickness=1, highlightbackground=GRID_BORDER_COLOR)
        # Adjust vertical padding to roughly align centers with drawing canvas (approximate)
        pady_grid = max(0, (CANVAS_SIZE - INPUT_GRID_VIS_SIZE) // 2 - 20) # Adjust '- 20' as needed
        self.grid_canvas.pack(pady=(pady_grid, 0)) # Add padding only on top

        # Calculate size of each pixel representation in the grid viz
        self.pixel_size = INPUT_GRID_VIS_SIZE / IMG_SIZE
        # Store rectangle objects for efficient color updates
        self.grid_pixel_rects = [[None for _ in range(IMG_SIZE)] for _ in range(IMG_SIZE)]
        # Draw the initial empty grid
        self._initialize_input_grid_vis()

    def _setup_network_panel(self):
        """Sets up the widgets for the network visualization."""
        lbl = Label(self.network_frame, text="Network Activity (Simplified)", fg=TEXT_COLOR, bg=BG_COLOR, anchor='center', font=('Helvetica', 11))
        lbl.pack(pady=(0, 5), fill=tk.X)

        # Canvas for network visualization
        self.network_canvas = Canvas(self.network_frame, bg=NETWORK_BG_COLOR, highlightthickness=0)
        # Bind the <Configure> event to handle window resizing
        self.network_canvas.bind('<Configure>', self._on_window_resize)
        self.network_canvas.pack(fill=tk.BOTH, expand=True)

        # Label to display the prediction result
        self.prediction_label = Label(self.network_frame, text="Prediction: -", font=("Consolas", 18, "bold"),
                                      fg=TEXT_COLOR, bg=BG_COLOR)
        self.prediction_label.pack(pady=15)

    # --------------------------------------------------------------------------
    # Drawing Canvas Methods
    # --------------------------------------------------------------------------
    def _draw_drawing_grid(self):
        """Draws faint grid lines on the main drawing canvas for guidance."""
        grid_spacing = CANVAS_SIZE / IMG_SIZE
        for i in range(1, IMG_SIZE):
            # Vertical lines
            x = i * grid_spacing
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, fill=DRAWING_GRID_COLOR, width=0.5, tags='grid')
            # Horizontal lines
            y = i * grid_spacing
            self.canvas.create_line(0, y, CANVAS_SIZE, y, fill=DRAWING_GRID_COLOR, width=0.5, tags='grid')
        # Ensure grid lines are drawn below user strokes
        self.canvas.tag_lower('grid')

    def _start_paint(self, event):
        """Records the starting position of a drawing stroke."""
        self.last_x, self.last_y = event.x, event.y
        # Cancel pending prediction if user starts drawing again quickly
        if self.predict_job:
            self.root.after_cancel(self.predict_job)
            self.predict_job = None


    def _paint(self, event):
        """Handles mouse movement while the button is pressed (drawing)."""
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Draw on Tkinter canvas for immediate visual feedback
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=LINE_WIDTH, fill=DRAWING_COLOR,
                                    capstyle=tk.ROUND, smooth=tk.TRUE, tags='drawing')
            # Draw on the backend PIL image (used for processing)
            # Use integer coordinates for PIL draw
            self.draw.line([int(self.last_x), int(self.last_y), int(x), int(y)],
                           fill=DRAWING_COLOR, width=LINE_WIDTH) # joint="round" is default for line

        # Update last position for the next segment
        self.last_x, self.last_y = x, y

    def _reset_last_pos(self):
        """Resets the last drawing position (e.g., when mouse button is released)."""
        self.last_x, self.last_y = None, None

    def _on_drawing_stop(self, event):
        """Called when the mouse button is released after drawing."""
        self._reset_last_pos()
        # Debounce: Cancel any previously scheduled prediction job
        if self.predict_job:
            self.root.after_cancel(self.predict_job)
            # print("Cancelled previous predict job.") # Debug print
        # Schedule predict_digit to run after a short delay
        # print(f"Scheduling predict job in {PREDICTION_DELAY_MS} ms.") # Debug print
        self.predict_job = self.root.after(PREDICTION_DELAY_MS, self.predict_digit)

    def clear_all(self):
        """Clears drawing canvas, image, grid, prediction, and cancels pending jobs."""
        # Cancel any pending prediction job
        if self.predict_job:
            self.root.after_cancel(self.predict_job)
            self.predict_job = None
            # print("Cancelled predict job on clear.") # Debug print

        # Clear the user's drawing strokes from the canvas (leaves grid)
        self.canvas.delete("drawing")
        # Reset the backend PIL image to black
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)
        # Reset the prediction label
        self.prediction_label.config(text="Prediction: -")
        # Clear the input grid visualization
        self._update_input_grid_vis(None)
        # Reset the network visualization to its default state
        self._draw_network_structure(activations=None, processing=False)
        # Reset line colors explicitly in case a prediction was interrupted
        self._reset_line_colors()

    # --------------------------------------------------------------------------
    # Input Grid Visualization Methods
    # --------------------------------------------------------------------------
    def _initialize_input_grid_vis(self):
        """Draws the initial empty grid visualization (all inactive pixels)."""
        self.grid_canvas.delete("all") # Clear previous grid elements
        for r in range(IMG_SIZE):
            for c in range(IMG_SIZE):
                x0 = c * self.pixel_size
                y0 = r * self.pixel_size
                x1 = x0 + self.pixel_size
                y1 = y0 + self.pixel_size
                # Create rectangle for this pixel and store its ID
                rect_id = self.grid_canvas.create_rectangle(x0, y0, x1, y1,
                fill=GRID_PIXEL_COLOR_INACTIVE,
                outline=GRID_BORDER_COLOR, width=0.5)
                self.grid_pixel_rects[r][c] = rect_id

    def _update_input_grid_vis(self, image_array_28x28):
        """Updates the grid visualization colors based on the processed 28x28 image array."""
        # If no image data (e.g., after clearing), reset the grid
        if image_array_28x28 is None:
            # Set all pixels to inactive color
            for r in range(IMG_SIZE):
                for c in range(IMG_SIZE):
                    rect_id = self.grid_pixel_rects[r][c]
                    # Safely update item color, checking if canvas and item exist
                    if self.grid_canvas.winfo_exists() and rect_id:
                        try:
                            # Check if the item ID is still valid on the canvas
                            if rect_id in self.grid_canvas.find_all():
                                self.grid_canvas.itemconfig(rect_id, fill=GRID_PIXEL_COLOR_INACTIVE)
                        except tk.TclError:
                            # Ignore error if item was deleted (e.g., during window close/resize)
                            # print(f"TclError updating grid pixel ({r},{c}) - item likely deleted.") # Debug
                            pass
            return

        # Extract RGB components for interpolation (using green channel for intensity)
        try:
            min_r = int(GRID_PIXEL_COLOR_INACTIVE[1:3], 16)
            min_g = int(GRID_PIXEL_COLOR_INACTIVE[3:5], 16)
            min_b = int(GRID_PIXEL_COLOR_INACTIVE[5:7], 16)
            max_r = int(GRID_PIXEL_COLOR_ACTIVE_MAX[1:3], 16)
            max_g = int(GRID_PIXEL_COLOR_ACTIVE_MAX[3:5], 16)
            max_b = int(GRID_PIXEL_COLOR_ACTIVE_MAX[5:7], 16)
        except ValueError:
            print("Error parsing grid colors. Using defaults.")
            min_r, min_g, min_b = 5, 5, 5
            max_r, max_g, max_b = 85, 255, 85 # Corresponds to #55FF55

        # Update each pixel's color based on its intensity in the array
        for r in range(IMG_SIZE):
            for c in range(IMG_SIZE):
                intensity = image_array_28x28[r, c] # Intensity value from 0.0 to 1.0

                # Interpolate RGB values
                current_r = min_r + int(intensity * (max_r - min_r))
                current_g = min_g + int(intensity * (max_g - min_g))
                current_b = min_b + int(intensity * (max_b - min_b))

                # Clamp values just in case (shouldn't be needed if intensity is 0-1)
                current_r = max(0, min(255, current_r))
                current_g = max(0, min(255, current_g))
                current_b = max(0, min(255, current_b))

                # Format as hex color string #RRGGBB
                color = f'#{current_r:02x}{current_g:02x}{current_b:02x}'

                rect_id = self.grid_pixel_rects[r][c]
                # Safely update item color, checking if canvas and item exist
                if self.grid_canvas.winfo_exists() and rect_id:
                    try:
                         if rect_id in self.grid_canvas.find_all():
                            self.grid_canvas.itemconfig(rect_id, fill=color)
                    except tk.TclError:
                        # print(f"TclError updating grid pixel ({r},{c}) - item likely deleted.") # Debug
                        pass # Ignore if item deleted unexpectedly

    # --------------------------------------------------------------------------
    # Network Visualization Methods
    # --------------------------------------------------------------------------
    def force_initial_draw(self):
         """Ensures the initial network draw occurs after the window is properly set up."""
         if not self.initial_draw_done and self.network_canvas.winfo_exists():
             # print("Forcing initial network draw...") # Debug
             # Trigger the resize handler manually to calculate layout and draw
             self._on_window_resize()
         elif not self.network_canvas.winfo_exists():
             # print("Initial draw skipped: network canvas not ready yet.") # Debug
             # Retry shortly if canvas wasn't ready
             self.root.after(200, self.force_initial_draw)


    def _on_window_resize(self, event=None):
        """Handles window resize events to recalculate layout and redraw the network."""
        # Check if the network canvas exists and has valid dimensions
        # (winfo_width/height can be 1 initially before the window is fully drawn)
        if self.network_canvas.winfo_exists() and \
           self.network_canvas.winfo_width() > 20 and \
           self.network_canvas.winfo_height() > 20:

            current_width = self.network_canvas.winfo_width()
            current_height = self.network_canvas.winfo_height()
            # print(f"Window resized: {current_width}x{current_height}") # Debug

            # Recalculate node positions based on the new canvas size
            layout_ok = self._calculate_network_layout(current_width, current_height)

            if layout_ok:
                self.layout_calculated = True
                # Redraw the network structure (usually without activations initially or after resize)
                # We pass current prediction state if available, else None
                current_prediction = None # Or retrieve last prediction if needed
                self._draw_network_structure(activations=current_prediction) # Pass last known state if desired
                if not self.initial_draw_done:
                    # print("Initial network draw successful.") # Debug
                    self.initial_draw_done = True
            else:
                # Layout calculation failed (likely due to zero/small dimensions)
                # print("Layout calculation skipped/failed (dimensions too small?)") # Debug
                self.layout_calculated = False
        elif not self.initial_draw_done:
             # If the initial draw hasn't happened and dimensions are still small,
             # schedule another attempt shortly.
             # print("Resize handler called too early, rescheduling.") # Debug
             self.root.after(200, self._on_window_resize)

    def _calculate_network_layout(self, current_width, current_height):
        """Calculates node positions based on current canvas size. Returns True if successful."""
        # Define the drawable area within the canvas, considering padding
        vis_w = current_width - 2 * NETWORK_PADDING_X
        vis_h = current_height - 2 * NETWORK_PADDING_Y

        # Ensure the drawable area is reasonably large
        if vis_w <= 10 or vis_h <= 10:
             # print(f"Network layout calculation failed: Invalid drawable area ({vis_w}x{vis_h})") # Debug
             return False # Indicate failure

        self.node_positions = {} # Reset dictionary to store new positions

        # Total number of "layers" including the conceptual input grid representation
        num_positional_layers = self.num_vis_layers + 1 # Input Grid + VisLayer1 + VisLayer2 + Output

        # Calculate horizontal (x) positions for the center of each layer/stage
        # Linspace includes start and end points.
        layer_xs = np.linspace(NETWORK_PADDING_X, current_width - NETWORK_PADDING_X, num_positional_layers)

        # Store the conceptual center of the input stage (for drawing lines from)
        self.input_grid_center_x = layer_xs[0]
        self.input_grid_center_y = NETWORK_PADDING_Y + vis_h / 2 # Vertically centered

        # Calculate positions for nodes in each *visualized* layer (H1, H2, Output)
        for i_vis, size in enumerate(VIS_LAYER_SIZES):
            layer_idx_pos = i_vis + 1 # Index in layer_xs (1 for H1, 2 for H2, 3 for Output)

            if size is None or size <= 0:
                # print(f"Skipping layout for visual layer {i_vis} (size {size})") # Debug
                continue # Skip layers with zero or invalid size

            layer_x = layer_xs[layer_idx_pos]

            # Estimate vertical space per node in this layer
            node_vertical_space = vis_h / max(size, 1) # Avoid division by zero if size is 0

            # Calculate node radius dynamically based on available space, clamped within min/max
            node_r = max(NODE_MIN_RADIUS, min(NODE_MAX_RADIUS, node_vertical_space * 0.25, NODE_BASE_RADIUS)) # Adjusted multiplier

            # Calculate vertical (y) positions for nodes in this layer, centered vertically
            # Total height spanned by nodes and their spacing
            total_layer_node_span = (size - 1) * (node_r * NODE_V_SPACE_MULT) if size > 1 else 0
            # Starting y position to center the nodes within the drawable height
            y_start = NETWORK_PADDING_Y + max(0, (vis_h - total_layer_node_span) / 2)

            # Calculate individual node y positions
            if size > 1:
                y_positions = [y_start + j * (node_r * NODE_V_SPACE_MULT) for j in range(size)]
            else: # Center single node vertically
                y_positions = [NETWORK_PADDING_Y + vis_h / 2]

            # Store position data (x, y, radius) for each node in the layer
            for j in range(size):
                # Key is (visual_layer_index, node_index_in_layer)
                self.node_positions[(i_vis, j)] = (layer_x, y_positions[j], node_r)

        # print(f"Network layout calculated. Nodes: {len(self.node_positions)}") # Debug
        return True # Indicate layout calculation succeeded

    def _draw_network_structure(self, activations=None, processing=False):
        """Draws the network nodes and connections based on calculated layout."""
        # Ensure layout is calculated and the canvas exists before attempting to draw
        if not self.layout_calculated or not self.node_positions or not self.network_canvas.winfo_exists():
             # print("Skipping network draw: Layout not ready or canvas invalid.") # Debug
             return

        # Clear previous drawings from the network canvas
        self.network_canvas.delete("all")
        # Reset storage for node and line canvas object IDs
        self.node_objects = {}
        self.line_objects = []

        # Determine the index of the output layer in our visualization scheme
        output_layer_idx_vis = self.num_vis_layers - 1
        # Choose line color based on whether we're showing the "processing" cue
        current_line_color = NETWORK_LINE_ACTIVE_COLOR if processing else NETWORK_LINE_COLOR

        # --- Draw Connections (Lines) ---
        # 1. Connect Conceptual Input Grid Center -> First Visualized Hidden Layer (VisLayer1)
        idx_h1 = 0 # Index of the first visualized layer (e.g., representing first dense/conv block)
        if idx_h1 < self.num_vis_layers and VIS_LAYER_SIZES[idx_h1] > 0:
            for k in range(VIS_LAYER_SIZES[idx_h1]):
                 # Check if position data exists for this target node
                 if (idx_h1, k) in self.node_positions:
                    pos_h1 = self.node_positions[(idx_h1, k)] # (x, y, radius)
                    if pos_h1: # Ensure position tuple is valid
                        line_id = self.network_canvas.create_line(
                            self.input_grid_center_x, self.input_grid_center_y, # Start at input center
                            pos_h1[0], pos_h1[1], # End at node center
                            fill=current_line_color, width=0.5)
                        self.line_objects.append(line_id) # Store ID for potential color change later

        # 2. Connect Between Visualized Layers (e.g., VisLayer1 -> VisLayer2, VisLayer2 -> Output)
        for i_vis in range(self.num_vis_layers - 1): # Iterate up to the second-to-last visualized layer
             idx_src = i_vis     # Source layer index (visual)
             idx_tgt = i_vis + 1 # Target layer index (visual)

             # Check if both source and target layers are valid (have nodes)
             if not (VIS_LAYER_SIZES[idx_src] > 0 and VIS_LAYER_SIZES[idx_tgt] > 0):
                 continue # Skip connecting if a layer is empty

             # Draw lines from every node in src layer to every node in tgt layer (dense look)
             for j in range(VIS_LAYER_SIZES[idx_src]):
                 # Check if source node position exists
                 if (idx_src, j) in self.node_positions:
                     pos_src = self.node_positions[(idx_src, j)]
                     if pos_src: # Check source position tuple valid
                         for k in range(VIS_LAYER_SIZES[idx_tgt]):
                             # Check if target node position exists
                             if (idx_tgt, k) in self.node_positions:
                                 pos_tgt = self.node_positions[(idx_tgt, k)]
                                 if pos_tgt: # Check target position tuple valid
                                     line_id = self.network_canvas.create_line(
                                         pos_src[0], pos_src[1], # Start at source node center
                                         pos_tgt[0], pos_tgt[1], # End at target node center
                                         fill=current_line_color, width=0.5)
                                     self.line_objects.append(line_id)

        # --- Draw Nodes (Circles) ---
        max_output_activation = 0.001 # Small value to prevent division by zero
        output_activations_array = None

        # Check if valid activations were passed (expecting list/tuple with output probabilities)
        if activations is not None and len(activations) > 0 and isinstance(activations[0], np.ndarray):
             output_activations_array = activations[0] # Assuming first element is the output probability array
             if len(output_activations_array) > 0:
                 # Find the maximum activation in the output layer for normalization/scaling
                 max_output_activation = max(np.max(output_activations_array), 0.001)

        # Iterate through the visualized layers (H1, H2, Output) to draw nodes
        for i_vis, size in enumerate(VIS_LAYER_SIZES):
            if size is None or size <= 0: continue # Skip empty layers
            for j in range(size):
                # Check if position data exists for this node
                if (i_vis, j) in self.node_positions:
                    pos_data = self.node_positions[(i_vis, j)]
                    if pos_data: # Check position tuple is valid
                        x, y, r = pos_data # Unpack position and radius
                        is_output_layer = (i_vis == output_layer_idx_vis)

                        # Determine node appearance
                        node_fill_color = NETWORK_NODE_COLOR # Default fill for hidden nodes
                        node_outline_color = NETWORK_NODE_OUTLINE
                        node_outline_width = NODE_BORDER_WIDTH

                        # Special handling for Output Layer nodes: Color based on activation
                        if is_output_layer and output_activations_array is not None:
                           if j < len(output_activations_array): # Ensure index is within bounds
                              activation = output_activations_array[j]
                              # Scale activation 0-1 relative to the max activation in this layer
                              scaled_act = max(0.0, min(1.0, activation / max_output_activation))

                              try:
                                  # Map the scaled activation (slightly offset for visibility) to the colormap
                                  rgba = NETWORK_ACTIVATED_COLOR_MAP(scaled_act * 0.85 + 0.15) # Apply colormap
                                  node_fill_color = self._rgb_to_hex(rgba[:3]) # Convert RGB part to hex string
                              except Exception as e:
                                  # print(f"Error applying colormap: {e}") # Debug
                                  node_fill_color = NETWORK_NODE_COLOR # Fallback on error
                           else:
                               # print(f"Warning: Output node index {j} out of bounds for activations array.") # Debug
                               node_fill_color = NETWORK_NODE_COLOR # Fallback if index is wrong

                        # Create the node (oval) on the canvas
                        node_id = self.network_canvas.create_oval(
                            x - r, y - r, x + r, y + r, # Bounding box for the oval
                            fill=node_fill_color,
                            outline=node_outline_color,
                            width=node_outline_width)
                        # Store the canvas ID of the node object
                        self.node_objects[(i_vis, j)] = node_id

                        # Add text labels (digit '0' through '9') next to the output layer nodes
                        if is_output_layer:
                           label_x = x + r + node_outline_width + LABEL_X_OFFSET # Position label to the right
                           # Calculate font size dynamically based on node radius, with a minimum size
                           font_size = max(int(r * LABEL_FONT_SIZE_MULT), LABEL_MIN_FONT_SIZE)
                           self.network_canvas.create_text(label_x, y, text=str(j), fill=TEXT_COLOR,
                                                           anchor=tk.W, # Anchor text to the West (left)
                                                           font=("Consolas", font_size, 'bold'))

    def _update_network_visualization(self, output_probs, processing=False):
        """Convenience helper to redraw the network structure with current activation state."""
        # Pass the output probabilities (or None) and processing flag to the main draw function
        self._draw_network_structure(activations=[output_probs] if output_probs is not None else None,
                                     processing=processing)

    def _reset_line_colors(self):
        """Resets the network connection lines to their default color after the 'processing' cue."""
        # Check if the canvas still exists (might be destroyed during app closure)
        if not self.network_canvas.winfo_exists():
            # print("Skipping line color reset: canvas destroyed.") # Debug
            return

        # print(f"Resetting {len(self.line_objects)} line colors.") # Debug
        for line_id in self.line_objects:
            try:
                # Check if the line object ID is still valid on the canvas before configuring
                if line_id in self.network_canvas.find_all():
                     self.network_canvas.itemconfig(line_id, fill=NETWORK_LINE_COLOR)
            except tk.TclError:
                 # Ignore error if the item was deleted between getting the list and configuring it
                 # This can happen during rapid events or window closure.
                 # print(f"TclError resetting line color - item {line_id} likely deleted.") # Debug
                 pass
        # It's generally safer to clear the list *after* attempting modifications,
        # but since we redraw lines completely each time, clearing here is okay.
        self.line_objects = [] # Clear the list as lines will be redrawn anyway


    # --------------------------------------------------------------------------
    # Prediction Logic
    # --------------------------------------------------------------------------
    def predict_digit(self):
        """Processes the drawn image, runs prediction, and updates the UI."""
        # print("Executing predict_digit...") # Debug
        self.predict_job = None # Mark that the scheduled prediction is now running
        processed_image_array = None # To store the 28x28 array for grid display

        try:
            # Ensure the network layout has been calculated at least once
            if not self.layout_calculated:
                print("Layout not calculated, cannot predict/visualize yet.")
                self.root.after(100, self.predict_digit) # Retry shortly
                return

            # --- Start 'Processing' Visual Cue ---
            # Redraw network with bright lines, but no activations yet
            self._update_network_visualization(None, processing=True)
            # Force Tkinter to update the display immediately to show the cue
            self.root.update_idletasks()

            # --- Preprocess the Drawing from PIL Image ---
            # 1. Find the bounding box of the drawn content (non-black pixels)
            bbox = self.image.getbbox()

            # If bbox is None, the canvas is empty
            if bbox is None:
                print("Canvas is empty. Clearing.")
                self.clear_all() # Clear everything including resetting visuals
                # Ensure lines are reset if cue started before finding empty canvas
                self._reset_line_colors()
                return

            # 2. Crop the image to the bounding box
            img_cropped = self.image.crop(bbox)

            # Calculate padding to make the cropped image square
            width, height = img_cropped.size
            padding = abs(width - height) // 2
            if width > height:
                # Add vertical padding
                img_padded = ImageOps.expand(img_cropped, border=(0, padding), fill='black')
            elif height > width:
                # Add horizontal padding
                img_padded = ImageOps.expand(img_cropped, border=(padding, 0), fill='black')
            else:
                img_padded = img_cropped # Already square

            # 3. Resize the square image to fit within the PREPROCESS_TARGET_SIZE (e.g., 20x20)
            # Using LANCZOS for high-quality downsampling
            img_resized = img_padded.resize((PREPROCESS_TARGET_SIZE, PREPROCESS_TARGET_SIZE), Image.Resampling.LANCZOS)

            # 4. Create a new black 28x28 image
            img_final_28x28 = Image.new("L", (IMG_SIZE, IMG_SIZE), "black")

            # 5. Paste the resized digit onto the center of the 28x28 canvas
            paste_x = (IMG_SIZE - PREPROCESS_TARGET_SIZE) // 2
            paste_y = (IMG_SIZE - PREPROCESS_TARGET_SIZE) // 2
            img_final_28x28.paste(img_resized, (paste_x, paste_y))

            # Optional: Save intermediate images for debugging
            # img_final_28x28.save("debug_processed_image.png")

            # --- Convert to NumPy array for Model and Visualization ---
            # Convert to float32 and normalize pixel values to [0, 1]
            img_array_norm = np.array(img_final_28x28).astype('float32') / 255.0
            processed_image_array = img_array_norm.copy() # Keep a copy for the grid visualization

            # Reshape for the model: (batch_size, height, width, channels)
            img_array_reshaped = img_array_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # --- Update Input Grid Visualization ---
            self._update_input_grid_vis(processed_image_array)

            # --- Run Prediction using the CNN Model ---
            predictions = self.model.predict(img_array_reshaped, verbose=0) # verbose=0 avoids predict logs
            output_probabilities = predictions[0] # Get probabilities for the single input image
            predicted_class = np.argmax(output_probabilities) # Find the class index with highest probability
            confidence = np.max(output_probabilities) # Get the highest probability value

            # --- Update UI with Prediction Results ---
            # Display prediction and confidence on the label
            self.prediction_label.config(text=f"Prediction: {predicted_class} ({confidence*100:.1f}%)")

            # Redraw the network visualization, now showing output activations (lines still bright)
            self._update_network_visualization(output_probabilities, processing=True)

            # --- Highlight the Predicted Output Node ---
            output_layer_idx_vis = self.num_vis_layers - 1
            # Check if the node object for the predicted class exists
            if (output_layer_idx_vis, predicted_class) in self.node_objects:
                 node_id = self.node_objects[(output_layer_idx_vis, predicted_class)]
                 try:
                    # Check if canvas and the specific node item still exist before configuring
                    if self.network_canvas.winfo_exists() and node_id in self.network_canvas.find_all():
                         # Make the border of the predicted node brighter/thicker
                         self.network_canvas.itemconfig(node_id, outline='white', width=NODE_BORDER_WIDTH + 0.5)
                 except tk.TclError as e:
                     # Error might occur if the window was closed or resized rapidly
                     print(f"Tkinter error highlighting predicted node: {e}")

            # --- End 'Processing' Visual Cue ---
            # Schedule the line colors to reset back to normal after a short duration
            self.root.after(PROCESSING_CUE_DURATION_MS, self._reset_line_colors)
            # print("Prediction complete. Scheduled line color reset.") # Debug

        except Exception as e:
            print(f"\n--- Error during prediction or visualization ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc() # Print the full traceback for debugging
            print("--------------------------------------------------")
            self.prediction_label.config(text="Prediction: Error")

            # Attempt to show the processed grid even if prediction fails (useful for debugging preprocessing)
            if processed_image_array is not None:
                self._update_input_grid_vis(processed_image_array)
            else: # If error happened before array creation, clear grid
                 self._update_input_grid_vis(None)

            # Ensure the network visualization is reset to a default state on error
            self._update_network_visualization(None, processing=False)
            # Explicitly reset line colors in case the error occurred after the cue started
            # but before the reset timer was set.
            self._reset_line_colors()

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    def _rgb_to_hex(self, rgb):
        """Converts a tuple of normalized RGB values (0-1) to a Tkinter hex color string."""
        # Scale float values (0.0-1.0) to integers (0-255)
        r, g, b = [int(c * 255.99) for c in rgb] # Use 255.99 to ensure 1.0 maps correctly to 255
        # Format as #RRGGBB hexadecimal string
        return f'#{r:02x}{g:02x}{b:02x}'

# ==============================================================================
#                                 MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Starting Digit Recognizer Application...")
    print("TensorFlow Version:", tf.__version__)

    # Optional: GPU Check/Configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
      print(f"Found GPU: {physical_devices[0].name}")
      # Optional: Configure memory growth to avoid allocating all GPU memory at once
      # This is often needed for complex models or when sharing the GPU.
      try:
          for gpu in physical_devices:
             tf.config.experimental.set_memory_growth(gpu, True)
          print("GPU memory growth enabled.")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(f"Could not set memory growth: {e}")
    else:
      print("No GPU found, using CPU.")

    # Attempt to train (first time) or load the pre-trained model
    print("\nLoading/Training Model...")
    cnn_model = train_or_load_model()

    if cnn_model: # Proceed only if model loaded/trained successfully
        print("\nModel Ready.")
        cnn_model.summary(line_length=100) # Print model architecture details

        # Create the Tkinter root window
        root = tk.Tk()

        # Create and run the application instance
        try:
            app = DigitRecognizerApp(root, cnn_model) # Pass the loaded/trained model
            print("\nStarting GUI...")
            root.mainloop() # Start the Tkinter event loop
            print("\nApplication closed.")
        except Exception as e:
             print(f"\n--- Critical GUI Error ---")
             print(f"An unexpected error occurred while running the application:")
             print(f"Error Type: {type(e).__name__}")
             print(f"Error Details: {e}")
             traceback.print_exc()
             print("-------------------------")

    else:
        print("\nExiting application: Model could not be loaded or trained.")
        # Optional: Add a simple Tkinter message box for failure if GUI libs loaded
        try:
            import tkinter.messagebox
            root = tk.Tk()
            root.withdraw() # Hide the main window
            tkinter.messagebox.showerror("Model Error", "Failed to load or train the CNN model. Please check console output for details. Exiting.")
            root.destroy()
        except ImportError:
            pass # Tkinter might not be fully available if there were earlier issues