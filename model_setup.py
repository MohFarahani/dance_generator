class Model_Setup:
    """
    This class hasall the CONSTANTS and Hyperparameter
    of the models.

    Args:
    Self explainable

    Returns:

    Raises:

    """

    def __init__(
        self,
        _presence_threshold=0.5,
        _rgb_channels=3,
        _visibility_threshold=0.5,
        batch_size=256,
        black_color=(0, 0, 0),
        blue_color=(255, 0, 0),
        buffer_size=150,
        epochs=150,
        green_color=(0, 128, 0),
        hist_windows=24 * 3,
        horizon=1,
        model="autoregression",
        num_coords=33,
        red_color=(0, 0, 255),
        show_window=False,
        steps_per_epoch=100,
        train_split=0.2,
        validation_steps=50,
        verbose=1,
        white_color=(224, 224, 224),
    ):
        # Train.py
        self.HIST_WINDOW = hist_windows
        self.HORIZON = horizon
        self.TRAIN_SPLIT = train_split

        self.MODEL = model
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size

        self.EPOCHS = epochs
        self.STEPS_PER_EPOCH = steps_per_epoch
        self.VALIDATION_STEPS = validation_steps
        self.VERBOSE = verbose

        # PoseDataGenerator
        self.SHOW_WINDOW = show_window
        self.NUM_COORDS = num_coords

        # Plot3D
        self.WHITE_COLOR = white_color
        self.BLACK_COLOR = black_color
        self.RED_COLOR = red_color
        self.GREEN_COLOR = green_color
        self.BLUE_COLOR = blue_color

        self._PRESENCE_THRESHOLD = _presence_threshold
        self._VISIBILITY_THRESHOLD = _visibility_threshold
        self._RGB_CHANNELS = _rgb_channels
