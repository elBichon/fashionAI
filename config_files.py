from mrcnn.config import Config
IMAGE_SIZE = 512

class TestConfig(Config):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 46
	NAME = "fashion"
	BACKBONE = 'resnet50'
	IMAGE_MIN_DIM = IMAGE_SIZE
	IMAGE_MAX_DIM = IMAGE_SIZE    
	IMAGE_RESIZE_MODE = 'none'
	RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

