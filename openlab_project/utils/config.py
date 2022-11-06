from pathlib import Path


ROOT_AIHUB = Path('./차량 및 사람 인지 영상') # AIHUB 데이터셋 경로
ROOT_CITYSCAPES = Path('./cityscapes') # cityscapes 데이터셋 경로
ROOT_ARTIFACTS = Path('./artifacts') # figre 저장 경로

ROOT_TRAIN = ROOT_AIHUB / 'Training' / '바운딩박스'
ROOT_VALID = ROOT_AIHUB / 'Validation' / '바운딩박스'
ROOT_TEST_IMGS = ROOT_CITYSCAPES / 'leftImg8bit' / 'val'
ROOT_TEST_LABELS = ROOT_CITYSCAPES / 'gtFine' / 'val'

ROOT_RESIZE_TRAIN_IMAGES = ROOT_AIHUB / 'images' / 'Training'
ROOT_RESIZE_TRAIN_LABELS = ROOT_AIHUB / 'labels' / 'Training'
ROOT_RESIZE_VALID_IMAGES = ROOT_AIHUB / 'images' / 'Validation'
ROOT_RESIZE_VALID_LABELS = ROOT_AIHUB / 'labels' / 'Validation'
ROOT_RESIZE_TEST_IMAGES = ROOT_CITYSCAPES / 'images'
ROOT_RESIZE_TEST_LABELS = ROOT_CITYSCAPES / 'labels'

RESIZE_TARGET = 512
TRAIN_ASPECT_RATIO = 1.7777777777777777
SEED = 0
