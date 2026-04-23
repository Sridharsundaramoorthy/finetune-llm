MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

DATASET_PATH = "data/dataset.json"

OUTPUT_DIR = "adapters/sample_output"

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]

# Training config
LEARNING_RATE = 0.00001
NUM_EPOCHS = 4
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LOGGING_STEPS = 10
SAVE_STEPS = 100

FP16 = True
