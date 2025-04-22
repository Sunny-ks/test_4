#!/bin/bash
# Complete pipeline script for preparing data and training the Phi-4 image classifier

# Default settings
INPUT_DIR="./image_dataset"
OUTPUT_DIR="./phi4_datasets"
MODEL_DIR="./phi4_image_classifier"
MODEL_NAME="microsoft/Phi-4-vision"  # Change to your specific model version
BATCH_SIZE=4
EPOCHS=3
LEARNING_RATE=5e-6

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --wandb)
      USE_WANDB="--log_wandb"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input_dir DIR       Input directory containing image folders (default: ./image_dataset)"
      echo "  --output_dir DIR      Output directory for prepared datasets (default: ./phi4_datasets)"
      echo "  --model_dir DIR       Directory to save the trained model (default: ./phi4_image_classifier)"
      echo "  --model_name NAME     Phi-4 model to use (default: microsoft/Phi-4-vision)"
      echo "  --batch_size N        Batch size for training (default: 4)"
      echo "  --epochs N            Number of training epochs (default: 3)"
      echo "  --learning_rate N     Learning rate (default: 5e-6)"
      echo "  --wandb               Enable Weights & Biases logging"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== Phi-4 Image Classification Pipeline ==="
echo "Input directory:    $INPUT_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "Model directory:    $MODEL_DIR"
echo "Model name:         $MODEL_NAME"
echo "Batch size:         $BATCH_SIZE"
echo "Epochs:             $EPOCHS"
echo "Learning rate:      $LEARNING_RATE"
echo ""

# Step 1: Prepare the datasets
echo "Step 1: Preparing datasets..."
python phi4-image-dataset-prep.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --include_multi_label

# Step 2: Train the model
echo "Step 2: Training the model..."
python phi4-image-classifier.py \
  --model_name "$MODEL_NAME" \
  --train_data "$OUTPUT_DIR/train.json" \
  --val_data "$OUTPUT_DIR/val.json" \
  --output_dir "$MODEL_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  $USE_WANDB

# Step 3: Test inference on a few sample images
echo "Step 3: Testing inference..."
python phi4-inference-script.py \
  --model_path "$MODEL_DIR" \
  --input "$OUTPUT_DIR/sample.json" \
  --output "$MODEL_DIR/predictions.json"

echo "Pipeline completed successfully!"
echo "Trained model saved to: $MODEL_DIR"
echo "Predictions saved to: $MODEL_DIR/predictions.json"
