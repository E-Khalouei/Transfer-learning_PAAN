# This script extracts feature embeddings from input audio data. The input is the audio data I have generated, and the output is a set of embeddings representing those features.

# ------ Inference audio tagging result with pretrained model. ------
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="/PAAN/feature-extraction/audioset_tagging_cnn-master_fine-tuning/scripts/Cnn14_mAP=0.431.pth"      #The pretrained model that we used. You can download: https://zenodo.org/record/3987831
AUDIO_DIR="/PAAN/feature-extraction/audioset_tagging_cnn-master_fine-tuning/scripts/data/GW_audio_32k_train_5000"
OUTPUT_DIR="/PAAN/feature-extraction/audioset_tagging_cnn-master_fine-tuning/scripts/data/embedding"
INFERENCE_SCRIPT=/PAAN/feature-extraction/audioset_tagging_cnn-master_fine-tuning/pytorch/inference.py" 
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Iterate over all WAV files in the audio directory
for AUDIO_PATH in $AUDIO_DIR/*.wav; do
    echo "Processing $AUDIO_PATH"
    python3 $INFERENCE_SCRIPT audio_tagging \
        --model_type=$MODEL_TYPE \
        --checkpoint_path=$CHECKPOINT_PATH \
        --audio_path=$AUDIO_PATH \
        --cuda \
        --output_dir=$OUTPUT_DIR
done
