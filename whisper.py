import time
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Load the model
llm = LLM(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=200,
)

def process_audio_sample(audio_data):
    """Process a single audio sample from Hugging Face dataset"""
    # Extract array and sampling rate from the audio dict
    array = audio_data['array']
    sampling_rate = audio_data['sampling_rate']
    
    # Ensure the array is float32
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    # Ensure audio is mono
    if len(array.shape) > 1 and array.shape[1] > 1:
        array = np.mean(array, axis=1)
    
    # Normalize if needed
    if np.max(np.abs(array)) > 1.0:
        array = array / np.max(np.abs(array))
    
    return array, float(sampling_rate)

def transcribe_single_sample(dataset_name, split, sample_index=0, config=None, audio_field='audio'):
    """
    Transcribe a single audio sample from a Hugging Face dataset
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to use
        sample_index: Index of the sample to transcribe
        config: Optional dataset configuration
        audio_field: Name of the audio field in the dataset
        
    Returns:
        Dictionary with transcription results
    """
    start_time = time.time()
    
    # Load the dataset
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}
    
    # Validate sample index
    if sample_index >= len(dataset):
        return {"error": f"Sample index {sample_index} out of range for dataset with {len(dataset)} samples"}
    
    # Check if audio field exists
    if audio_field not in dataset.features:
        # Try to find an alternative audio field
        for field in ['audio', 'sound', 'speech']:
            if field in dataset.features:
                audio_field = field
                break
        else:
            return {"error": f"Could not find audio field in dataset {dataset_name}"}
    
    # Get the sample
    sample = dataset[sample_index]
    
    # Get and process audio
    audio_data = sample[audio_field]
    audio_array, sample_rate = process_audio_sample(audio_data)
    
    # Create prompt
    prompt = {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": (audio_array, sample_rate),
        },
    }
    
    # Generate transcription
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]
    
    # Find reference text if available
    reference = None
    for field in ['text', 'transcript', 'sentence', 'transcription']:
        if field in sample:
            reference = sample[field]
            break
    
    # Create result
    result = {
        "dataset": dataset_name,
        "split": split,
        "sample_index": sample_index,
        "transcription": output.outputs[0].text,
        "processing_time": time.time() - start_time
    }
    
    if reference:
        result["reference"] = reference
    
    return result

def test_with_different_datasets():
    """Test transcription with different datasets, one sample at a time"""
    # Common Voice
    print("Testing with Common Voice:")
    result = transcribe_single_sample(
        dataset_name="mozilla-foundation/common_voice_11_0",
        config="en",
        split="test[:10]",
        sample_index=0
    )
    print(f"Transcription: {result['transcription']}")
    if 'reference' in result:
        print(f"Reference: {result['reference']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print("-" * 40)
    
    # LibriSpeech
    print("Testing with LibriSpeech:")
    result = transcribe_single_sample(
        dataset_name="librispeech_asr",
        config="clean",
        split="test.other[:10]",
        sample_index=0
    )
    print(f"Transcription: {result['transcription']}")
    if 'reference' in result:
        print(f"Reference: {result['reference']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print("-" * 40)
    
    # FLEURS
    print("Testing with FLEURS:")
    result = transcribe_single_sample(
        dataset_name="google/fleurs",
        config="en_us",
        split="test[:10]",
        sample_index=0
    )
    print(f"Transcription: {result['transcription']}")
    if 'reference' in result:
        print(f"Reference: {result['reference']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    
if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Test with multiple datasets
    test_with_different_datasets()
    
    # Option 2: Test with a specific dataset and sample
    # result = transcribe_single_sample(
    #     dataset_name="mozilla-foundation/common_voice_11_0",
    #     config="en",
    #     split="test[:100]",
    #     sample_index=42  # Choose a specific sample
    # )
    # print(result)
