import onnxruntime as ort

providers = ort.get_available_providers()

if 'ROCMExecutionProvider' in providers and 'MIGraphXExecutionProvider' in providers:
    print("Onnx runtime has been installed successfully with both ROCm and MIGraphX providers.")
else:
    missing_providers = []
    if 'ROCMExecutionProvider' not in providers:
        missing_providers.append('ROCMExecutionProvider')
    if 'MIGraphXExecutionProvider' not in providers:
        missing_providers.append('MIGraphXExecutionProvider')

    if missing_providers:
        print(f"ERROR: Onnx runtime is not installed with the following provider(s): {', '.join(missing_providers)}")
    else:
        print("ERROR: Onnx runtime is not installed with either ROCm or MIGraphX provider.")
