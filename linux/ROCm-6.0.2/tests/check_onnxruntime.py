import onnxruntime as ort

providers = ort.get_available_providers()
if 'MIGraphXExecutionProvider' in providers:
    print("Onnx runtime has been installed successfully with the MIGraphX provider")
else:
    print("ERROR: Onnx runtime is not installed with the MIGraphX provider")


