import torch
import whisper

model = whisper.load_model('tiny') # tiny — лёгкий, чтобы быстро экспортировать
encoder = model.encoder
encoder.eval()
dummy = torch.zeros(1, 80, 3000)
torch.onnx.export(
    encoder,
    dummy,
    'docs/whisper_encoder.onnx',
    input_names=['mel'],
    output_names=['features'],
    opset_version=14,
)
print('Модель экспортирована в docs/whisper_encoder.onnx')