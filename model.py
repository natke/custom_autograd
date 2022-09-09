from tracemalloc import get_traced_memory
from unittest import result
import torch

class MyLogExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        h = input.exp()
        h = h.log()
        ctx.save_for_backward(h)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * result

class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.caf = MyLogExp()
  
    def forward(self, inp):
        return self.caf.apply(inp)

model = MyModel()
model.eval()

input = torch.tensor([1.0, 2.0, 3.0])
print(input)

output = model(input)

print(output)

torch.onnx.export(model, (input,), "model-onnx.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
torch.onnx.export(model, (input,), "model-onnx-fallthrough.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
torch.onnx.export(model, (input,), "model-onnx-aten-fallback.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)