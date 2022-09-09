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

class MyLogExpModel(torch.nn.Module):

    def __init__(self):
        super(MyLogExpModel, self).__init__()
        self.caf = MyLogExp()
  
    def forward(self, inp):
        return self.caf.apply(inp)

model = MyLogExpModel()
model.eval()

input = torch.tensor([1.0, 2.0, 3.0])
print(input)

output = model(input)

print(output)

torch.onnx.export(model, (input,), "model-onnx.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
torch.onnx.export(model, (input,), "model-onnx-fallthrough.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
#torch.onnx.export(model, (input,), "model-onnx-aten-fallback.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

class MyGridSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        return torch.grid_sampler(input, grid, interpolation_mode=0, padding_mode=0, align_corners=True)


class MyGridSampleModel(torch.nn.Module):
    def __init__(self):
        super(MyGridSampleModel, self).__init__()
        self.caf = MyGridSampler()
  
    def forward(self, inp, grid):
        return self.caf.apply(inp, grid)

input = torch.rand((1,2,224, 244))
grid = torch.rand((1,4,4,2))

model = MyGridSampleModel()
output = model(input, grid)

print(output)

#torch.onnx.export(model, (input,grid), "gs-model-onnx.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
torch.onnx.export(model, (input,grid), "gs-model-onnx-fallthrough.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
#torch.onnx.export(model, (input,grid), "gs-model-onnx-aten-fallback.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

