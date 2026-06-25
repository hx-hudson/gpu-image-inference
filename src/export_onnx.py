import os
import onnx
import torch
import onnxruntime
from model import load_model

def main():

    device = torch.device("cuda")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    save_root = os.path.join(root_dir, "models")
    save_path = os.path.join(save_root, "resnet_cifar10.onnx")

    torch_model = load_model(
        os.path.join(save_root, "resnet_cifar10.pth"),
        device
    )

    dummy_input = (torch.randn(1, 3, 32, 32, device=device),)

    torch.onnx.export(
        torch_model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Check ONNX model validity
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(save_path, providers=providers)

    for batch_size in [1, 8, 16, 32]:
        x = torch.randn(batch_size, 3, 32, 32, device=device)

        with torch.no_grad():
            torch_outputs = torch_model(x)

        ort_inputs = {
            "input": x.detach().cpu().numpy()
        }

        ort_outputs = ort_session.run(["output"], ort_inputs)[0]
        ort_outputs = torch.from_numpy(ort_outputs).to(device)

        assert torch_outputs.shape == ort_outputs.shape

        max_abs_diff = (torch_outputs - ort_outputs).abs().max().item()
        mean_abs_diff = (torch_outputs - ort_outputs).abs().mean().item()

        print(f"Batch size: {batch_size}")
        print(f"PyTorch output shape: {torch_outputs.shape}")
        print(f"ONNX output shape: {ort_outputs.shape}")
        print(f"Max abs diff: {max_abs_diff:.6f}")
        print(f"Mean abs diff: {mean_abs_diff:.6f}")

        torch.testing.assert_close(
            torch_outputs,
            ort_outputs,
            rtol=1e-3,
            atol=2e-3,
        )

    print("PyTorch and ONNX Runtime output matched!")

if __name__ == "__main__":
    main()
