import torch

class MyIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, indices, inverted_indices):
        # tensor: 2-d tensor (bs * seq_len) x (student_vocab_size + 1)
        ctx.tensor_shape = tensor.shape
        ctx.inverted_indices = inverted_indices
        return tensor[:, indices]
        
    @staticmethod
    def backward(ctx, grad_output):
        # 3-d grad_output, gradients in last dimension are equal (as grad_output is result of torch.sum)
        # (bs * seq_len) x teacher_vocab_size x X
        x = grad_output[:,:, 0]
        device = x.device
        x = torch.cat([x, torch.zeros((grad_output.shape[0], 1), device=device)], dim=-1)
        return torch.sum(x[:, ctx.inverted_indices], dim=-1), None, None
