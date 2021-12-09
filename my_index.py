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
        x = grad_output[:, :, 0]
        device = x.device
        x = torch.cat([x, torch.zeros((grad_output.shape[0], 1), device=device)], dim=-1)
        return torch.sum(x[:, ctx.inverted_indices], dim=-1), None, None


class MyIndex_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, indices, inverted_indices_dim_1, inverted_indices_dim_2):
        # tensor: 2-d tensor (bs * seq_len) x (student_vocab_size + 1)
        # indices: indicies to perform tensor[:, indices]
        # inverted_indices_dim_1: inverted indices (of indices tensor) to perform reverse transformation, for dim 1
        # inverted_indices_dim_2: for dim 2
        # inverted_indices are used to perform optimized bacward pass
        ctx.tensor_shape = tensor.shape
        ctx.inverted_indices = (inverted_indices_dim_1, inverted_indices_dim_2)
        ctx.device = tensor.device
        return tensor[:, indices]

    @staticmethod
    def backward(ctx, grad_output):
        # 3-d grad_output
        # (bs * seq_len) x teacher_vocab_size x X
        bs, voc_size, x_dim = grad_output.shape
        g = torch.cat([grad_output, torch.zeros((bs, 1, x_dim), device=ctx.device)], dim=1)
        return torch.sum(g[:, ctx.inverted_indices[0], ctx.inverted_indices[1]], dim=-1), None, None, None
