import unittest

import torch

from model import (
    AntiSymmetricBranchConv1d,
    MLP,
    MixedParityTemporalConv1d,
    MultiScaleAntiSymmetricConv1d,
    MultiScaleSymmetricConv1d,
    ParityProjectedPointwiseHead,
    SymmetricBranchConv1d,
)


def _assert_close(a, b, atol=1e-6, rtol=1e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def _branches(module):
    return getattr(module, "temporal_branches", [module])


class MixedParityTests(unittest.TestCase):
    def test_symmetric_effective_kernels_are_palindromic(self):
        torch.manual_seed(0)
        modules = [
            SymmetricBranchConv1d(2, 3, 5, conv_layers=1),
            SymmetricBranchConv1d(2, 3, 7, conv_layers=2),
            MultiScaleSymmetricConv1d(2, 4, kernels=(3, 5), conv_layers=2),
        ]
        for module in modules:
            for branch in _branches(module):
                for layer in range(1, branch.conv_layers + 1):
                    w = branch.effective_weight(layer)
                    _assert_close(w, w.flip(-1))

    def test_antisymmetric_effective_kernels_are_anti_palindromic_with_zero_center(self):
        torch.manual_seed(1)
        modules = [
            AntiSymmetricBranchConv1d(2, 3, 5, conv_layers=1, bias=False),
            AntiSymmetricBranchConv1d(2, 3, 7, conv_layers=2, bias=False),
            MultiScaleAntiSymmetricConv1d(2, 4, kernels=(3, 5), conv_layers=2, bias=False),
        ]
        for module in modules:
            for branch in _branches(module):
                for layer in range(1, branch.conv_layers + 1):
                    w = branch.effective_weight(layer)
                    _assert_close(w, -w.flip(-1))
                    _assert_close(w[..., w.shape[-1] // 2], torch.zeros_like(w[..., w.shape[-1] // 2]))

    def test_mixed_frontend_time_reversal_parity_and_forced_odd_single_layer(self):
        torch.manual_seed(2)
        temporal_conv = MixedParityTemporalConv1d(
            in_channels=3,
            filters_per_channel=4,
            kernels=(3, 5),
            conv_layers=2,
        ).eval()
        self.assertEqual(temporal_conv.sym_conv.conv_layers, 2)
        self.assertEqual(temporal_conv.anti_conv.conv_layers, 1)
        self.assertTrue(all(branch.conv.bias is None for branch in temporal_conv.anti_conv.temporal_branches))

        x = torch.randn(5, 3, 23)
        e, o = temporal_conv(x)
        e_rev, o_rev = temporal_conv(x.flip(-1))
        _assert_close(e_rev, e.flip(-1))
        _assert_close(o_rev, -o.flip(-1))

    def test_projected_head_direct_parity_with_dropout_eval_and_train(self):
        for training in (False, True):
            torch.manual_seed(3)
            head = ParityProjectedPointwiseHead(
                even_in_dim=4,
                odd_in_dim=6,
                even_out_dim=4,
                odd_out_dim=2,
                hidden_dim=12,
                depth=3,
                dropout=0.5,
            )
            head.train(training)
            e = torch.randn(3, 4, 11)
            o = torch.randn(3, 6, 11)

            if training:
                (even, odd), (even_flip, odd_flip) = head.forward_parts_with_parity_flip(e, o)
            else:
                even, odd = head.forward_parts(e, o)
                even_flip, odd_flip = head.forward_parts(e, -o)

            _assert_close(even_flip, even)
            _assert_close(odd_flip, -odd)

    def test_complete_mixed_model_time_reversal_parity(self):
        configs = [
            (8, 0, (3, 5)),
            (8, 4, (3, 5)),
            (8, 2, (3, 5)),
            (12, 3, (3, 5, 7)),
        ]
        for d, antisymmetric_planes, kernels in configs:
            with self.subTest(d=d, antisymmetric_planes=antisymmetric_planes, kernels=kernels):
                torch.manual_seed(4)
                model = MLP(
                    in_channels=3,
                    d=d,
                    hidden_dim=16,
                    depth=3,
                    dropout=0.25,
                    temporal_filters=max(3, len(kernels)),
                    temporal_frontend="mixed_parity",
                    residual_kernels=kernels,
                    multiscale_symmetric_conv_layers=2,
                    antisymmetric_planes=antisymmetric_planes,
                ).eval()
                combined_in_dim = (
                    model.temporal_conv.sym_conv.out_channels
                    + model.temporal_conv.anti_conv.out_channels
                )
                if model.sym_net is not None:
                    self.assertEqual(model.sym_net[0].in_features, combined_in_dim)
                if model.anti_net is not None:
                    self.assertEqual(model.anti_net[0].in_features, combined_in_dim)
                x = torch.randn(4, 3, 29)
                y = model(x)
                y_rev = model(x.flip(-1))
                even_out_dim = model.sym_out_dim
                odd_out_dim = model.anti_out_dim
                _assert_close(y_rev[:, :even_out_dim], y[:, :even_out_dim].flip(-1))
                _assert_close(
                    y_rev[:, even_out_dim:even_out_dim + odd_out_dim],
                    -y[:, even_out_dim:even_out_dim + odd_out_dim].flip(-1),
                )

                diagnostics = model.check_time_reversal_parity(x)
                self.assertLess(diagnostics["symmetric_frontend_error"], 1e-6)
                self.assertLess(diagnostics["antisymmetric_frontend_error"], 1e-6)
                self.assertLess(diagnostics["even_output_error"], 1e-6)
                self.assertLess(diagnostics["odd_output_error"], 1e-6)

    def test_mixed_model_gradient_flow_reaches_frontend_and_projected_heads(self):
        torch.manual_seed(5)
        model = MLP(
            in_channels=2,
            d=8,
            hidden_dim=10,
            depth=2,
            dropout=0.0,
            temporal_filters=3,
            temporal_frontend="mixed_parity",
            residual_kernels=(3,),
            multiscale_symmetric_conv_layers=1,
            antisymmetric_planes=2,
        )
        x = torch.randn(4, 2, 17)
        y = model(x)
        loss = y[:, : model.sym_out_dim].pow(2).mean() + y[:, model.sym_out_dim :].pow(2).mean()
        loss.backward()

        groups = {
            "symmetric temporal kernels": [branch.conv.weight for branch in model.temporal_conv.sym_conv.temporal_branches],
            "antisymmetric temporal kernels": [branch.conv.weight for branch in model.temporal_conv.anti_conv.temporal_branches],
            "even_net parameters": list(model.parity_head.even_net.parameters()),
            "odd_net parameters": list(model.parity_head.odd_net.parameters()),
        }
        for name, params in groups.items():
            self.assertTrue(params, name)
            self.assertTrue(any(param.grad is not None for param in params), name)
            for param in params:
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
