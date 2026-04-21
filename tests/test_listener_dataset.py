import pytest
import torch

from scripts.datasets.listener_dataset import get_listener_loader


def test_cycle_through_all_listeners_during_batch():
    
    lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 100)])
    listener_batch_size = 4
    symmetric_listener_sampling = False

    listener_data_loader, listener_dataset = get_listener_loader(
        lstn_pos, listener_batch_size, symmetric_listener_sampling
    )

    sampled_lstn_pos = torch.empty((0, 2))  

    for listener_idx, listener_pos in listener_data_loader:
        sampled_lstn_pos = torch.cat((sampled_lstn_pos, listener_pos), dim=0)

    # sort returns (values, indices)
    sorted_vals, _ = sampled_lstn_pos.sort(dim=0)

    # comparison must use torch.equal or allclose
    assert torch.allclose(sorted_vals, lstn_pos)


def test_symmetric_sampling_during_batch():

    lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 101)])
    listener_batch_size = 10
    symmetric_listener_sampling = True

    listener_data_loader, _ = get_listener_loader(
        lstn_pos, listener_batch_size, symmetric_listener_sampling
    )

    for listener_idx, listener_pos in listener_data_loader:
        sorted_listener_pos, _ = listener_pos.sort(dim=0)

        for i in range(len(sorted_listener_pos)):
            assert torch.allclose(sorted_listener_pos[i], -1*sorted_listener_pos[-i-1])


def test_correct_idx_in_listener_batch_compared_to_origianl_array():

    lstn_pos = torch.tensor([[0.0, float(y.item())] for y in torch.linspace(-0.5, 0.5, 101)])
    listener_batch_size = 10
    symmetric_listener_sampling = False

    listener_data_loader, _ = get_listener_loader(
        lstn_pos, listener_batch_size, symmetric_listener_sampling
    )

    for listener_idx, listener_pos in listener_data_loader:
        
        assert torch.allclose(lstn_pos[listener_idx], listener_pos)
