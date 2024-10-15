from ...imports import *

class MaskingAgent:
    """
    Class for performing masking operations on embeddings.

    Methods:
    --------
    __init__():
        Initializes the MaskingAgent.

    _gather(embeddings, ids):
        Gathers embeddings based on provided indices.

    create_shuffle_ids(embeddings):
        Creates shuffled indices based on random noise for embeddings.

    shuffle_and_select(embeddings, k):
        Selects a subset of embeddings based on shuffled indices.

    unshuffle(embeddings, shuffle_ids):
        Restores the order of embeddings using the shuffled indices.
 
    pad_mask_token(embeddings, mask_token, T):
        Pads embeddings with a mask token to reach a specified length T.
    """

    def __init__(self):
        pass  # No initialization needed here for mask_token

    def _gather(self, embeddings, ids):
        """
        Gathers embeddings based on provided indices.

        Parameters:
        -----------
        embeddings : torch.Tensor
            Tensor of shape (N, T, E) where N is the batch size, T is the sequence length,
            and E is the embedding dimension.
        ids : torch.Tensor
            Tensor of shape (N, k) containing indices to gather embeddings.

        Returns:
        --------
        torch.Tensor
            Gathered embeddings of shape (N, k, E).
        """
        B, T, E = embeddings.shape
        return torch.gather(embeddings, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, E))

    def create_shuffle_ids(self, embeddings):
        """
        Creates shuffled indices based on random noise for embeddings, 
        ensuring that all zero entries (pad embeddings) come at the end.

        Parameters:
        -----------
        embeddings : torch.Tensor
            Tensor of shape (N, T, E) where N is the batch size, T is the sequence length,
            and E is the embedding dimension.

        Returns:
        --------
        torch.Tensor
            Shuffled indices of shape (N, T) based on random noise, with zero entries at the end.
        """
        B, T, E = embeddings.shape

        # Calculate the mask where embeddings are all zeros
        zero_mask = (embeddings == 0).all(dim=-1)
        
        # Generate random noise for non-zero entries, and high values for zero entries
        noise = torch.rand(B, T, device=embeddings.device)
        noise[zero_mask] = float('inf')  # Assign high noise to zero entries to push them to the end
        
        # Get the shuffled indices based on the noise
        ids = torch.argsort(noise, dim=1)
        
        return ids

    def shuffle_and_select(self, embeddings, k):
        """
        Selects a subset of embeddings based on shuffled indices.

        Parameters:
        -----------
        embeddings : torch.Tensor
            Tensor of shape (N, T, E) where N is the batch size, T is the sequence length,
            and E is the embedding dimension.
        k : int
            Number of embeddings to keep after shuffling.

        Returns:
        --------
        torch.Tensor
            Shortlisted embeddings of shape (N, k, E).
        torch.Tensor
            Shuffled indices of shape (N, T) used for shortlisting.
        """
        shuffle_ids = self.create_shuffle_ids(embeddings)
        keep_ids = shuffle_ids[:, :k]
        shortlisted_embeddings = self._gather(embeddings, keep_ids)
        return shortlisted_embeddings, shuffle_ids

    def unshuffle(self, embeddings,is_pad_token_mask, shuffle_ids):
        """
        Restores the order of embeddings using the shuffled indices.

        Parameters:
        -----------
        embeddings : torch.Tensor
            Tensor of shape (N, T, E) where N is the batch size, T is the sequence length,
            and E is the embedding dimension.
        shuffle_ids : torch.Tensor
            Shuffled indices of shape (N, T) used for shuffling.

        Returns:
        --------
        torch.Tensor
            Unshuffled embeddings of shape (N, T, E).
        """
        assert embeddings.shape[1] == shuffle_ids.shape[1], "embeddings and shuffle_ids must have same length in time dim"
        restore_ids = torch.argsort(shuffle_ids, dim=1)
        unshuffled_embeddings = self._gather(embeddings, restore_ids)
        is_pad_token_mask = self._gather(is_pad_token_mask.unsqueeze(-1), restore_ids).squeeze(-1)
        return unshuffled_embeddings, is_pad_token_mask

    def pad_mask_token(self, embeddings, mask_token, T):
        """
        Pads embeddings with a mask token to reach a specified length T.

        Parameters:
        -----------
        embeddings : torch.Tensor
            Tensor of shape (N, T, E) where N is the batch size, T is the sequence length,
            and E is the embedding dimension.
        mask_token : torch.Tensor
            Tensor of shape (1, 1, E) representing the mask token to pad with.
        T : int
            Desired length to pad the embeddings to.

        Returns:
        --------
        torch.Tensor
            Padded embeddings of shape (N, T, E).
        """
        B = embeddings.shape[0]
        num_pads = T - embeddings.shape[1]
        mask_tokens = mask_token.repeat(B, num_pads, 1)
        embeddings = torch.cat([embeddings, mask_tokens], dim=1)
        is_pad_token_mask = torch.zeros(B, T, dtype=torch.bool, device=embeddings.device)
        is_pad_token_mask[:, -num_pads:] = True
        return embeddings, is_pad_token_mask