import torch

def centered_svd_val(Z, alpha=0.001):
    """Compute the mean log singular value of a centered covariance matrix.

    This function centers the data and computes the singular value decomposition
    (SVD) of the resulting covariance matrix. It then returns the mean of the
    log singular values, regularized by `alpha`.

    Args:
        Z (torch.Tensor): A 2D tensor representing features hidden acts.
        alpha (float, optional): Regularization parameter added to the covariance matrix.
            Defaults to 0.001.

    Returns:
        float: The mean of the log singular values of the centered covariance matrix.
    """
    # assumes Z is in full precision
    J = torch.eye(Z.shape[0]) - (1 / Z.shape[0]) * torch.ones(Z.shape[0], Z.shape[0])
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0])
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore
