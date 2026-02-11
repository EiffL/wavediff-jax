"""Pure numpy Zernike polynomial generation using Noll indexing.

Reimplements the functionality of ``zernike_generator`` from the ``zernike``
library (jacopoantonello/zernike) without any external dependency on that
package.  Noll indexing convention is used throughout.

:Authors: WaveDiff-JAX contributors
"""

import numpy as np
from math import factorial


def noll_to_nm(j):
    """Convert Noll index *j* (1-based) to radial order *n* and azimuthal
    frequency *m*.

    Parameters
    ----------
    j : int
        Noll index (starting from 1).

    Returns
    -------
    n : int
        Radial order.
    m : int
        Azimuthal frequency (signed).

    References
    ----------
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence."
    *J. Opt. Soc. Am.*, 66(3), 207-211.
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")

    # Find the radial order n such that n(n+1)/2 < j <= (n+1)(n+2)/2
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1

    # 1-based position within the radial order n
    remaining = j - n * (n + 1) // 2  # ranges from 1 to n+1

    # Determine |m|.
    # Within each radial order n the |m| values are arranged as:
    #   n even:  0, 2, 2, 4, 4, ..., n, n   (n+1 entries)
    #   n odd:   1, 1, 3, 3, ..., n, n       (n+1 entries)
    if n % 2 == 0:
        abs_m = 2 * (remaining // 2)
    else:
        abs_m = 2 * ((remaining - 1) // 2) + 1

    # Sign convention: even j -> cosine (m > 0), odd j -> sine (m < 0)
    if abs_m == 0:
        m = 0
    elif j % 2 == 0:
        m = abs_m
    else:
        m = -abs_m

    return n, m


def zernike_radial(n, abs_m, rho):
    """Compute the Zernike radial polynomial R_n^{|m|}(rho).

    Parameters
    ----------
    n : int
        Radial order.
    abs_m : int
        Absolute value of the azimuthal frequency.
    rho : np.ndarray
        Radial coordinate array.

    Returns
    -------
    np.ndarray
        Radial polynomial evaluated at *rho*.
    """
    if (n - abs_m) % 2 != 0:
        return np.zeros_like(rho)

    result = np.zeros_like(rho, dtype=np.float64)
    num_terms = (n - abs_m) // 2 + 1

    for s in range(num_terms):
        coef = ((-1) ** s * factorial(n - s)
                / (factorial(s)
                   * factorial((n + abs_m) // 2 - s)
                   * factorial((n - abs_m) // 2 - s)))
        result = result + coef * rho ** (n - 2 * s)

    return result


def zernike_generator(n_zernikes, wfe_dim):
    """Generate Zernike polynomial maps using Noll indexing.

    Produces a list of 2-D arrays, one per Zernike mode from j = 1 up to
    *n_zernikes*.  Each map is evaluated on a ``[-1, 1]`` Cartesian grid of
    size ``wfe_dim x wfe_dim``.  Pixels outside the unit circle are set to
    ``NaN``.

    Parameters
    ----------
    n_zernikes : int
        Number of Zernike modes (starting from j = 1).
    wfe_dim : int
        Dimension of the output square maps.

    Returns
    -------
    list of np.ndarray
        List of 2-D arrays of shape ``(wfe_dim, wfe_dim)``.  Values outside
        the unit circle are ``NaN``.
    """
    # Build polar coordinate grid on [-1, 1]
    x = np.linspace(-1.0, 1.0, wfe_dim)
    y = np.linspace(-1.0, 1.0, wfe_dim)
    xv, yv = np.meshgrid(x, y)

    rho = np.sqrt(xv ** 2 + yv ** 2)
    theta = np.arctan2(yv, xv)

    # Mask for unit circle
    outside = rho > 1.0

    zernikes = []
    for j in range(1, n_zernikes + 1):
        n, m = noll_to_nm(j)
        abs_m = abs(m)

        # Normalization factor
        if m == 0:
            norm = np.sqrt(float(n + 1))
        else:
            norm = np.sqrt(2.0 * float(n + 1))

        # Radial part
        R = zernike_radial(n, abs_m, rho)

        # Angular part
        if m > 0:
            angular = np.cos(m * theta)
        elif m < 0:
            angular = np.sin(abs_m * theta)
        else:
            angular = np.ones_like(theta)

        Z = norm * R * angular
        Z[outside] = np.nan
        zernikes.append(Z)

    return zernikes
