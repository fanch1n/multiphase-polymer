from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import minimize
import numpy as np

def density_filter(x, threshold=0.1):
    return np.heaviside(np.array(x) - threshold, 1)

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def one_runs(a):
    # Create an array that is 2 where a is 1, and pad each end with an extra 2.
    isone = np.concatenate(([2], np.equal(a, 1).view(np.int8), [2]))
    absdiff = np.abs(np.diff(isone))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def find_condensed_phase_edges(phi, bins):
    '''
    assumes box with one connected condensed region in contact with a dilute phase
    (at most two regions due to the PBC)
    '''
    a = density_filter(phi, 0.1)
    condensed_region = one_runs(a)
    dilute_region = zero_runs(a)
    if len(condensed_region) + len(dilute_region) > 3:
        left = right = None
        print("WARNING: disconnected condensed region! Following analysis does not apply!")
    elif len(condensed_region) + len(dilute_region) < 3:
        if len(dilute_region) == 0:
            print("WARNING: no dilute phase")
        elif len(condensed_region) == 1:
            left, right = condensed_region[0]
    else:
        # two interfaces
        if len(condensed_region) == 2:
            region1, region2 = condensed_region[0], condensed_region[1]
        # check if region1 and region2 are connected
            assert (region2[-1] + 1) % len(a) <= region1[1], f'disconnected condensed region!'
            left = region2[0]
            right = region1[1]
        elif len(condensed_region) == 1:
            left, right = condensed_region[0]
    left_edge, right_edge = None, None
    if left:
        left_edge = bins[left]
    if right:
        right_edge = bins[right]
    if not left_edge:
        left_edge = bins[0]
    if not right_edge:
        right_edge = bins[-1]

    return left_edge, right_edge

def locate_interface(density, x, start, end):
    xsmooth = np.arange(x.min(), x.max(), 0.1)
    cs = CubicSpline(x, density)
    def objective(mid, intp_density, x):
        a = quad(intp_density,start, mid)
        b = quad(intp_density,mid, end)
        diff = np.abs(a[0]-b[0])
        return diff
    mid = (start + end) / 2.
    res = minimize(objective, mid, args=(cs, xsmooth))
    return res.x

#def get_shiftProfile(data, bins, target_center_loc=None, search_width=5.):
#    '''
#    data: 2 x N numpy array, the binned profile along z
#    1st col: profile for the alpha phase, 2nd col: profile for the beta phase
#    '''
#    v0 = np.pi/6
#    phi = data[:, -1]/v0
#    opa, opb = data[:, 0], data[:, 1]
#    index_mask = np.arange(0, len(bins), 1)
#
#    left_edge, right_edge = find_condensed_phase_edges(phi, bins)
#    start = right_edge - search_width
#    end = right_edge + search_width
#    loc = locate_interface(phi, bins, start, end)
#    loc_index = int(loc/bin_width)
#
#    if target_center_loc is None:
#        target_center_loc = (bins[-1] - bins[0]) / 2.
#
#    dshift = target_center_pos - loc
#    dmove = int(dshift / bin_width)
#    shift_bins =  bins + dmove * bin_width
#    rotate = unwrap_indx(index_mask, len(bins)-dmove)
#    shift_bins = unwrap_z(shift_bins , max(bins))
#    shifted_a = opa[rotate]
#    shifted_b = opb[rotate]
#
#    return shifted_a, shifted_b, shift_bins


def get_shiftProfile(data, bins, target_center_loc=None, style='mid', search_width=5.):
    '''
    data: 2 x N numpy array, the binned profile along z
    1st col: profile for the alpha phase, 2nd col: profile for the beta phase
    '''
    v0 = np.pi/6
    phi = data[:, -1]/v0
    opa, opb = data[:, 0], data[:, 1]
    index_mask = np.arange(0, len(bins), 1)
    left_edge, right_edge = find_condensed_phase_edges(phi, bins)

    if style == 'mid':
        lft = locate_interface(phi, bins, left_edge-search_width, left_edge+search_width)
        lrt = locate_interface(phi, bins, left_edge-search_width, left_edge+search_width)
        dshift = target_center_pos - (loc_left + loc_right)/2.
    else:
        loc = locate_interface(phi, bins, right_edge-search_width, right_edge+search_width)
        dshift = target_center_pos - loc

    if target_center_loc is None:
        target_center_loc = (bins[-1] - bins[0]) / 2.

    dmove = int(dshift / bin_width)
    shift_bins =  bins + dmove * bin_width
    rotate = unwrap_indx(index_mask, len(bins)-dmove)
    shift_bins = unwrap_z(shift_bins , max(bins))
    shifted_a = opa[rotate]
    shifted_b = opb[rotate]

    return shifted_a, shifted_b, shift_bins

def get_shiftAverage(list_data, bins):
    list_a, list_b = [], []
    index_mask = np.arange(0, len(bins), 1)

    for i in range(len(list_data)):
        a, b, nbins = get_shiftProfile(list_data[i], bins)
        list_a.append(a)
        list_b.append(b)
    return np.average(np.array(list_a), axis=0), np.average(np.array(list_b), axis=0)

def unwrap_z(pos, Lz):
    return pos - np.floor(pos / Lz) * Lz

def unwrap_indx(index, mid):
    return list(index[mid:]) + list(index[:mid])

